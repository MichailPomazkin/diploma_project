import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Tuple, List
from PIL import Image

# Добавили TF для преобразования PIL-картинки (маски) в тензор PyTorch
import torchvision.transforms.functional as TF

from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline
from .base_inverter import BaseInverter


class NullTextInverter(BaseInverter):
    """
    Null-text Inversion с поддержкой пространственного маскирования.
    Оптимизирует unconditional эмбеддинги, чтобы при высоком CFG сохранять фон.
    """

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        super().__init__(pipeline)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config)
        self.forward_scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        with torch.no_grad():
            self.empty_embeds, _, self.empty_pooled, _ = self.pipeline.encode_prompt(
                prompt="", device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

    def invert(
            self,
            image: Image.Image,
            prompt: str,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            num_inner_steps: int = 5,
            learning_rate: float = 1e-3,
            # Добавили параметры для передачи маски
            use_spatial_mask: bool = False,
            mask: Optional[Image.Image] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Основной метод инверсии.
        Если use_spatial_mask=True, передаётся маска (белое – изменяемая область, чёрное – фон).
        """

        if guidance_scale <= 1.0:
            print(
                "[Null-text] Внимание: guidance_scale <= 1.0. При таком значении инверсия точна и без оптимизации, метод излишен.")

        print("[Null-text] Этап 1: получение эталонной DDIM-траектории...")
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.mode()
            latents = latents * self.pipeline.vae.config.scaling_factor

            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

            h, w = image_tensor.shape[-2:]
            time_ids = self.pipeline._get_add_time_ids(
                (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
            ).to(self.device)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

        self.inverse_scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.inverse_scheduler.timesteps

        trajectory = [latents.clone()]
        current_latents = latents.clone()
        with torch.no_grad():
            for t in timesteps:
                noise_pred = self.pipeline.unet(
                    current_latents, t, encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                current_latents = self.inverse_scheduler.step(noise_pred, t, current_latents).prev_sample
                trajectory.append(current_latents.clone())

        trajectory = list(reversed(trajectory))
        latent_noise = trajectory[0]

        # Подготовка маски перед циклом, чтобы не делать ресайз на каждой итерации (экономит время)
        prepared_mask_latent = None
        if use_spatial_mask:
            if mask is None:
                print("[Null-text] Предупреждение: use_spatial_mask=True, но маска не передана. Работаем без маски.")
            else:
                mask_tensor = TF.to_tensor(mask).to(self.device)

                # Оставляем один канал, если маска цветная
                if mask_tensor.shape[0] > 1:
                    mask_tensor = mask_tensor[0:1]

                # Инвертируем маску (фон=1, объект=0). Это нужно, чтобы при умножении
                # ошибка на объекте обнулялась, и оптимизатор игнорировал эту область.
                mask_bg = 1.0 - mask_tensor
                mask_bg = mask_bg.unsqueeze(0)

                # Сжимаем маску до размеров латентного пространства (обычно 128x128).
                # mode='nearest' сохраняет маску строгой (только 0 и 1, без серых пикселей).
                h_lat, w_lat = latent_noise.shape[-2:]
                prepared_mask_latent = F.interpolate(mask_bg, size=(h_lat, w_lat), mode='nearest')

                # Копируем маску на 4 канала, так как у латентов 4 канала
                prepared_mask_latent = prepared_mask_latent.expand(-1, latent_noise.shape[1], -1, -1)
                print("[Null-text] Маска загружена и масштабирована для латентного пространства.")

        print(f"[Null-text] Этап 2: градиентная оптимизация (Adam, {num_inner_steps} итераций/шаг)...")
        self.forward_scheduler.set_timesteps(num_steps, device=self.device)
        forward_timesteps = self.forward_scheduler.timesteps

        optimized_uncond_embeddings = []
        current_latent = latent_noise.clone().detach()

        for i, t in enumerate(forward_timesteps):
            target_latent = trajectory[i + 1].detach()
            uncond_embeds_opt = self.empty_embeds.clone().detach().to(torch.float32).requires_grad_(True)
            optimizer = Adam([uncond_embeds_opt], lr=learning_rate)
            pred_latent = None

            for inner_step in range(num_inner_steps):
                optimizer.zero_grad()
                uncond_fp16 = uncond_embeds_opt.to(dtype=prompt_embeds.dtype)

                latent_input = torch.cat([current_latent] * 2)
                latent_input = self.forward_scheduler.scale_model_input(latent_input, t)

                embeds_input = torch.cat([uncond_fp16, prompt_embeds])
                pooled_input = torch.cat([self.empty_pooled, pooled_prompt_embeds])
                time_ids_input = torch.cat([time_ids, time_ids])

                noise_pred = self.pipeline.unet(
                    latent_input, t, encoder_hidden_states=embeds_input,
                    added_cond_kwargs={"text_embeds": pooled_input, "time_ids": time_ids_input}
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                pred_latent = self.forward_scheduler.step(noise_pred_cfg, t, current_latent).prev_sample

                # Расчет функции потерь
                if use_spatial_mask and prepared_mask_latent is not None:
                    # Вычисляем разницу и умножаем на маску. Ошибка на объекте становится равной 0.
                    # Это заставляет алгоритм подгонять эмбеддинги только под фон.
                    diff = pred_latent.float() - target_latent.float()
                    masked_diff = diff * prepared_mask_latent
                    loss = (masked_diff ** 2).mean()
                else:
                    # Обычный расчет ошибки по всей картинке
                    loss = F.mse_loss(pred_latent.float(), target_latent.float())

                if torch.isnan(loss) or torch.isnan(pred_latent).any() or torch.isinf(pred_latent).any():
                    print(f"  [Null-text] NaN/Inf на шаге {i}, итерация {inner_step}. Прерываем оптимизацию.")
                    pred_latent = target_latent
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_([uncond_embeds_opt], 1.0)
                optimizer.step()
                with torch.no_grad():
                    uncond_embeds_opt.clamp_(-10.0, 10.0)

            optimized_uncond_embeddings.append(uncond_embeds_opt.detach().to(dtype=prompt_embeds.dtype))

            if pred_latent is not None:
                mse_error = F.mse_loss(pred_latent.float(), target_latent.float()).item()
                if mse_error > 1.0:
                    current_latent = target_latent.clone()
                else:
                    current_latent = pred_latent.detach()
            else:
                current_latent = target_latent.clone()

        return latent_noise, optimized_uncond_embeddings

    def reconstruct(
            self,
            latent_noise: torch.Tensor,
            prompt: str,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            context: Optional[List[torch.Tensor]] = None,
            **kwargs
    ) -> Image.Image:
        """Стандартный метод реконструкции – без изменений, маска здесь не нужна."""
        if context is None:
            print("[Null-text] Внимание: контекст не передан, выполняется обычный цикл генерации.")
        elif len(context) != num_steps:
            raise ValueError(f"Длина контекста ({len(context)}) должна равняться num_steps ({num_steps})")

        print("[Null-text] Восстанавливаем с использованием оптимизированного контекста...")

        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.forward_scheduler

        try:
            self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
            timesteps = self.pipeline.scheduler.timesteps
            do_classifier_free_guidance = guidance_scale > 1.0

            with torch.no_grad():
                prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                    prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                )

                vae_scale_factor = getattr(self.pipeline, "vae_scale_factor", 8)
                h, w = latent_noise.shape[-2] * vae_scale_factor, latent_noise.shape[-1] * vae_scale_factor
                time_ids = self.pipeline._get_add_time_ids(
                    (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
                ).to(self.device)

                latents = latent_noise.clone()

                for i, t in enumerate(timesteps):
                    uncond_emb = context[i] if context else self.empty_embeds

                    if do_classifier_free_guidance:
                        latent_input = torch.cat([latents] * 2)
                    else:
                        latent_input = latents
                    latent_input = self.pipeline.scheduler.scale_model_input(latent_input, t)

                    if do_classifier_free_guidance:
                        embeds_input = torch.cat([uncond_emb, prompt_embeds])
                        pooled_input = torch.cat([self.empty_pooled, pooled_prompt_embeds])
                        time_ids_input = torch.cat([time_ids, time_ids])
                    else:
                        embeds_input = prompt_embeds
                        pooled_input = pooled_prompt_embeds
                        time_ids_input = time_ids

                    noise_pred = self.pipeline.unet(
                        latent_input, t, encoder_hidden_states=embeds_input,
                        added_cond_kwargs={"text_embeds": pooled_input, "time_ids": time_ids_input}
                    ).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample

                image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[
                    0]
                image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

            return image

        finally:
            self.pipeline.scheduler = original_scheduler