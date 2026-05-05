import torch
import torchvision
import os
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Tuple, List
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from scipy import ndimage

from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline
from .base_inverter import BaseInverter
from .attention_utils import CrossAttentionManager


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
            self.empty_embeds = self.empty_embeds.to(self.device)
            self.empty_pooled = self.empty_pooled.to(self.device)

        self.attn_manager = CrossAttentionManager(self.pipeline.unet)

    def invert(
            self,
            image: Image.Image,
            prompt: str,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            num_inner_steps: int = 5,
            learning_rate: float = 1e-3,
            use_spatial_mask: bool = False,
            mask: Optional[Image.Image] = None,
            token_index: Optional[int] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        if guidance_scale <= 1.0:
            print("[Null-text] Внимание: guidance_scale <= 1.0. Инверсия точна и без оптимизации, метод излишен.")

        print("[Null-text] Этап 1: получение эталонной DDIM-траектории...")
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.mode()
            latents = latents * self.pipeline.vae.config.scaling_factor
            latents = latents.to(self.device)

            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            prompt_embeds = prompt_embeds.to(self.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)

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

        print(f"[Null-text] Этап 2: градиентная оптимизация (Adam, {num_inner_steps} итераций/шаг)...")
        self.forward_scheduler.set_timesteps(num_steps, device=self.device)
        forward_timesteps = self.forward_scheduler.timesteps

        prepared_mask_latent = None
        if use_spatial_mask:
            if mask is not None:
                mask_tensor = TF.to_tensor(mask).to(self.device)
                if mask_tensor.shape[0] > 1:
                    mask_tensor = mask_tensor[0:1]
                mask_bg = 1.0 - mask_tensor
                mask_bg = mask_bg.unsqueeze(0)
                h_lat, w_lat = latent_noise.shape[-2:]
                prepared_mask_latent = F.interpolate(mask_bg, size=(h_lat, w_lat), mode='nearest')
                prepared_mask_latent = prepared_mask_latent.expand(-1, latent_noise.shape[1], -1, -1)
                print("[Null-text] Внешняя маска загружена и масштабирована.")

            elif token_index is not None and token_index != -1:
                print(f"[Null-text] Генерируем маску из Cross-Attention для токена №{token_index} на 25-м шаге...")
                self.attn_manager.attach()

                step_idx = min(25, len(forward_timesteps) - 1)
                t_dummy = forward_timesteps[step_idx]
                latent_scaled = self.forward_scheduler.scale_model_input(trajectory[step_idx].clone(), t_dummy)

                with torch.no_grad():
                    _ = self.pipeline.unet(
                        latent_scaled, t_dummy, encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
                    )
                    h_lat, w_lat = latent_noise.shape[-2:]

                    raw_mask = self.attn_manager.get_mask_for_token(
                        token_index=token_index, threshold=0.15, resolution=h_lat, device=self.device
                    )

                    # --- ИСПРАВЛЕННАЯ ОЧИСТКА МАСКИ ---
                    # 1. Ищем черные пиксели (объект)
                    object_pixels = raw_mask.cpu().numpy() < 0.5

                    # 2. Раздуваем (Dilation) для склейки разрывов
                    object_pixels = ndimage.binary_dilation(object_pixels, iterations=2)

                    # 3. Ищем самый большой компонент
                    labeled_mask, num_features = ndimage.label(object_pixels)
                    if num_features > 0:
                        sizes = ndimage.sum(object_pixels, labeled_mask, range(1, num_features + 1))
                        largest_label = np.argmax(sizes) + 1
                        clean_object = (labeled_mask == largest_label)
                    else:
                        clean_object = object_pixels

                    # 4. Заливаем дырки внутри машины/объекта
                    clean_object = ndimage.binary_fill_holes(clean_object).astype(np.float32)

                    # 5. Проверяем покрытие (защита от "черного квадрата" на цветах)
                    coverage = clean_object.mean()
                    print(f"  [Отладка] Покрытие объекта: {coverage * 100:.1f}%")

                    if coverage > 0.8:
                        print("  [ВНИМАНИЕ] Объект занял весь экран. Отключаем маску.")
                        use_spatial_mask = False
                        prepared_mask_latent = None
                    else:
                        # 6. Формируем итоговую маску (Фон = 1.0, Объект = 0.0)
                        bg_mask_np = 1.0 - clean_object
                        clean_mask_tensor = torch.from_numpy(bg_mask_np).to(self.device)

                        save_dir = "/content/debug_masks_final"
                        os.makedirs(save_dir, exist_ok=True)
                        safe_img_id = kwargs.get('image_id', 'debug')

                        torchvision.utils.save_image(
                            clean_mask_tensor.unsqueeze(0).cpu(),
                            os.path.join(save_dir, f"img_{safe_img_id}_MASK.png")
                        )

                        bg_mask = clean_mask_tensor

                self.attn_manager.detach()

                if use_spatial_mask:
                    prepared_mask_latent = bg_mask.unsqueeze(0).unsqueeze(0)
                    prepared_mask_latent = prepared_mask_latent.expand(-1, latent_noise.shape[1], -1, -1)
                    print("[Null-text] Очищенная боевая маска успешно сгенерирована!")

            else:
                print(f"[Null-text] Неверный токен (index: {token_index}). Маска отключена.")
                use_spatial_mask = False
                prepared_mask_latent = None

        if use_spatial_mask and prepared_mask_latent is not None:
            # ИСПРАВЛЕНИЕ ТИПОВ ДАННЫХ
            self.spatial_mask = prepared_mask_latent.detach().to(latent_noise.dtype)
            self.original_trajectory = [lat.detach().clone().to(latent_noise.dtype) for lat in trajectory]
        else:
            self.spatial_mask = None
            self.original_trajectory = None

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

                latent_scaled = self.forward_scheduler.scale_model_input(current_latent, t)

                noise_pred_uncond = self.pipeline.unet(
                    latent_scaled, t, encoder_hidden_states=uncond_fp16,
                    added_cond_kwargs={"text_embeds": self.empty_pooled, "time_ids": time_ids}
                ).sample

                with torch.no_grad():
                    noise_pred_text = self.pipeline.unet(
                        latent_scaled, t, encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
                    ).sample

                noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                pred_latent = self.forward_scheduler.step(noise_pred_cfg, t, current_latent).prev_sample

                if use_spatial_mask and prepared_mask_latent is not None:
                    diff = pred_latent.float() - target_latent.float()
                    mask_float = prepared_mask_latent.float()
                    masked_diff = diff * mask_float
                    loss = (masked_diff ** 2).sum() / (mask_float.sum() + 1e-8)
                else:
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
                if use_spatial_mask and prepared_mask_latent is not None:
                    diff = pred_latent.float() - target_latent.float()
                    mask_float = prepared_mask_latent.float()
                    masked_diff = diff * mask_float
                    mse_error = ((masked_diff ** 2).sum() / (mask_float.sum() + 1e-8)).item()
                else:
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

                prompt_embeds = prompt_embeds.to(self.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)

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

                    # ИСПРАВЛЕНИЕ ТИПОВ ДАННЫХ ПРИ СЛИЯНИИ
                    if hasattr(self, 'spatial_mask') and self.spatial_mask is not None:
                        target_latent = self.original_trajectory[i + 1].to(device=self.device, dtype=latents.dtype)
                        mask = self.spatial_mask.to(device=self.device, dtype=latents.dtype)
                        latents = latents * (1.0 - mask) + target_latent * mask

                image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[
                    0]
                image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

            return image

        finally:
            self.pipeline.scheduler = original_scheduler