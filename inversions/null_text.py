import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Tuple, Any, List
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline

from .base_inverter import BaseInverter


class NullTextInverter(BaseInverter):
    """
    Реализация метода Null-text Inversion для SDXL.
    Алгоритм оптимизирует безусловные текстовые эмбеддинги (пустой промпт) на каждом шаге генерации,
    чтобы минимизировать ошибку восстановления изображения при использовании Classifier-Free Guidance (CFG > 1.0).
    """

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        super().__init__(pipeline)

        # Инициализируем планировщики на основе конфигурации основного пайплайна
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config)
        self.forward_scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        # Заранее вычисляем и кешируем эмбеддинги пустого промпта
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
            num_inner_steps: int = 10,
            learning_rate: float = 1e-2,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:

        self.opt_guidance_scale = guidance_scale
        # Временно сохраняем оригинальный планировщик
        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.inverse_scheduler

        print("[Null-text] Этап 1: Извлечение опорной DDIM-траектории...")
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            # Кодируем изображение в латентное пространство VAE
            generator = torch.Generator(device=self.device).manual_seed(42)
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.sample(generator=generator)
            latents = latents * self.pipeline.vae.config.scaling_factor

            # Кодируем целевой текстовый промпт
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

            # Динамически получаем размер из входного тензора
            h, w = image_tensor.shape[-2:]

            time_ids = self.pipeline._get_add_time_ids(
                (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
            ).to(self.device)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps

        trajectory = [latents.clone()]
        current_latents = latents.clone()

        with torch.no_grad():
            for t in timesteps:
                noise_pred = self.pipeline.unet(
                    current_latents, t, encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                current_latents = self.pipeline.scheduler.step(noise_pred, t, current_latents).prev_sample
                trajectory.append(current_latents.clone())

        trajectory = list(reversed(trajectory))
        latent_noise = trajectory[0]

        print(f"[Null-text] Этап 2: Градиентная оптимизация (Adam, {num_inner_steps} итераций/шаг)...")
        self.pipeline.scheduler = self.forward_scheduler
        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        forward_timesteps = self.pipeline.scheduler.timesteps

        optimized_uncond_embeddings = []
        current_latent = latent_noise.clone().detach()

        for i, t in enumerate(forward_timesteps):
            target_latent = trajectory[i + 1].detach()

            uncond_embeds_opt = self.empty_embeds.clone().detach().requires_grad_(True)
            optimizer = Adam([uncond_embeds_opt], lr=learning_rate)

            for _ in range(num_inner_steps):
                optimizer.zero_grad()

                latent_input = torch.cat([current_latent] * 2)
                latent_input = self.pipeline.scheduler.scale_model_input(latent_input, t)

                embeds_input = torch.cat([uncond_embeds_opt, prompt_embeds])
                pooled_input = torch.cat([self.empty_pooled, pooled_prompt_embeds])
                time_ids_input = torch.cat([time_ids, time_ids])

                noise_pred = self.pipeline.unet(
                    latent_input, t, encoder_hidden_states=embeds_input,
                    added_cond_kwargs={"text_embeds": pooled_input, "time_ids": time_ids_input}
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                pred_latent = self.pipeline.scheduler.step(noise_pred_cfg, t, current_latent).prev_sample
                loss = F.mse_loss(pred_latent.float(), target_latent.float())

                loss.backward()
                optimizer.step()

            optimized_uncond_embeddings.append(uncond_embeds_opt.detach())

            with torch.no_grad():
                current_latent = target_latent.clone()

        self.pipeline.scheduler = original_scheduler
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
        """
        Восстанавливает изображение, используя классический цикл диффузии SDXL,
        но подменяя пустые эмбеддинги на оптимизированные на каждом шаге.
        """
        print("[Null-text] Восстанавливаем с использованием оптимизированного контекста...")

        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.forward_scheduler
        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps

        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

            # Разрешение берем из размера латентного шума (умножаем на 8, так как VAE сжимает в 8 раз)
            h, w = latent_noise.shape[-2] * 8, latent_noise.shape[-1] * 8
            time_ids = self.pipeline._get_add_time_ids(
                (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
            ).to(self.device)

            latents = latent_noise.clone()

            # Ручной цикл генерации SDXL
            for i, t in enumerate(timesteps):
                # Достаем оптимизированный эмбеддинг
                uncond_emb = context[i] if context else self.empty_embeds

                latent_input = torch.cat([latents] * 2)
                latent_input = self.pipeline.scheduler.scale_model_input(latent_input, t)

                embeds_input = torch.cat([uncond_emb, prompt_embeds])
                pooled_input = torch.cat([self.empty_pooled, pooled_prompt_embeds])
                time_ids_input = torch.cat([time_ids, time_ids])

                noise_pred = self.pipeline.unet(
                    latent_input, t, encoder_hidden_states=embeds_input,
                    added_cond_kwargs={"text_embeds": pooled_input, "time_ids": time_ids_input}
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample

            # Декодируем латенты в пиксели через VAE и встроенный процессор
            image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
            image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

        self.pipeline.scheduler = original_scheduler
        return image