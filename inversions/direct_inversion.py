import torch
from typing import Optional, Tuple, Any, List
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline

from .base_inverter import BaseInverter


class DirectInverter(BaseInverter):
    """
    Реализация метода Direct Inversion для SDXL.
    Сохраняет траекторию инверсии и использует её для направленной генерации.
    """

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        super().__init__(pipeline)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config)
        self.forward_scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

    def invert(
            self,
            image: Image.Image,
            prompt: str,
            num_steps: int = 50,
            guidance_scale: float = 1.0,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:

        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.inverse_scheduler

        print("[Direct Inversion] Извлечение опорной траектории (без оптимизации)...")
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            generator = torch.Generator(device=self.device).manual_seed(42)
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.sample(generator=generator)
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

        self.pipeline.scheduler = original_scheduler
        return latent_noise, trajectory

    def reconstruct(
            self,
            latent_noise: torch.Tensor,
            prompt: str,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            context: Optional[List[torch.Tensor]] = None,
            alpha: float = 0.5,  # Сила смешивания с оригиналом (0.0 - нет смешивания, 1.0 - полная копия)
            blend_threshold: float = 0.7,  # Доля шагов (от начала), на которых применяется смешивание
            **kwargs
    ) -> Image.Image:

        if context is None:
            print("[Direct Inversion] Внимание: контекст не передан, выполняется обычный DDIM.")
            return super().reconstruct(latent_noise, prompt, num_steps, guidance_scale, **kwargs)

        print(f"[Direct Inversion] Направленная реконструкция (alpha={alpha}, threshold={blend_threshold})...")

        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.forward_scheduler
        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps

        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipeline.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True
            )

            embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

            h, w = latent_noise.shape[-2] * 8, latent_noise.shape[-1] * 8
            time_ids = self.pipeline._get_add_time_ids(
                (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
            ).to(self.device)
            time_ids_input = torch.cat([time_ids, time_ids])

            latents = latent_noise.clone()

            # Определяем, на каком шаге мы должны перестать подмешивать оригинальную структуру
            max_blend_step = int(num_steps * blend_threshold)

            for i, t in enumerate(timesteps):
                source_latents = context[i]

                latent_input = torch.cat([latents] * 2)
                latent_input = self.pipeline.scheduler.scale_model_input(latent_input, t)

                noise_pred = self.pipeline.unet(
                    latent_input, t, encoder_hidden_states=embeds_input,
                    added_cond_kwargs={"text_embeds": pooled_input, "time_ids": time_ids_input}
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample

                # Магия умного смешивания: подмешиваем оригинал только на ранних шагах генерации
                if i < max_blend_step:
                    latents = alpha * source_latents + (1 - alpha) * latents

            image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
            image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

        self.pipeline.scheduler = original_scheduler
        return image