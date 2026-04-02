import torch
from typing import Optional, Tuple, Any
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline

from .base_inverter import BaseInverter


class DDIMInverter(BaseInverter):
    """
    Реализация DDIM инверсии для SDXL.
    """

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        super().__init__(pipeline)

        # Специфичные планировщики для DDIM
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
        )
        self.forward_scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
        )

    def invert(
            self,
            image: Image.Image,
            prompt: str,
            num_steps: int = 50,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        print("[DDIM] Кодируем изображение в латентный шум...")
        self.pipeline.scheduler = self.inverse_scheduler

        # 1. Чистый препроцессинг через метод базового класса!
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            # 2. VAE Кодирование (Пиксели в латенты)
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor

            # 3. Кодирование текста (для SDXL нужно два вида эмбеддингов)
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            time_ids = self.pipeline._get_add_time_ids(
                (1024, 1024), (0, 0), (1024, 1024),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
            ).to(self.device)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps

        inverted_latents = latents.clone()

        # 4. Обратный цикл диффузии
        with torch.no_grad():
            for t in timesteps:
                noise_pred = self.pipeline.unet(
                    inverted_latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                inverted_latents = self.pipeline.scheduler.step(
                    noise_pred,
                    t,
                    inverted_latents
                ).prev_sample

        # DDIM не оптимизирует эмбеддинги, поэтому контекст None
        return inverted_latents, None

    def reconstruct(
            self,
            latent_noise: torch.Tensor,
            prompt: str,
            num_steps: int = 50,
            guidance_scale: float = 1.0,
            **kwargs
    ) -> Image.Image:
        print("[DDIM] Восстанавливаем из шума...")
        self.pipeline.scheduler = self.forward_scheduler

        # Вытаскиваем контекст, если он есть
        context = kwargs.get("context", None)

        # Вызываем базовый процесс генерации из BaseInverter
        return super().reconstruct(
            latent_noise=latent_noise,
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            context=context
        )