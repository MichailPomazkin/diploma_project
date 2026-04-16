import torch
from typing import Optional, Tuple, Any
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline

from .base_inverter import BaseInverter


class DDIMInverter(BaseInverter):
    """
    DDIM инверсия для SDXL.

    Использует детерминистический обратный процесс для получения латентного шума из изображения.
    """

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        super().__init__(pipeline)

        # Создаём копии шедулеров из конфигурации текущего пайплайна,
        # чтобы не загружать их повторно из интернета.
        self.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
        self.forward_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    def invert(
            self,
            image: Image.Image,
            prompt: str,
            num_steps: int = 50,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Преобразует изображение в латентный шум через DDIM инверсию.
        """
        print("[DDIM] Кодируем изображение в латентный шум...")

        # Предобработка изображения (нормализация, приведение к тензору)
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            # Кодирование в латентное пространство VAE
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor

            # Кодирование текстового промпта для SDXL
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            h, w = image_tensor.shape[-2:]
            time_ids = self.pipeline._get_add_time_ids(
                (h, w), (0, 0), (h, w),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
            ).to(self.device)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

        # Настройка временных шагов для обратного процесса
        self.inverse_scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.inverse_scheduler.timesteps

        inverted_latents = latents.clone()

        # Обратный процесс: от изображения к шуму
        with torch.no_grad():
            for t in timesteps:
                noise_pred = self.pipeline.unet(
                    inverted_latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                inverted_latents = self.inverse_scheduler.step(
                    noise_pred,
                    t,
                    inverted_latents
                ).prev_sample

        # DDIM не оптимизирует эмбеддинги, поэтому контекст не возвращается
        return inverted_latents, None

    def reconstruct(
            self,
            latent_noise: torch.Tensor,
            prompt: str,
            num_steps: int = 50,
            guidance_scale: float = 1.0,
            **kwargs
    ) -> Image.Image:
        """
        Восстанавливает изображение из латентного шума с помощью DDIM.
        """
        print("[DDIM] Восстанавливаем из шума...")

        context = kwargs.get("context", None)

        # Временно подменяем шедулер пайплайна на прямой DDIM-шедулер,
        # так как базовый метод reconstruct использует self.pipeline.scheduler.
        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.forward_scheduler

        try:
            result = super().reconstruct(
                latent_noise=latent_noise,
                prompt=prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                context=context
            )
            return result
        finally:
            # Восстанавливаем исходный шедулер, чтобы не повлиять на другие операции
            self.pipeline.scheduler = original_scheduler