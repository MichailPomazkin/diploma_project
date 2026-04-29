import torch
from typing import Optional, Tuple, List
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline

from .base_inverter import BaseInverter


class DirectInverter(BaseInverter):
    """
    Реализация метода Direct Inversion для модели Stable Diffusion XL (SDXL).

    Принцип работы:
        1. На этапе invert выполняется обратный процесс: от исходного латента z0
           к зашумлённому zT. Все промежуточные латенты сохраняются в траекторию.
        2. На этапе reconstruct при генерации нового изображения на первых шагах
           (обычно до 70% процесса) выполняется линейное смешивание текущего латента
           с сохранённым из траектории. Это обеспечивает сохранение композиции
           и геометрии исходного изображения при изменении текстового описания.
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
            mask: Optional[Image.Image] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Выполняет обратный процесс: преобразует входное изображение в шум
        и возвращает полную траекторию промежуточных латентов.
        """
        print("[Direct Inversion] Извлечение опорной траектории...")

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

        return latent_noise, trajectory

    def reconstruct(
            self,
            latent_noise: torch.Tensor,
            prompt: str,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            context: Optional[List[torch.Tensor]] = None,
            alpha: float = 0.5,
            blend_threshold: float = 0.7,
            mask: Optional[Image.Image] = None,
            **kwargs
    ) -> Image.Image:
        """
        Выполняет прямой процесс генерации изображения с использованием опорной траектории.
        """
        if context is None:
            print("[Direct Inversion] Траектория не найдена. Выполняется стандартная DDIM-генерация.")
            return super().reconstruct(latent_noise, prompt, num_steps, guidance_scale, **kwargs)

        if len(context) != num_steps + 1:
            raise ValueError(
                f"Несоответствие размерности: длина контекста ({len(context)}) "
                f"должна быть равна количеству шагов плюс один ({num_steps + 1})."
            )

        print(f"[Direct Inversion] Запуск направленной генерации. Коэффициент смешивания: {alpha}, "
              f"длительность смешивания: {blend_threshold}.")

        # Явный перенос шума на GPU
        latent_noise = latent_noise.to(self.device)

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

                if do_classifier_free_guidance:
                    empty_embeds, _, empty_pooled, _ = self.pipeline.encode_prompt(
                        prompt="", device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                    empty_embeds = empty_embeds.to(self.device)
                    empty_pooled = empty_pooled.to(self.device)

                    embeds_input = torch.cat([empty_embeds, prompt_embeds])
                    pooled_input = torch.cat([empty_pooled, pooled_prompt_embeds])
                else:
                    embeds_input = prompt_embeds
                    pooled_input = pooled_prompt_embeds

                vae_scale_factor = getattr(self.pipeline, "vae_scale_factor", 8)
                h, w = latent_noise.shape[-2] * vae_scale_factor, latent_noise.shape[-1] * vae_scale_factor

                time_ids = self.pipeline._get_add_time_ids(
                    (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
                ).to(self.device)

                if do_classifier_free_guidance:
                    time_ids_input = torch.cat([time_ids, time_ids])
                else:
                    time_ids_input = time_ids

                latents = latent_noise.clone()
                max_blend_step = int(num_steps * blend_threshold)

                for i, t in enumerate(timesteps):
                    # Явный перенос латента из траектории на GPU
                    source_latents = context[i].to(self.device)

                    if do_classifier_free_guidance:
                        latent_input = torch.cat([latents] * 2)
                    else:
                        latent_input = latents

                    latent_input = self.pipeline.scheduler.scale_model_input(latent_input, t)

                    noise_pred = self.pipeline.unet(
                        latent_input, t, encoder_hidden_states=embeds_input,
                        added_cond_kwargs={"text_embeds": pooled_input, "time_ids": time_ids_input}
                    ).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample

                    if i < max_blend_step:
                        latents = alpha * source_latents + (1 - alpha) * latents

                image = self.pipeline.vae.decode(
                    latents / self.pipeline.vae.config.scaling_factor, return_dict=False
                )[0]
                image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

            return image

        finally:
            self.pipeline.scheduler = original_scheduler