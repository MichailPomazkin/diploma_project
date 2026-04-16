import torch
from typing import Optional, Tuple, List
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline

from .base_inverter import BaseInverter


class DirectInverter(BaseInverter):
    """
    Реализация метода Direct Inversion для SDXL.

    Идея:
        - На этапе invert сохраняется полная траектория латентных векторов от z_0 до z_T.
        - На этапе reconstruct эта траектория используется для «привязки» генерации
          к исходной структуре изображения при изменённом текстовом промпте.
          Это позволяет редактировать изображение (замена объекта, фона, стиля),
          сохраняя композицию и геометрию оригинала.
    """

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        super().__init__(pipeline)

        # Создаём независимые копии шедулеров на основе конфигурации текущего пайплайна.
        # Это позволяет не загружать их из интернета и не влиять на глобальное состояние.
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config)
        self.forward_scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

    def invert(
            self,
            image: Image.Image,
            prompt: str,
            num_steps: int = 50,
            **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Преобразует изображение в шум и сохраняет траекторию всех промежуточных латентов.

        Возвращает:
            - latent_noise: финальный латентный шум z_T
            - trajectory: список латентов от z_T до z_0 (в порядке от шума к чистому изображению)
        """
        print("[Direct Inversion] Извлечение опорной траектории (детерминировано)...")
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            # Детерминированное кодирование: берём центр распределения (mode) вместо случайной выборки.
            # Это гарантирует, что повторные запуски дадут одинаковый результат.
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.mode()
            latents = latents * self.pipeline.vae.config.scaling_factor

            # Кодируем текстовый промпт. do_classifier_free_guidance=False, потому что
            # на этапе инверсии CFG не применяется (мы просто идём по обратному пути).
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

            # SDXL требует дополнительных временных идентификаторов (time_ids), которые
            # кодируют размеры выходного изображения. Они передаются в UNet через added_cond_kwargs.
            h, w = image_tensor.shape[-2:]
            time_ids = self.pipeline._get_add_time_ids(
                (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
            ).to(self.device)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

        # Настраиваем обратный шедулер на нужное количество шагов
        self.inverse_scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.inverse_scheduler.timesteps

        # Начинаем с чистого латента z_0
        trajectory = [latents.clone()]
        current_latents = latents.clone()

        with torch.no_grad():
            for t in timesteps:
                # Предсказываем шум, который нужно добавить, чтобы перейти от текущего латента
                # к более зашумлённому (обратный процесс DDIM)
                noise_pred = self.pipeline.unet(
                    current_latents, t, encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                # Делаем шаг инверсии: получаем латент для предыдущего (более шумного) шага
                current_latents = self.inverse_scheduler.step(noise_pred, t, current_latents).prev_sample
                trajectory.append(current_latents.clone())

        # Траектория сейчас: [z_0, z_1, ..., z_T] (от чистого к шуму).
        # Переворачиваем, чтобы первый элемент был z_T (шум), последний — z_0.
        trajectory = list(reversed(trajectory))
        latent_noise = trajectory[0]   # z_T

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
            **kwargs
    ) -> Image.Image:
        """
        Восстанавливает изображение из шума, используя сохранённую траекторию (context).

        Параметры:
            alpha: сила смешивания с исходной траекторией (0 – только генерация, 1 – только исходный латент)
            blend_threshold: доля шагов (от начала), на которых применяется смешивание.
                             Например, 0.7 означает, что на первых 70% шагов мы подмешиваем исходную траекторию.
        """
        # Если контекст не передан — выполняем обычную реконструкцию (как в DDIM)
        if context is None:
            print("[Direct Inversion] Внимание: контекст не передан, выполняется обычный DDIM.")
            return super().reconstruct(latent_noise, prompt, num_steps, guidance_scale, **kwargs)

        # Проверяем, что длина контекста совпадает с количеством шагов + 1
        if len(context) != num_steps + 1:
            raise ValueError(f"Длина контекста ({len(context)}) не совпадает с num_steps + 1 ({num_steps + 1})")

        print(f"[Direct Inversion] Направленная реконструкция (alpha={alpha}, threshold={blend_threshold})...")

        # Временно подменяем шедулер пайплайна на прямой DDIM-шедулер.
        # Это нужно, потому что базовый метод reconstruct использует self.pipeline.scheduler.
        # Блок try/finally гарантирует возврат исходного шедулера даже при ошибке.
        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.forward_scheduler

        try:
            self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
            timesteps = self.pipeline.scheduler.timesteps

            # CFG (Classifier-Free Guidance) ускоряет и экономит память, если guidance_scale == 1.0
            do_classifier_free_guidance = guidance_scale > 1.0

            with torch.no_grad():
                # Кодируем текстовый промпт (без CFG, только один раз)
                prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                    prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                )

                # Если включён CFG, добавляем пустые эмбеддинги для безусловной ветки
                if do_classifier_free_guidance:
                    empty_embeds, _, empty_pooled, _ = self.pipeline.encode_prompt(
                        prompt="", device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                    embeds_input = torch.cat([empty_embeds, prompt_embeds])      # [uncond, cond]
                    pooled_input = torch.cat([empty_pooled, pooled_prompt_embeds])
                else:
                    embeds_input = prompt_embeds
                    pooled_input = pooled_prompt_embeds

                # Вычисляем размеры для time_ids. Масштабный коэффициент VAE получаем из пайплайна.
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

                # Основной цикл генерации
                for i, t in enumerate(timesteps):
                    # Берём соответствующий латент из сохранённой траектории (z_{T-i})
                    source_latents = context[i]

                    # Подготовка входа для UNet: если CFG — удваиваем батч
                    if do_classifier_free_guidance:
                        latent_input = torch.cat([latents] * 2)
                    else:
                        latent_input = latents

                    # Масштабирование входа в соответствии с сигмой шедулера (важно для DDIM)
                    latent_input = self.pipeline.scheduler.scale_model_input(latent_input, t)

                    # Предсказание шума
                    noise_pred = self.pipeline.unet(
                        latent_input, t, encoder_hidden_states=embeds_input,
                        added_cond_kwargs={"text_embeds": pooled_input, "time_ids": time_ids_input}
                    ).sample

                    # Применяем CFG, если нужно
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Шаг DDIM вперёд (добавление шума → очистка)
                    latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample

                    # Смешивание с исходной траекторией на первых max_blend_step шагах
                    # Это ключевая идея Direct Inversion: мы «привязываем» генерацию к оригинальной структуре.
                    if i < max_blend_step:
                        latents = alpha * source_latents + (1 - alpha) * latents

                # Декодируем латенты обратно в пиксельное пространство через VAE
                image = self.pipeline.vae.decode(
                    latents / self.pipeline.vae.config.scaling_factor, return_dict=False
                )[0]
                # Преобразуем тензор в PIL-изображение
                image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

            return image

        finally:
            # Восстанавливаем исходный шедулер, чтобы не повлиять на другие инвертеры
            self.pipeline.scheduler = original_scheduler