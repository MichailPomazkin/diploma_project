import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Tuple, List
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline

from .base_inverter import BaseInverter


class NullTextInverter(BaseInverter):
    """
    Реализация метода Null-text Inversion
    Основная идея:
        - На этапе invert получаем опорную DDIM-траекторию (z_0 → z_T) для исходного изображения.
        - Затем для каждого шага t (от T до 1) оптимизируем unconditional text embeddings (пустой промпт)
          так, чтобы при CFG с фиксированным guidance_scale шаг DDIM приводил к целевому латенту z_{t-1}.
        - В результате получаем набор оптимизированных unconditional эмбеддингов (по одному на шаг).
        - На этапе reconstruct эти эмбеддинги используются вместо стандартных пустых, что позволяет
          точно восстановить исходное изображение даже при guidance_scale > 1.
    """

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        super().__init__(pipeline)

        # Создаём независимые копии шедулеров из конфигурации (без загрузки из интернета)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config)
        self.forward_scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        # Кешируем эмбеддинги пустого промпта – они понадобятся как начальное приближение
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
            num_inner_steps: int = 5,       # уменьшено для стабильности и скорости
            learning_rate: float = 1e-3,    # пониженная скорость обучения для SDXL
            **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Выполняет null-text инверсию.

        Возвращает:
            - latent_noise: шум z_T
            - optimized_uncond_embeddings: список оптимизированных unconditional эмбеддингов (длина num_steps)
        """
        if guidance_scale <= 1.0:
            print("[Null-text] Внимание: guidance_scale <= 1.0. Метод оптимизирует CFG, поэтому при таких значениях теряет математический смысл.")

        # ---- Этап 1: Получение опорной DDIM-траектории ----
        print("[Null-text] Этап 1: Извлечение опорной DDIM-траектории (детерминировано)...")
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            # Детерминированное кодирование: берём моду распределения латентов
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.mode()
            latents = latents * self.pipeline.vae.config.scaling_factor

            # Кодирование текстового промпта (без CFG, т.к. инверсия идёт без guidance)
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

            # Для SDXL необходимы time_ids, которые передаются в UNet как дополнительное условие
            h, w = image_tensor.shape[-2:]
            time_ids = self.pipeline._get_add_time_ids(
                (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
            ).to(self.device)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

        # Настройка обратного шедулера
        self.inverse_scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.inverse_scheduler.timesteps

        # Сохраняем траекторию: начиная с z_0, затем z_1, ..., z_T
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

        # Переворачиваем траекторию: теперь порядок от z_T (шум) до z_0
        trajectory = list(reversed(trajectory))
        latent_noise = trajectory[0]   # z_T

        # ---- Этап 2: Оптимизация unconditional эмбеддингов ----
        print(f"[Null-text] Этап 2: Градиентная оптимизация (Adam, {num_inner_steps} итераций/шаг)...")
        self.forward_scheduler.set_timesteps(num_steps, device=self.device)
        forward_timesteps = self.forward_scheduler.timesteps

        optimized_uncond_embeddings = []
        current_latent = latent_noise.clone().detach()   # начинаем с z_T

        for i, t in enumerate(forward_timesteps):
            target_latent = trajectory[i + 1].detach()   # целевой латент z_{T-i-1}

            # Клонируем начальный пустой эмбеддинг, переводим в float32 для устойчивости градиентов
            uncond_embeds_opt = self.empty_embeds.clone().detach().to(torch.float32).requires_grad_(True)
            optimizer = Adam([uncond_embeds_opt], lr=learning_rate)

            pred_latent = None   # будет хранить предсказание на последней итерации

            # Внутренний цикл градиентного спуска для подбора эмбеддинга на данном шаге t
            for inner_step in range(num_inner_steps):
                optimizer.zero_grad()

                # Возвращаем эмбеддинг в fp16 для прогона через UNet
                uncond_fp16 = uncond_embeds_opt.to(dtype=prompt_embeds.dtype)

                # Для CFG нужно два экземпляра латента (conditional и unconditional)
                latent_input = torch.cat([current_latent] * 2)
                latent_input = self.forward_scheduler.scale_model_input(latent_input, t)

                # Конкатенируем оптимизируемый unconditional эмбеддинг с текстовым эмбеддингом
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
                loss = F.mse_loss(pred_latent.float(), target_latent.float())

                # Защита: при обнаружении NaN в loss или в предсказании – прерываем итерации
                if torch.isnan(loss) or torch.isnan(pred_latent).any() or torch.isinf(pred_latent).any():
                    print(f"  [Null-text] NaN/Inf на шаге {i}, итерация {inner_step}. Прерываем оптимизацию.")
                    pred_latent = target_latent
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_([uncond_embeds_opt], 1.0)  # защита от взрыва градиентов
                optimizer.step()

                # Клиппинг самих эмбеддингов, чтобы они не уходили в бесконечность
                with torch.no_grad():
                    uncond_embeds_opt.clamp_(-10.0, 10.0)

            # Сохраняем оптимизированный эмбеддинг для данного шага (возвращаем в fp16)
            optimized_uncond_embeddings.append(uncond_embeds_opt.detach().to(dtype=prompt_embeds.dtype))

            # Используем предсказание с последней итерации как текущий латент для следующего шага.
            # Если ошибка слишком велика, используем целевой латент из траектории (fallback).
            if pred_latent is not None:
                mse_error = F.mse_loss(pred_latent.float(), target_latent.float()).item()
                if mse_error > 1.0:
                    print(f"  [Null-text] Ошибка велика (MSE={mse_error:.4f}). Сброс на целевой латент.")
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
        """
        Восстанавливает изображение из шума, используя оптимизированные unconditional эмбеддинги (context).
        Если context не передан, выполняется обычный DDIM (без оптимизации).
        """
        if context is None:
            print("[Null-text] Внимание: контекст не передан, выполняется обычный цикл генерации.")
        elif len(context) != num_steps:
            raise ValueError(f"Длина контекста ({len(context)}) должна равняться num_steps ({num_steps})")

        print("[Null-text] Восстанавливаем с использованием оптимизированного контекста...")

        # Временно подменяем шедулер пайплайна на forward_scheduler.
        # Блок try/finally гарантирует восстановление исходного состояния.
        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = self.forward_scheduler

        try:
            self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
            timesteps = self.pipeline.scheduler.timesteps

            # CFG имеет смысл только если guidance_scale > 1.0
            do_classifier_free_guidance = guidance_scale > 1.0

            with torch.no_grad():
                prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                    prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                )

                # Вычисляем размеры для time_ids (масштаб VAE обычно 8)
                vae_scale_factor = getattr(self.pipeline, "vae_scale_factor", 8)
                h, w = latent_noise.shape[-2] * vae_scale_factor, latent_noise.shape[-1] * vae_scale_factor
                time_ids = self.pipeline._get_add_time_ids(
                    (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim
                ).to(self.device)

                latents = latent_noise.clone()

                for i, t in enumerate(timesteps):
                    # Берём оптимизированный unconditional эмбеддинг для данного шага (если есть)
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

                # Декодирование латентов в изображение
                image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
                image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

            return image

        finally:
            self.pipeline.scheduler = original_scheduler