import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
from PIL import Image
from diffusers import StableDiffusionXLPipeline

class BaseInverter(ABC):
    """
    Базовый абстрактный класс для всех методов инверсии.
    """

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.dtype = pipeline.dtype

    @abstractmethod
    def invert(
        self,
        image: Image.Image,
        prompt: str,
        num_steps: int = 50,
        mask: Optional[Image.Image] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Инвертирует изображение в латентный шум.
        Args:
            image: PIL Image (RGB).
            prompt: текстовый промпт.
            num_steps: количество шагов инверсии.
            mask: опциональная маска (PIL Image, ч/б), может использоваться в некоторых методах (например, Null-text).
            **kwargs: дополнительные аргументы.
        Returns:
            (latent_noise, context): латентный шум и дополнительный контекст (если нужен).
        """
        pass

    def reconstruct(
        self,
        latent_noise: torch.Tensor,
        prompt: str,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        **kwargs
    ) -> Image.Image:
        """Восстанавливает изображение из латентного шума."""
        # Достаем контекст (например, эмбеддинги для Null-text), если он передан
        context = kwargs.get("context", None)
        with torch.no_grad():
            restored_image = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                latents=latent_noise,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=context
            ).images[0]
        return restored_image

    def run(
            self,
            image: Image.Image,
            orig_prompt: str,
            edit_prompt: str,
            mask: Optional[Image.Image] = None,
            **kwargs
    ) -> Image.Image:
        """
        Единая точка входа для конвейера тестирования (Оркестратора).
        Выполняет полный цикл: инверсия оригинального изображения -> генерация с новым промптом.
        """
        # Вызываем метод invert, передавая маску, если она есть
        invert_results = self.invert(
            image=image,
            prompt=orig_prompt,
            num_steps=kwargs.get("num_steps", 50),
            mask=mask,
            **kwargs
        )
        # Распаковываем результаты (возможен кортеж: шум и контекст)
        if isinstance(invert_results, tuple):
            noise, context = invert_results
        else:
            noise = invert_results
            context = None

        # Реконструкция с новым промптом
        return self.reconstruct(
            latent_noise=noise,
            prompt=edit_prompt,
            context=context,
            num_steps=kwargs.get("num_steps", 50),
            guidance_scale=kwargs.get("guidance_scale", 7.5)
        )

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Преобразует PIL Image в тензор, готовый для pipeline SDXL."""
        tensor = self.pipeline.image_processor.preprocess(image)
        return tensor.to(device=self.device, dtype=self.dtype)

    def postprocess_image(self, image: Image.Image) -> Image.Image:
        """Постобработка, если потребуется."""
        return image
