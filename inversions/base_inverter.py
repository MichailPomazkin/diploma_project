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
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Инвертирует изображение в латентный шум.
        Args:
            image: PIL Image (RGB).
            prompt: текстовый промпт.
            num_steps: количество шагов инверсии.
            **kwargs: дополнительные аргументы для конкретного метода.
        Returns:
            (latent_noise, context): латентный шум и дополнительный контекст (если нужен).
        """
        pass

    @abstractmethod
    def reconstruct(
        self,
        latent_noise: torch.Tensor,
        prompt: str,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        **kwargs
    ) -> Image.Image:
        """
        Восстанавливает изображение из латентного шума.
        Args:
            latent_noise: тензор шума (обычно из метода invert).
            prompt: текстовый промпт.
            num_steps: количество шагов восстановления.
            guidance_scale:
            **kwargs:масштаб guidance (1.0 = без guidance). дополнительные аргументы.
        Returns:
            PIL Image.
        """
        pass

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Преобразует PIL Image в тензор, готовый для pipeline SDXL.
        """
        # Используем встроенный процессор пайплайна и сразу кидаем на нужный девайс
        tensor = self.pipeline.image_processor.preprocess(image)
        return tensor.to(device=self.device, dtype=self.dtype)

    def postprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Постобработка, если потребуется.
        """
        return image
