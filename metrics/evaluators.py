import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import Optional
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from lpips import LPIPS
from transformers import CLIPProcessor, CLIPModel


class ImageInversionEvaluator:
    """
    Класс для оценки качества инверсии изображений.
    Содержит методы для вычисления математических и семантических метрик.
    """

    def __init__(self, device="cuda"):
        self.device = device

        print("Инициализация математических метрик (PSNR, SSIM, LPIPS)...")
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LPIPS(net='vgg').to(device)

        print("Загружаем веса CLIP...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Приводит PIL изображение к тензору [1, 3, H, W] в диапазоне [0, 1]."""
        img = np.array(pil_image).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def calculate_metrics(
            self,
            original: Image.Image,
            reconstructed: Image.Image,
            prompt_orig: Optional[str] = None,
            prompt_edit: Optional[str] = None
    ):
        """Считает метрики для пары изображений."""
        with torch.no_grad():
            img1 = self.preprocess(original)
            img2 = self.preprocess(reconstructed)

            psnr_val = self.psnr(img1, img2).item()
            ssim_val = self.ssim(img1, img2).item()

            img1_lpips = img1 * 2.0 - 1.0
            img2_lpips = img2 * 2.0 - 1.0
            lpips_val = self.lpips_metric(img1_lpips, img2_lpips).item()
            mse_val = F.mse_loss(img1, img2).item()

            results = {
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "mse": mse_val
            }

            if prompt_edit:
                results["clip_tgt_recon"] = self.calculate_clip_score(reconstructed, prompt_edit)
                results["clip_tgt_orig"] = self.calculate_clip_score(original, prompt_edit)

            if prompt_orig:
                results["clip_src_orig"] = self.calculate_clip_score(original, prompt_orig)

            if prompt_orig and prompt_edit:
                results["directional_clip"] = self.calculate_directional_clip(
                    original, reconstructed, prompt_orig, prompt_edit
                )

        return results

    def get_clip_embeddings(self, image: Image.Image = None, text: str = None):
        """
        Извлекает нормализованные эмбеддинги для текста или картинки.
        Возвращает тензор формы (1, embedding_dim).
        """
        with torch.no_grad():
            if text is not None:
                inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
                # Ручное извлечение через text_model (совместимо со старыми версиями transformers)
                text_outputs = self.clip_model.text_model(**inputs)
                pooled = text_outputs.pooler_output  # shape (1, 768) для ViT-B/32
                emb = self.clip_model.text_projection(pooled)  # проекция в 512
            elif image is not None:
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                vision_outputs = self.clip_model.vision_model(**inputs)
                pooled = vision_outputs.pooler_output  # shape (1, 768)
                emb = self.clip_model.visual_projection(pooled)  # проекция в 512
            else:
                raise ValueError("Нужно передать либо текст, либо картинку")

        # Нормализация (L2)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb

    def calculate_clip_score(self, image: Image.Image, prompt: str) -> float:
        """Косинусное сходство между эмбеддингами изображения и текста (от -1 до 1)."""
        img_emb = self.get_clip_embeddings(image=image)
        txt_emb = self.get_clip_embeddings(text=prompt)
        return F.cosine_similarity(img_emb, txt_emb).item()

    def calculate_directional_clip(self, img_orig: Image.Image, img_edit: Image.Image,
                                   prompt_orig: str, prompt_edit: str) -> float:
        """Directional CLIP: косинусное сходство между векторами изменений изображения и текста."""
        img_orig_emb = self.get_clip_embeddings(image=img_orig)
        img_edit_emb = self.get_clip_embeddings(image=img_edit)
        txt_orig_emb = self.get_clip_embeddings(text=prompt_orig)
        txt_edit_emb = self.get_clip_embeddings(text=prompt_edit)

        img_diff = img_edit_emb - img_orig_emb
        txt_diff = txt_edit_emb - txt_orig_emb
        return F.cosine_similarity(img_diff, txt_diff).item()

