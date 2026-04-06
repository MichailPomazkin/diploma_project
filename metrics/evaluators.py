import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
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
        # Инициализируем метрики
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # LPIPS требует загрузки весов (vgg по умолчанию)
        self.lpips_metric = LPIPS(net='vgg').to(device)

        print("Загружаем веса CLIP...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Приводит PIL изображение к тензору [1, 3, H, W] в диапазоне [0, 1]."""
        img = np.array(pil_image).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def calculate_metrics(self, original: Image.Image, reconstructed: Image.Image, prompt: str = None):
        """
        Считает метрики для пары изображений.
        """
        img1 = self.preprocess(original)
        img2 = self.preprocess(reconstructed)

        # 1. PSNR (выше - лучше)
        psnr_val = self.psnr(img1, img2).item()

        # 2. SSIM (ближе к 1 - лучше)
        ssim_val = self.ssim(img1, img2).item()

        # 3. LPIPS (ближе к 0 - лучше)
        img1_lpips = img1 * 2.0 - 1.0
        img2_lpips = img2 * 2.0 - 1.0
        lpips_val = self.lpips_metric(img1_lpips, img2_lpips).item()

        # 4. MSE (ниже - лучше)
        mse_val = F.mse_loss(img1, img2).item()

        results = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "mse": mse_val
        }

        if prompt:
            results["clip_score_orig"] = self.calculate_clip_score(original, prompt)
            results["clip_score_recon"] = self.calculate_clip_score(reconstructed, prompt)

        return results

    def calculate_clip_score(self, image: Image.Image, prompt: str) -> float:
        "Считает классический CLIP-score (насколько картинка подходит под текст)."
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image[0][0].item()

    def get_clip_embeddings(self, image: Image.Image = None, text: str = None):
        """Извлекает нормализованные эмбеддинги для текста или картинки."""
        inputs = {}
        if text:
            text_inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
            inputs.update({k: v.to(self.device) for k, v in text_inputs.items()})
        if image:
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs.update({k: v.to(self.device) for k, v in image_inputs.items()})

        with torch.no_grad():
            # ФИКС: Обращаемся к энкодерам напрямую! Это невозможно сломать.
            if text and not image:
                outputs = self.clip_model.text_model(**inputs)
                emb = self.clip_model.text_projection(outputs.pooler_output)
            elif image and not text:
                outputs = self.clip_model.vision_model(**inputs)
                emb = self.clip_model.visual_projection(outputs.pooler_output)
            else:
                raise ValueError("Нужно передать либо текст, либо картинку")

        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    def calculate_directional_clip(self, img_orig: Image.Image, img_edit: Image.Image,
                                   prompt_orig: str, prompt_edit: str) -> float:
        """Считает Directional CLIP (насколько вектор изменения картинки совпадает с вектором текста)."""
        img_orig_emb = self.get_clip_embeddings(image=img_orig)
        img_edit_emb = self.get_clip_embeddings(image=img_edit)

        txt_orig_emb = self.get_clip_embeddings(text=prompt_orig)
        txt_edit_emb = self.get_clip_embeddings(text=prompt_edit)

        # Векторы направлений
        img_diff = img_edit_emb - img_orig_emb
        txt_diff = txt_edit_emb - txt_orig_emb

        # Косинусное сходство
        directional_similarity = F.cosine_similarity(img_diff, txt_diff).item()
        return directional_similarity

