import os
import re
import gc
import traceback
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Any, List
from PIL import Image
from datasets import load_dataset

# Импортируем CLIPSeg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from metrics.performance import PerformanceMonitor


class EvaluationPipeline:
    """
    Оркестратор бенчмарка: загружает датасет PIE_Bench_pp, генерирует внешние маски (CLIPSeg),
    прогоняет изображения через методы инверсии, собирает метрики и сохраняет результаты.
    """

    def __init__(self, methods_dict: Dict[str, Any], evaluator: Any, device: str = "cuda"):
        self.hf_repo = "UB-CVML-Group/PIE_Bench_pp"
        self.methods = methods_dict
        self.evaluator = evaluator
        self.device = device

        self.results: List[Dict[str, Any]] = []
        self.dataset: List[Dict[str, Any]] = []

        # --- ИНИЦИАЛИЗАЦИЯ CLIPSeg ---
        print("[Оркестратор] Загрузка модели CLIPSeg для точной сегментации...")
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        print("[Оркестратор] CLIPSeg готов!")

    def _get_clipseg_mask(self, image: Image.Image, text: str, threshold: float = 0.4) -> Image.Image:
        """Внутренний метод генерации маски по слову с помощью CLIPSeg."""
        inputs = self.clipseg_processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.clipseg_model(**inputs)

        # Преобразуем логиты в вероятности (Sigmoid)
        probs = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()

        # Создаем бинарную маску
        mask_np = (probs > threshold).astype(np.uint8) * 255
        return Image.fromarray(mask_np, mode='L')

    def load_data(self, subsets: List[str], split: str = "V1"):
        """Загружает указанные подмножества и парсит слова для редактирования."""
        self.dataset = []
        print(f"Загрузка данных из {self.hf_repo} (split='{split}')...")

        for subset_name in subsets:
            try:
                ds = load_dataset(self.hf_repo, subset_name, split=split)
                for idx, item in enumerate(ds):
                    source_prompt = item.get('source_prompt', '')
                    target_prompt = item.get('target_prompt', '')
                    img_id = str(item.get('id', idx))

                    word_to_replace = None
                    edit_action = item.get('edit_action', {})

                    if edit_action:
                        try:
                            if isinstance(edit_action, str):
                                import ast
                                edit_action = ast.literal_eval(edit_action)

                            # --- ИСПРАВЛЕННЫЙ ПАРСЕР (Поддерживает и Change, и Delete) ---
                            if isinstance(edit_action, dict):
                                target_key = list(edit_action.keys())[0]  # 'change' или 'delete'
                                action_data = edit_action[target_key]

                                if isinstance(action_data, dict):
                                    if 'action' in action_data:
                                        raw_word = str(action_data['action'])
                                        if raw_word != '+':
                                            word_to_replace = raw_word
                                elif isinstance(action_data, str):
                                    # Это спасет категорию delete (например, {'delete': 'cat'})
                                    word_to_replace = action_data

                        except Exception as e:
                            print(f"  [Оркестратор] Не удалось распарсить edit_action: {e}")

                    self.dataset.append({
                        "category": subset_name,
                        "image": item['image'].convert('RGB'),
                        "source_prompt": source_prompt,
                        "target_prompt": target_prompt,
                        "image_id": img_id,
                        "word_to_replace": word_to_replace
                    })
            except Exception as e:
                print(f"  Ошибка при загрузке {subset_name}: {e}")

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name))

    def run_evaluation(self, results_dir: str = "results", output_csv: str = "evaluation_results.csv") -> pd.DataFrame:
        """Запускает процесс оценки: генерация маски, инверсия, расчёт метрик, сохранение."""
        print("\n=== Запуск пайплайна оценки ===")

        os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "errors"), exist_ok=True)
        # Опционально: сохраняем маски CLIPSeg, чтобы вы могли вставить их в диплом!
        os.makedirs(os.path.join(results_dir, "clipseg_masks"), exist_ok=True)

        csv_path = os.path.join(results_dir, output_csv)

        processed_keys = set()
        if os.path.exists(csv_path):
            try:
                df_existing = pd.read_csv(csv_path)
                for _, row in df_existing.iterrows():
                    cat = row.get('category', 'unknown')
                    img = row.get('image_id', 'unknown')
                    meth = row.get('method', 'unknown')
                    processed_keys.add(f"{cat}_{img}_{meth}")

                self.results = df_existing.to_dict('records')
                print(f"Найдено ранее обработанных записей: {len(processed_keys)}. Продолжаем.")
            except Exception as e:
                print(f"Ошибка чтения {csv_path}: {e}. Начинаем заново.")

        for item in tqdm(self.dataset, desc="Обработка датасета"):
            image = item['image']
            source_prompt = item['source_prompt']
            target_prompt = item['target_prompt']
            category = item['category']
            img_id = item['image_id']
            word_to_replace = item.get('word_to_replace')

            # --- ГЕНЕРИРУЕМ МАСКУ CLIPSeg ---
            clipseg_mask = None
            if word_to_replace and word_to_replace.lower() != "none":
                try:
                    clipseg_mask = self._get_clipseg_mask(image, word_to_replace)
                    # Сохраним маску для наглядности (полезно для диплома)
                    mask_path = os.path.join(results_dir, "clipseg_masks", f"mask_{img_id}_{word_to_replace}.png")
                    clipseg_mask.save(mask_path)
                except Exception as e:
                    print(f"  [Внимание] Ошибка CLIPSeg для {word_to_replace}: {e}")

            for method_name, method_pipeline in self.methods.items():
                run_key = f"{category}_{img_id}_{method_name}"
                if run_key in processed_keys:
                    continue

                safe_cat = self._sanitize_filename(category)
                safe_id = self._sanitize_filename(img_id)
                safe_meth = self._sanitize_filename(method_name)

                row_data = {}

                if self.device == "cuda" and torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                try:
                    print(
                        f"\n  [DEBUG] Метод: {method_name} | Ищем: '{word_to_replace}' | Маска CLIPSeg: {'Да' if clipseg_mask else 'Нет'}")

                    with PerformanceMonitor() as monitor:
                        # --- ПЕРЕДАЕМ ИЗОБРАЖЕНИЕ И МАСКУ ---
                        edited_image = method_pipeline.run(
                            image=image,
                            source_prompt=source_prompt,
                            target_prompt=target_prompt,
                            mask=clipseg_mask,  # <--- ВОТ НАША ИДЕАЛЬНАЯ МАСКА!
                            image_id=img_id
                        )

                    if edited_image is None:
                        raise ValueError(f"Метод {method_name} вернул None вместо изображения.")

                    metrics_dict = self.evaluator.calculate_metrics(
                        original=image,
                        reconstructed=edited_image,
                        source_prompt=source_prompt,
                        target_prompt=target_prompt
                    )

                    img_filename = f"{safe_cat}_{safe_id}_{safe_meth}.png"
                    img_path = os.path.join(results_dir, "images", img_filename)
                    edited_image.save(img_path)

                    row_data = {
                        "image_id": img_id,
                        "category": category,
                        "method": method_name,
                        "time_sec": monitor.execution_time,
                        "vram_mb": monitor.peak_vram_mb,
                        "saved_path": img_path,
                        "error": None
                    }
                    row_data.update(metrics_dict)

                except Exception as e:
                    error_msg = traceback.format_exc()
                    print(f"\nОшибка метода {method_name} на изображении {img_id}: {str(e)}")

                    error_file = os.path.join(results_dir, "errors", f"error_{safe_cat}_{safe_id}_{safe_meth}.txt")
                    with open(error_file, "w") as f:
                        f.write(error_msg)

                    row_data = {
                        "image_id": img_id,
                        "category": category,
                        "method": method_name,
                        "time_sec": None,
                        "vram_mb": None,
                        "saved_path": None,
                        "error": str(e)
                    }

                finally:
                    if row_data:
                        self.results.append(row_data)
                        pd.DataFrame(self.results).to_csv(csv_path, index=False)
                        processed_keys.add(run_key)

                    if self.device == "cuda" and torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()

        print(f"\nПайплайн завершён. Результаты сохранены в {csv_path}")
        return pd.DataFrame(self.results)