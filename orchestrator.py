import os
import re
import traceback
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Any, List
from metrics.performance import PerformanceMonitor


class EvaluationPipeline:
    """
    Оркестратор бенчмарка: загружает датасет PIE_Bench_pp, прогоняет изображения через
    указанные методы инверсии (DDIM, Direct, Null-text), собирает метрики и сохраняет результаты.
    """

    def __init__(self, methods_dict: Dict[str, Any], evaluator: Any, device: str = "cuda"):
        self.hf_repo = "UB-CVML-Group/PIE_Bench_pp"
        self.methods = methods_dict
        self.evaluator = evaluator
        self.device = device

        self.results: List[Dict[str, Any]] = []
        self.dataset: List[Dict[str, Any]] = []

    def load_data(self, subsets: List[str], split: str = "V1"):
        """Загружает указанные подмножества датасета PIE_Bench_pp из Hugging Face."""
        self.dataset = []
        print(f"Загрузка данных из {self.hf_repo} (split='{split}')...")

        for subset_name in subsets:
            try:
                print(f"  Категория: {subset_name}")
                ds = load_dataset(self.hf_repo, subset_name, split=split)

                for idx, item in enumerate(ds):
                    source_prompt = item.get('source_prompt', '')
                    target_prompt = item.get('target_prompt', '')
                    img_id = str(item.get('id', idx))

                    mask_img = item.get('mask')
                    if mask_img is not None:
                        mask_img = mask_img.convert('L')

                    self.dataset.append({
                        "category": subset_name,
                        "image": item['image'].convert('RGB'),
                        "mask": mask_img,
                        "source_prompt": source_prompt,
                        "target_prompt": target_prompt,
                        "image_id": img_id
                    })
            except Exception as e:
                print(f"  Ошибка при загрузке {subset_name}: {e}")

        print(f"Загружено изображений: {len(self.dataset)}")

    def _sanitize_filename(self, name: str) -> str:
        """Заменяет недопустимые символы на '_' для безопасного имени файла."""
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name))

    def run_evaluation(self, results_dir: str = "results", output_csv: str = "evaluation_results.csv") -> pd.DataFrame:
        """Запускает процесс оценки: генерация, расчёт метрик, сохранение."""
        print("\n=== Запуск пайплайна оценки ===")

        os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "errors"), exist_ok=True)
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
            mask = item.get('mask')

            for method_name, method_pipeline in self.methods.items():
                run_key = f"{category}_{img_id}_{method_name}"
                if run_key in processed_keys:
                    continue

                safe_cat = self._sanitize_filename(category)
                safe_id = self._sanitize_filename(img_id)
                safe_meth = self._sanitize_filename(method_name)

                row_data = {}

                try:
                    with PerformanceMonitor() as monitor:
                        edited_image = method_pipeline.run(
                            image=image,
                            source_prompt=source_prompt,
                            target_prompt=target_prompt,
                            mask=mask
                        )

                    if edited_image is None:
                        raise ValueError(f"Метод {method_name} вернул None вместо изображения.")

                    metrics_dict = self.evaluator.calculate_metrics(
                        original=image,
                        reconstructed=edited_image,
                        prompt_orig=source_prompt,
                        prompt_edit=target_prompt
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
                        torch.cuda.empty_cache()

        print(f"\nПайплайн завершён. Результаты сохранены в {csv_path}")
        return pd.DataFrame(self.results)