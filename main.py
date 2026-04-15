import sys
import argparse
import torch
from diffusers import StableDiffusionXLPipeline

from inversions.ddim import DDIMInverter
from inversions.direct_inversion import DirectInverter
from inversions.null_text import NullTextInverter
from metrics.evaluators import ImageInversionEvaluator
from orchestrator import EvaluationPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Пайплайн оценки методов инверсии для SDXL.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=[
            "1_change_object_80",
            "8_change_background_80",
            "9_change_style_80"
        ],
        help="Список категорий датасета PIE-Bench++ для тестирования."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Сплит датасета для загрузки."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Инициализация пайплайна. Устройство: {device.upper()}, тип данных: {dtype}")

    print("Загрузка базовой модели Stable Diffusion XL 1.0...")
    try:
        # Умная загрузка: используем fp16 только если мы на видеокарте (GPU)
        pipe_kwargs = {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "torch_dtype": dtype,
            "use_safetensors": True
        }
        if device == "cuda":
            pipe_kwargs["variant"] = "fp16"

        pipe = StableDiffusionXLPipeline.from_pretrained(**pipe_kwargs).to(device)

        # Полное отключение цензуры и связанных с ней компонентов для чистоты эксперимента
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None

        # ОПЦИОНАЛЬНО: Раскомментируй строку ниже, если при запуске Null-Text будет ошибка Out Of Memory.
        # Это сэкономит VRAM, но замедлит генерацию на 20-30%.
        # pipe.enable_model_cpu_offload()

        if device == "cuda":
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"Модель успешно загружена. Использовано VRAM: {vram_used:.2f} GB")

    except Exception as e:
        print(f"Критическая ошибка при загрузке модели SDXL: {e}")
        sys.exit(1)

    print("Инициализация методов инверсии...")
    methods_dict = {
        "DDIM": DDIMInverter(pipe),
        "DirectInversion": DirectInverter(pipe),
        # NullTextInverter закомментирован для первого прогона.
        # "NullText": NullTextInverter(pipe)
    }

    print("Инициализация модуля оценки качества...")
    evaluator = ImageInversionEvaluator(device=device)

    print("Настройка параметров конвейера данных...")
    pipeline = EvaluationPipeline(
        methods_dict=methods_dict,
        evaluator=evaluator,
        device=device
    )

    pipeline.load_data(subsets=args.subsets, split=args.split)

    results_df = pipeline.run_evaluation(
        results_dir="results",
        output_csv="evaluation_results.csv"
    )

    print("\nПроцесс генерации и оценки завершен.")

    # Безопасное формирование итоговой статистики
    if results_df is not None and not results_df.empty:
        total_runs = len(results_df)
        errors_df = results_df[results_df['error'].notna()]
        total_errors = len(errors_df)
        successful_runs = total_runs - total_errors

        print(f"Итоговая статистика:")
        print(f"Всего обработано: {total_runs}")
        print(f"Успешно: {successful_runs}")
        print(f"Ошибок: {total_errors}")

        if total_errors > 0:
            print("Внимание: зафиксированы ошибки выполнения. Подробности сохранены в директории results/errors/.")
    else:
        print("Внимание: Нет результатов для анализа (возможно, датасет пуст или произошла ошибка).")


if __name__ == "__main__":
    main()