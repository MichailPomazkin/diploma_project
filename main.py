import sys
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from inversions.ddim import DDIMInverter
from inversions.direct_inversion import DirectInverter
from inversions.null_text import NullTextInverter
from metrics.evaluators import ImageInversionEvaluator
from orchestrator import EvaluationPipeline


class InverterWrapper:
    """
    Обёртка, позволяющая передавать дополнительные аргументы (например, use_spatial_mask)
    в метод run инвертера, не меняя код оркестратора.
    """
    def __init__(self, inverter_instance, **custom_kwargs):
        self.inverter = inverter_instance
        self.custom_kwargs = custom_kwargs

    def run(self, image, prompt_orig, prompt_edit, mask=None, **kwargs):
        final_kwargs = {**kwargs, **self.custom_kwargs}
        if mask is not None:
            final_kwargs['mask'] = mask
        return self.inverter.run(image, prompt_orig, prompt_edit, **final_kwargs)


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
        default="V1",
        help="Сплит датасета для загрузки (PIE_Bench_pp использует 'V1')."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Инициализация пайплайна. Устройство: {device.upper()}, тип данных: {dtype}")

    print("Загрузка модели SDXL с исправленным VAE...")
    try:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=dtype,
            use_safetensors=True
        ).to(device)

        pipe_kwargs = {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "vae": vae,
            "torch_dtype": dtype,
            "use_safetensors": True
        }
        if device == "cuda":
            pipe_kwargs["variant"] = "fp16"

        pipe = StableDiffusionXLPipeline.from_pretrained(**pipe_kwargs).to(device)
        pipe.upcast_vae = False

        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None

        if device == "cuda":
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"Модель загружена. VRAM: {vram_used:.2f} GB")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        sys.exit(1)

    print("Инициализация методов инверсии...")
    base_null = NullTextInverter(pipe)

    methods_dict = {
        "DDIM": DDIMInverter(pipe),
        "DirectInversion": DirectInverter(pipe),
        "NullText_Original": InverterWrapper(base_null, use_spatial_mask=False),
        "NullText_Masked": InverterWrapper(base_null, use_spatial_mask=True),
    }

    print("Инициализация модуля оценки качества...")
    evaluator = ImageInversionEvaluator(device=device)

    print("Настройка конвейера...")
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

    print("\nПроцесс завершён.")
    if results_df is not None and not results_df.empty:
        total_runs = len(results_df)
        errors_df = results_df[results_df['error'].notna()]
        total_errors = len(errors_df)
        successful_runs = total_runs - total_errors
        print(f"Всего обработано: {total_runs}, Успешно: {successful_runs}, Ошибок: {total_errors}")
        if total_errors > 0:
            print("Подробности ошибок в results/errors/")
    else:
        print("Нет результатов для анализа.")


if __name__ == "__main__":
    main()