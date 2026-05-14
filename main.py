import sys
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# Импорты ваших локальных модулей
from inversions.ddim import DDIMInverter
from inversions.direct_inversion import DirectInverter
from inversions.null_text import NullTextInverter
from metrics.evaluators import ImageInversionEvaluator
from orchestrator import EvaluationPipeline


class InverterWrapper:
    """
    Обёртка, позволяющая передавать дополнительные аргументы (например, use_spatial_mask
    и reconstruct_kwargs) в метод run инвертера, не меняя код оркестратора.
    """

    def __init__(self, inverter_instance, **custom_kwargs):
        self.inverter = inverter_instance
        self.custom_kwargs = custom_kwargs

    def run(self, image, source_prompt, target_prompt, mask=None, token_index=None, **kwargs):
        # Базовое слияние словарей
        final_kwargs = {**self.custom_kwargs, **kwargs}

        if "reconstruct_kwargs" in self.custom_kwargs and "reconstruct_kwargs" in kwargs:
            final_kwargs["reconstruct_kwargs"] = {
                **self.custom_kwargs["reconstruct_kwargs"],
                **kwargs["reconstruct_kwargs"]
            }

        if mask is not None:
            final_kwargs['mask'] = mask
        if token_index is not None:
            final_kwargs['token_index'] = token_index

        return self.inverter.run(image, source_prompt, target_prompt, **final_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Пайплайн оценки методов инверсии для SDXL.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=[
            "1_change_object_80"  # Строго одна категория для честного теста геометрии
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

    print(f"=== Инициализация пайплайна ===")
    print(f"Устройство: {device.upper()}, тип данных: {dtype}")
    print(f"Категории: {args.subsets}")

    print("\n[1/4] Загрузка модели SDXL с исправленным VAE...")
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

        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("  Оптимизация памяти: xformers включены.")
            except Exception as e:
                print(f"  Внимание: не удалось включить xformers: {e}")

        # Отключаем лишнее для экономии памяти
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None

        # ГЛОБАЛЬНАЯ ЗАМОРОЗКА ГРАДИЕНТОВ (для экономии VRAM при инверсии)
        pipe.unet.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)
        print("  Градиенты глобально заморожены.")

        if device == "cuda":
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"  Модель загружена. Занято VRAM: {vram_used:.2f} GB")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        sys.exit(1)

    print("\n[2/4] Инициализация методов инверсии (Матрица из 5 моделей)...")
    base_null = NullTextInverter(pipe)

    methods_dict = {
        # --- Быстрые глобальные методы ---
        "1_DDIM": DDIMInverter(pipe),
        "2_DirectInversion": DirectInverter(pipe),

        # --- Baseline: Оригинальный NullText (Медленный, глобальный) ---
        "3_NullText_Original": InverterWrapper(
            base_null,
            use_spatial_mask=False,
            num_inner_steps=5
        ),

        # --- Optimization-based Masked Inversion (Тяжелый метод) ---
        "4_NullText_Masked": InverterWrapper(
            base_null,
            use_spatial_mask=True,
            num_inner_steps=5,
            reconstruct_kwargs={"cutoff_ratio": 0.55, "noise_strength": 0.5}
        ),

        # --- Optimization-free Masked DDIM Inversion (Легкий метод) ---
        "5_DDIM_Masked": InverterWrapper(
            base_null,
            use_spatial_mask=True,
            num_inner_steps=0,
            reconstruct_kwargs={"cutoff_ratio": 0.55, "noise_strength": 0.5}
        ),
    }

    print("\n[3/4] Инициализация модуля оценки качества и Оркестратора...")
    evaluator = ImageInversionEvaluator(device=device)

    pipeline = EvaluationPipeline(
        methods_dict=methods_dict,
        evaluator=evaluator,
        device=device
    )

    print("\n[4/4] Загрузка данных и старт расчета...")
    pipeline.load_data(subsets=args.subsets, split=args.split)

    # Запускаем конвейер
    results_df = pipeline.run_evaluation(
        results_dir="results",
        output_csv="evaluation_results.csv"
    )

    print("\n=== Процесс завершён ===")
    if results_df is not None and not results_df.empty:
        total_runs = len(results_df)
        errors_df = results_df[results_df['error'].notna()]
        total_errors = len(errors_df)
        successful_runs = total_runs - total_errors

        print(f"Всего генераций: {total_runs}")
        print(f"Успешно: {successful_runs}")
        print(f"Ошибок: {total_errors}")

        if total_errors > 0:
            print("Подробности об ошибках можно найти в папке results/errors/")
    else:
        print("Нет результатов для анализа.")


if __name__ == "__main__":
    main()