import time
import torch

class PerformanceMonitor:
    """
    Контекстный менеджер для замера времени выполнения
    и пикового потребления видеопамяти (VRAM).
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.execution_time = 0.0
        self.peak_vram_mb = 0.0

    def __enter__(self):
        # Очищаем кэш и сбрасываем пиковые значения CUDA перед замером
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        # 1. Время в секундах
        self.execution_time = self.end_time - self.start_time

        # 2. Пиковое потребление VRAM в мегабайтах
        if torch.cuda.is_available():
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)