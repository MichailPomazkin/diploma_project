import time
import torch

class PerformanceMonitor:
    """
    Контекстный менеджер для точного замера времени выполнения
    и пикового потребления видеопамяти (VRAM) с учетом асинхронности CUDA.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.execution_time = 0.0
        self.peak_vram_mb = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Ждем завершения всех фоновых задач на GPU перед стартом секундомера
            torch.cuda.synchronize()

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            # Ждем пока GPU реально закончит генерацию картинки
            torch.cuda.synchronize()
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time