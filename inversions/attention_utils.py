import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class StoreCrossAttnProcessor:
    """
    Кастомный процессор внимания для diffusers.
    Вычисляет внимание стандартным способом, но дополнительно сохраняет карту вероятностей.
    """

    def __init__(self):
        self.attentions = []

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        # Если encoder_hidden_states передан, значит это Cross-Attention (Текст -> Картинка)
        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Вычисляем саму матрицу внимания
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # СОХРАНЯЕМ ТОЛЬКО CROSS-ATTENTION (чтобы не забивать память Self-Attention)
        if is_cross_attention:
            self.attentions.append(attention_probs.detach().cpu())

        # Завершаем стандартный проход
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CrossAttentionManager:
    """
    Менеджер для удобного подключения и отключения перехвата карт внимания.
    """

    def __init__(self, unet):
        self.unet = unet
        self.processor = StoreCrossAttnProcessor()
        self.original_processors = {}

    def attach(self):
        """Подменяет процессоры в UNet на наш перехватчик."""
        self.processor.attentions.clear()
        self.original_processors = self.unet.attn_processors

        custom_processors = {}
        for name, proc in self.unet.attn_processors.items():
            # Нас интересуют только слои 'attn2' (Cross-Attention)
            if 'attn2' in name:
                custom_processors[name] = self.processor
            else:
                custom_processors[name] = proc

        self.unet.set_attn_processor(custom_processors)

    def detach(self):
        """Возвращает оригинальные процессоры (например, xformers)."""
        self.unet.set_attn_processor(self.original_processors)
        self.processor.attentions.clear()

    def get_mask_for_token(self, token_index: int, threshold: float = 0.3, resolution: int = 128, device="cuda"):
        """
        Усредняет собранные карты для нужного слова и возвращает бинарную маску фона.
        """
        if not self.processor.attentions:
            raise ValueError("Карты внимания не собраны! Сначала сделайте forward pass.")

        attn_maps = []
        for attn in self.processor.attentions:
            # attn shape: (batch * heads, seq_len, context_len)
            # context_len — это 77 токенов промпта. Берем колонку нашего слова!
            token_attn = attn[:, :, token_index].mean(dim=0)  # Усредняем по головам и батчу

            # seq_len — это пиксели (h * w). Превращаем обратно в квадрат.
            h = int(token_attn.shape[0] ** 0.5)
            if h * h != token_attn.shape[0]:
                continue  # Пропускаем неквадратные матрицы (иногда бывают в SDXL)

            token_map = token_attn.view(1, 1, h, h)

            # Масштабируем до 128x128 (размер латентов SDXL)
            token_map = F.interpolate(token_map, size=(resolution, resolution), mode='bilinear')[0, 0]
            attn_maps.append(token_map)

        # Усредняем по всем слоям UNet
        avg_attn = torch.stack(attn_maps).mean(dim=0)

        # Нормализуем от 0 до 1, чтобы порог работал четко
        avg_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)

        # Бинаризуем (1 - объект, 0 - фон)
        object_mask = (avg_attn > threshold).float()

        # Инвертируем: нам нужен фон для защиты (1 - фон, 0 - объект)
        bg_mask = (1.0 - object_mask).to(device)

        return bg_mask