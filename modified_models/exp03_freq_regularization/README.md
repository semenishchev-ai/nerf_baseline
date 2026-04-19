# EXP-03: Frequency Regularization (G2fR-style)

**Источник:** Xie, Zhou, Sakurada, Ishikawa, Onishi, Oishi. *"G2fR: Frequency Regularization in Grid-based Feature Encoding Neural Radiance Fields"*. ECCV 2024  
**Офиц. реализация:** Отсутствует

## Идея
В оригинальном torch-ngp hash grid encoding разные уровни имеют разные разрешения (например, 16, 32, 64, 128).
Из-за этого на разных уровнях кодируются разные частотные диапазоны.

**Ключевая идея:**
- Добавить L2-регуляризацию на разницу спектров между уровнями
- Это заставляет hash grid кодировать схожие частоты на всех уровнях
- Улучшает качество реконструкции, особенно на текстурированных поверхностях

## Архитектурное изменение
- В `GridEncoder` добавлен метод `compute_frequency_loss()`
- В `train_nerf` добавлен `freq_reg_weight` параметр
- Добавлен `freq_reg_type` для выбора типа регуляризации:
  - `'l2_diff'` — L2 разница между спектрами (по умолчанию)
  - `'l2_abs'` — L2 разница абсолютных значений
  - `'l1_abs'` — L1 разница абсолютных значений

## Файлы
- `network.py` — GridEncoder с compute_frequency_loss
- `main_nerf.py` — обучение с freq_reg_weight и freq_reg_type

## Пример использования
```bash
python main_nerf.py data/lego \
    --workspace logs/exp03_freq_reg \
    --iters 10000 \
    --freq_reg_weight 0.01 \
    --freq_reg_type l2_diff
```
