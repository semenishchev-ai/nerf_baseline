# EXP-06: Antialiased Hash Grid Encoding

**Источник:** Wang, Wang, Zhang, et al. *"Antialiased Hash Grid Encoding for Neural Radiance Fields"*. arXiv:2507.02266  
**Офиц. реализация:** Отсутствует

## Идея
В оригинальном torch-ngp hash grid encoding используется трилинейное сэмплирование, которое
может вызывать алиасинг (aliasing) на высоких частотах.

**Ключевая идея:**
- Использовать **антиалиасинговое трилинейное сэмплирование** (antialiased trilinear interpolation)
- Это достигается за счет использования **фильтра Гаусса** (Gaussian filter) при интерполяции
- Уменьшает алиасинг и улучшает качество реконструкции

## Архитектурное изменение
- В `GridEncoder` модифицирован метод `trilinear_interp`
- Добавлен параметр `use_gaussian_filter` для включения/выключения
- Реализовано гауссово сглаживание для уменьшения алиасинга

## Файлы
- `network.py` — GridEncoder с antialiased trilinear interpolation
- `main_nerf.py` — точка входа

## Пример использования
```bash
python main_nerf.py data/lego \
    --workspace logs/exp06_antialias \
    --iters 10000 \
    --use_gaussian_filter
```
