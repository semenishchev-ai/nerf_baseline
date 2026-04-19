# EXP-01: Spatially-Adaptive Hash Grid Masking

**Источник:** Walker, Mariotti, Vaxman, Bilen. *"Spatially-Adaptive Hash Encodings for Neural Surface Reconstruction"*. WACV 2025. arXiv:2412.05179  
**Офиц. реализация:** Отсутствует

## Идея
Обучаемая маленькая MLP-сеть (`MaskNet`) предсказывает per-level веса для hash grid encoding
в зависимости от 3D-координат точки. Это позволяет модели подавлять шумные уровни в гладких
областях сцены и усиливать нужные уровни в детализированных областях.

## Архитектурное изменение
- Добавлена `MaskNet`: MLP (3 → 32 → 16), sigmoid-активация на выходе
- Выход GridEncoder [B, L*C] → reshape [B, L, C] → умножение на маски → reshape обратно
- ~1.1K дополнительных параметров

## Файлы
- `network.py` — модифицированный NeRFNetwork с MaskNet
- `main_nerf.py` — точка входа с корректными путями
