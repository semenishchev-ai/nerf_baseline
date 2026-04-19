# EXP-02: Rotated Hash Grid Encoding

**Источник:** Wang, Wang, Zhang, et al. *"Rotated Hash Grid Encoding for Neural Radiance Fields"*. arXiv:2507.02266  
**Офиц. реализация:** Отсутствует

## Идея
Вместо стандартного трилинейного сэмплирования (bilinear interpolation) в hash grid encoding
используется **ротированное трилинейное сэмплирование** (rotated trilinear interpolation).

**Ключевая идея:**
- В оригинальном NeRF трилинейное сэмплирование использует 8 угловых вокселей
- Rotated Hash Grid использует 8 **повернутых** угловых вокселей
- Поворот выполняется с помощью обучаемой матрицы $R_l$ для каждого уровня $l$
- Это позволяет лучше захватывать анизотропные структуры сцены

## Архитектурное изменение
- В `GridEncoder` добавлено поле `rotation_matrices` (torch.nn.Parameter)
- В `GridEncoder.forward` используется `rotated_trilinear_interp` вместо стандартного `trilinear_interp`
- Добавлено 3 новых параметра на каждый уровень (матрица $3 \times 3$)

## Файлы
- `network.py` — модифицированный NeRFNetwork с Rotated Hash Grid
- `main_nerf.py` — точка входа с корректными путями
