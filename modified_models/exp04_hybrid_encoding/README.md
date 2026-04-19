# EXP-04: Hybrid Positional + Hash Encoding

**Источник:** Wang, Gong, Zeng. *"Hyb-NeRF: A Multiresolution Hybrid Encoding for Neural Radiance Fields"*. WACV 2024  
**Офиц. реализация:** Отсутствует

## Идея
Использование **позиционного кодирования** (positional encoding) для низкочастотных уровней
и **hash grid encoding** для высокочастотных уровней.

**Ключевая идея:**
- Низкочастотные уровни (low-frequency) отвечают за общую форму сцены
- Высокочастотные уровни (high-frequency) отвечают за мелкие детали
- Hybrid Encoding сочетает преимущества обоих подходов

## Архитектурное изменение
- В `GridEncoder` добавлен параметр `use_hybrid_encoding`
- В `GridEncoder.forward` используется `hybrid_encoding` функция
- Для низкочастотных уровней применяется positional encoding
- Для высокочастотных уровней применяется hash grid encoding

## Файлы
- `network.py` — GridEncoder с hybrid_encoding
- `main_nerf.py` — обучение с hybrid_encoding

## Пример использования
```bash
python main_nerf.py data/lego \
    --workspace logs/exp04_hybrid \
    --iters 10000 \
    --use_hybrid_encoding
```
