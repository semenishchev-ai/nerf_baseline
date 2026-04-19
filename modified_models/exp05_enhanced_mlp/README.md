# EXP-05: Enhanced MLP Architecture

**Источник:** На основе torch-ngp с улучшениями из SNeRF и других работ
**Офиц. реализация:** Отсутствует (авторская модификация)

## Идея
Улучшение архитектуры MLP для лучшего качества реконструкции:
1. Увеличение количества слоев
2. Добавление skip-connections
3. Использование более эффективных активаций
4. Улучшенная обработка feature vector

## Архитектурное изменение
- В `NeRFNetwork` модифицирован `forward` метод
- Добавлены skip-connections между слоями
- Увеличина hidden_dim с 64 до 128
- Добавлен дополнительный MLP-блок для feature vector

## Файлы
- `network.py` — NeRFNetwork с enhanced MLP
- `main_nerf.py` — точка входа

## Пример использования
```bash
python main_nerf.py data/lego \
    --workspace logs/exp05_enhanced_mlp \
    --iters 10000
```
