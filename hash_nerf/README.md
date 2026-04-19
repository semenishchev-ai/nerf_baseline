# Hash-NeRF (pytorch)

Реализация NeRF с хеш-кодированием на базе PyTorch.

## Особенности
- **Медленное обучение:** ~1.5 - 2 часа для 10,000 итераций.
- **Качество:** PSNR ~27.85 dB.

## Команды (через run.py)
```bash
# Обучение
python run.py --model hash_nerf --mode train

# Оценка
python run.py --model hash_nerf --mode eval --workspace bench_lego_10k
```

## Структура
- `run_nerf.py`: Основной скрипт обучения и рендеринга.
- `hash_encoding.py`: Реализация хеш-таблиц.
- `configs/`: Конфигурационные файлы для разных сцен.
