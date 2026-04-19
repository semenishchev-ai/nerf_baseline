# torch-ngp (ashawkey)

Эта реализация Instant-NGP на чистом PyTorch с использованием кастомных CUDA-расширений для кодирования и рендеринга.

## Особенности
- **Высокая производительность:** Обучение на сцене Lego занимает ~3 минуты.
- **Качество:** PSNR ~32.53 dB.

## Команды (через run.py)
```bash
# Обучение
python run.py --model torch_ngp --mode train

# Оценка
python run.py --model torch_ngp --mode eval --workspace torch_ngp/trial_lego
```

## Структура
- `nerf/`: Ядро модели и рендерера.
- `gridencoder/`, `raymarching/` и др.: CUDA расширения.
- `main_nerf.py`: Основной скрипт запуска.
