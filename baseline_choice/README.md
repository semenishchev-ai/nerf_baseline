# NeRF Baseline Choice Benchmarks

Этот репозиторий содержит набор эталонных реализаций NeRF для сравнения производительности и качества на синтетических данных (Blender Lego).

## Структура репозитория

- `common/`: Общий код (метрики PSNR, SSIM, LPIPS).
- `torch_ngp/`: Оптимизированная реализация Instant-NGP на PyTorch.
- `hash_nerf/`: Реализация NeRF с хеш-кодированием.
- `nerfacto/`, `instant_ngp/`, `tensorf/`: Конфигурации для Nerfstudio моделей.
- `run.py`: Единая точка входа для запуска обучения и оценки.

## Настройка окружения

Рекомендуется использовать Conda для управления зависимостями.

```bash
conda create -n baseline_ngp python=3.10
conda activate baseline_ngp
bash setup_env.sh
```

## Использование

Для запуска любой модели используйте `run.py`:

```bash
# Обучение torch-ngp
python run.py --model torch_ngp --mode train

# Оценка hash_nerf
python run.py --model hash_nerf --mode eval
```

### Доступные модели:
- `torch_ngp`
- `hash_nerf`
- `nerfacto`
- `instant_ngp`
- `tensorf`

## Логи и результаты

Все логи, чекпоинты и результаты рендеринга сохраняются в соответствующие подпапки в директории `logs/`:

- `train.log`: Полный текстовый лог вывода консоли (stdout/stderr). Позволяет отслеживать прогресс и отлаживать ошибки.
- `psnr_plot.png`: Автоматически генерируемый график PSNR (или Loss, если PSNR еще не вычислен). Позволяет визуально оценить сходимость модели.
- `eval.json` (или аналоги): Финальные метрики после запуска в режиме `--mode eval`.
- `results/`: Видео (.mp4) или изображения с результатами рендеринга тестовой выборки.

## Метрики

Все модели используют единую реализацию метрик из `common/metrics.py` (PSNR, SSIM, LPIPS на базе AlexNet) для обеспечения честности сравнения.
