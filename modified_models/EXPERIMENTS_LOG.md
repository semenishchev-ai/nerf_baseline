# Журнал экспериментов — Модификации архитектуры torch-ngp

## Baseline

| Метрика | Значение |
|---------|----------|
| PSNR    | 32.53 dB |
| SSIM    | 0.967    |
| LPIPS   | 0.019    |
| Время обучения | ~3 мин |
| GPU     | 1× A100  |
| Итерации | 10,000  |
| Сцена   | Lego (Blender Synthetic) |

---

## EXP-01: Spatially-Adaptive Hash Grid Masking
- **Источник:** Walker, Mariotti, Vaxman, Bilen. *"Spatially-Adaptive Hash Encodings for Neural Surface Reconstruction"*. WACV 2025. arXiv:2412.05179
- **Офиц. реализация:** Отсутствует
- **Идея:** Обучаемые маски подавляют вклад отдельных уровней hash grid в зависимости от пространственного контекста.
- **Результаты:**

| PSNR | SSIM | LPIPS | Время | Δ PSNR |
|------|------|-------|-------|--------|
| 32.41| 0.965| 0.023 | ~3 мин| -0.12  |

---

## EXP-02: Rotated Multi-Resolution Hash Encoding (R-MHE)
- **Источник:** Dai, Fan. *"Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings"*. ICLR 2026. arXiv:2602.10495
- **Офиц. реализация:** Отсутствует
- **Идея:** Применение различных матриц поворота к входным координатам на каждом уровне для декоррелирования коллизий.
- **Результаты:**

| PSNR | SSIM | LPIPS | Время | Δ PSNR |
|------|------|-------|-------|--------|
| 32.55| 0.967| 0.020 | ~3 мин| +0.02  |

---

## EXP-03: Frequency Regularization (G2fR-style)
- **Источник:** Xie, Zhou, Sakurada, Ishikawa, Onishi, Oishi. *"G2fR: Frequency Regularization in Grid-based Feature Encoding Neural Radiance Fields"*. ECCV 2024
- **Офиц. реализация:** Отсутствует
- **Идея:** Регуляризация несогласованного распределения частот между уровнями hash grid.
- **Результаты:**

| PSNR | SSIM | LPIPS | Время | Δ PSNR |
|------|------|-------|-------|--------|
| 31.20| 0.952| 0.031 | ~3 мин| -1.33  |

---

## EXP-04: Hybrid Positional + Hash Encoding
- **Источник:** Wang, Gong, Zeng. *"Hyb-NeRF: A Multiresolution Hybrid Encoding for Neural Radiance Fields"*. WACV 2024
- **Офиц. реализация:** Отсутствует
- **Идея:** Frequency encoding для грубых уровней + hash grid для тонких деталей.
- **Результаты:**

| PSNR | SSIM | LPIPS | Время | Δ PSNR |
|------|------|-------|-------|--------|
| 32.29| 0.964| 0.022 | ~3 мин| -0.24  |

---

## EXP-05: Enhanced MLP Decoder с Residual Connections
- **Источник:** He et al. *"Deep Residual Learning"* (CVPR 2016) — архитектурный принцип; применение к NeRF MLP — общий тренд (NGP-RT, arXiv 2024, без офиц. кода).
- **Офиц. реализация для NGP:** Отсутствует
- **Идея:** Расширить tiny MLP decoder (64→128, +skip connections, GELU).
- **Результаты:**

| PSNR | SSIM | LPIPS | Время | Δ PSNR |
|------|------|-------|-------|--------|
| 32.80| 0.967| 0.021 | ~3 мин| +0.27  |

---

## EXP-06: Scale-Aware Level Weighting (Anti-Aliasing)
- **Источник:** Barron et al. *"Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields"*. ICCV 2023 — упрощённая версия scale-aware weighting без cone tracing.
- **Офиц. реализация (PyTorch для torch-ngp):** Отсутствует
- **Идея:** Взвешивать уровни hash grid по расстоянию от камеры (soft anti-aliasing).
- **Результаты:**

| PSNR | SSIM | LPIPS | Время | Δ PSNR |
|------|------|-------|-------|--------|
| 32.67| 0.967| 0.020 | ~3.5 м| +0.14  |

---

## EXP-07: Opacity Regularization + Total Variation Loss
- **Источник:** Barron et al. *"Mip-NeRF 360"* (CVPR 2022) — distortion/opacity loss; TV reg — стандарт (TensoRF, DVGO).
- **Офиц. реализация для torch-ngp:** Отсутствует
- **Идея:** Opacity regularization + total variation для cleaner geometry.
- **Результаты:**

| PSNR | SSIM | LPIPS | Время | Δ PSNR |
|------|------|-------|-------|--------|
| 32.18| 0.963| 0.023 | ~3.5 м| -0.35  |

---

## EXP-08: Stochastic Coordinate Jittering (SCJ)
- **Идея:** Добавление случайного шума к координатам (±1/2048) перед хеш-кодированием в режиме обучения.
- **Результаты (10k iters):**

| PSNR | SSIM | LPIPS | Δ PSNR |
|------|------|-------|--------|
| 32.34| 0.965| 0.022 | -0.19  |

---

## EXP-09: Progressive Frequency Masking (PFM)
- **Идея:** Постепенное включение уровней хеш-сетки (1 -> 16) в течение первых 5000 шагов.
- **Результаты (10k iters):**

| PSNR | SSIM | LPIPS | Δ PSNR |
|------|------|-------|--------|
| 32.10| 0.964| 0.022 | -0.43  |

---

## EXP-10: Gated MLP Decoder (GMLP)
- **Идея:** Замена стандартных линейных слоев в декодере на Gated Linear Units (GLU).
- **Результаты (10k iters):**

| PSNR | SSIM | LPIPS | Δ PSNR |
|------|------|-------|--------|
| 32.51| 0.966| 0.023 | -0.02  |

---

## EXP-11: Learnable Density Shift (LDS)
- **Идея:** Добавление глобального обучаемого параметра смещения плотности для адаптации к прозрачности сцены.
- **Результаты (10k iters):**

| PSNR | SSIM | LPIPS | Δ PSNR |
|------|------|-------|--------|
| 32.48| 0.966| 0.020 | -0.05  |

---

## EXP-12: Normal-Oriented View Encoding (NOVE)
- **Идея:** Вычисление нормалей (finite difference) и вектора отражения для подачи в Color MLP.
- **Результаты (10k iters):**

| PSNR | SSIM | LPIPS | Δ PSNR |
|------|------|-------|--------|
| 32.43| 0.965| 0.021 | -0.10  |

---

## EXP-13_1: Learnable Error Predictor (LRS-A)
- **Идея:** Отдельная MLP обучается предсказывать MSE ошибку луча. Предсказание используется для обновления `error_map`.
- **Результаты (10k iters):**

| PSNR | SSIM | LPIPS | Δ PSNR |
|------|------|-------|--------|
| 32.13| 0.963| 0.023 | -0.40  |

---

## EXP-13_2: Boundary-Aware Sampler (LRS-C)
- **Идея:** Приоритизация сэмплирования лучей на границах объектов (где `weights_sum` ≈ 0.5).
- **Результаты (10k iters):**

| PSNR | SSIM | LPIPS | Δ PSNR |
|------|------|-------|--------|
| **33.82** | **0.973**| **0.015** | **+1.29** |

---

## Итоговый анализ

Эксперимент **EXP-13_2 (Boundary-Aware Sampler)** показал лучший результат.

### Основные выводы:
1.  **Победитель — EXP-13_2**: Прирост **+1.29 dB PSNR** подтверждает гипотезу о том, что работа с сэмплированием лучей — ключ к высокому качеству. Фокусировка на границах (где прозрачность не 0 и не 1) позволяет модели тратить бюджет на самые сложные участки (тонкие детали, края), что критично для сцен типа Lego.

## FINAL BENCHMARK: Statistical Comparison (30,000 iters)
*Сцена: Lego | 10 запусков на конфиг*

Финальное сравнение для подтверждения статистической значимости улучшений.

| Конфигурация | PSNR (avg) | SSIM (avg) | LPIPS (avg) | Время (мин) |
|--------------|------------|------------|-------------|-------------|
| **Baseline** | 33.96      | 0.9741     | 0.0141      | 11.7        |
| **EXP-13_2**    | **34.68**  | **0.9771** | **0.0115**  | **10.3**    |

### Итоговые выводы:
1.  **EXP-13_2** является оптимальной архитектурой. Прирост в **+0.72 dB** над сильным Baseline на 30к итерациях при **снижении времени обучения на 12%**. 
3.  **LPIPS**: Значительное улучшение LPIPS (0.014 -> 0.011) визуально выражается в отсутствии "зубчатости" на краях и более чистых тенях.
