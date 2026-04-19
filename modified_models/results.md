# Сравнение: Baseline vs New Model V1

## Запуски на 10,000 и 30,000 итераций (Synthetic Lego Dataset)

**В `new_model_v1` объединены:**
- **EXP-05 (Enhanced MLP):** 128 hidden dim, 4/3 layers, GELU, skip connections
- **EXP-06 (Scale-Aware Weighting):** Distance-based hash grid masking (Anti-Aliasing)

### Результаты на Test Set

| Метрика | Baseline (10k) | Baseline (30k) | New Model V1 (10k) | New Model V1 (30k) |
|---|---|---|---|---|
| **PSNR** | 32.53 dB | 34.31 dB | 32.90 dB | **34.40 dB** |
| **SSIM** | 0.967 | 0.976 | 0.968 | 0.976 |
| **LPIPS** | 0.019 | 0.0124 | 0.019 | 0.0128 |

### PSNR на валидации (Convergence tracking)

По графику (./logs/comparative_psnr_plot.png):
- **Baseline (30k)** выходит на стабильное плато Validation PSNR ~ 35.40 dB
- **New Model V1 (30k)** достигает Validation PSNR ~ **35.55 dB**

Модификация стабильно пробивает потолок Baseline на ~0.10 - 0.15 dB на больших итерациях, так же как давала ~0.37 dB прироста (в сумме) на ранних этапах (10k).
