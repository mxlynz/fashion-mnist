| Arch | Run | Änderung | Bester Val-Loss | Val-Acc | Dauer |
|---|---|---|---|---|---|
| lenet | 00_baseline | – | ~0.297 (Ep. 7) | 88.95 % | unsicher |
| lenet | 01_bs64_lr0.01 | bs 64, SGD lr 0.01 | 0.2995 (Ep. 9) | 89.30 % | 3.4 min |
| lenet | 02_bs64_adam_lr0.01 | Adam, lr 0.01 ✗ | 0.363 (Ep. 10) | 87.35 % | 3.6 min |
| lenet | 02_bs64_adam_lr0.001 | Adam, lr 0.001 | 0.2911 (Ep. 9) | 89.87 % | 3.5 min |
| lenet | 03_bs64_adam_norm | + echte Normalisierung | 0.2796 (Ep. 9) | 89.47 % | 4.4 min |
| v2 | 01_bs64_adam_norm | v2-Baseline (BN, 5 Conv, 3×3) | 0.2820 (Ep. 10) | 89.92 % | 5.0 min |
| v2 | 02_bs64_adam_norm_ep30 | 30 Epochen | 0.2668 (Ep. 16) | 90.38 % | 13.9 min |