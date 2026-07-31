# Current validated DiffusionDB TR results

Only complete, validated 1001-sample results are included.

| Variant | N | Attacked TPR @ original threshold | Attacked TPR @ recalibrated threshold | Attack success @ original threshold | Attack success @ recalibrated threshold | Attacked ROC-AUC | FID | CLIP | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DDIM no shift + no color | 1001 | 0.998002 | 0.998002 | 0.001998 | 0.001998 | 0.999931 | 6.530667 | 0.459354 | 33.222941 | 0.935582 |
| DDPM nearest + shift + aligned color | 1001 | 0.004995 | 0.065934 | 0.995005 | 0.934066 | 0.677154 | 28.985595 | 0.458448 | 23.731504 | 0.687744 |
| DDIM inverse + DDPM forward + nearest + shift + aligned color | 1001 | 0.018981 | 0.082917 | 0.981019 | 0.917083 | 0.747935 | 27.144841 | 0.458042 | 25.798094 | 0.769073 |
| DDIM bilinear + shift + aligned color | 1001 | 0.023976 | 0.065934 | 0.976024 | 0.934066 | 0.796764 | 45.011472 | 0.446226 | 24.157994 | 0.670867 |
| DDIM bilinear + shift + no color | 1001 | 0.091908 | 0.076923 | 0.908092 | 0.923077 | 0.784604 | 47.736092 | 0.441377 | 24.243971 | 0.675593 |
| DDIM nearest + shift + no color | 1001 | 0.124875 | 0.132867 | 0.875125 | 0.867133 | 0.797839 | 23.672713 | 0.459853 | 31.134245 | 0.919305 |
| DDIM nearest + shift + paper-exact unaligned color | 1001 | 0.106893 | 0.347652 | 0.893107 | 0.652348 | 0.877266 | 46.503007 | 0.447976 | 22.075174 | 0.783325 |
| DDIM nearest + shift + aligned color | 1001 | 0.103896 | 0.149850 | 0.896104 | 0.850150 | 0.841210 | 23.512190 | 0.460506 | 30.466041 | 0.915708 |
| DDIM centered bilinear + shift + aligned color | 1001 | 0.027972 | 0.076923 | 0.972028 | 0.923077 | 0.795732 | 43.632362 | 0.447604 | 23.300199 | 0.626961 |
| DDIM bilinear + shift + paper-exact unaligned color | 1001 | 0.106893 | 0.409590 | 0.893107 | 0.590410 | 0.897796 | 62.302573 | 0.426722 | 20.004805 | 0.574555 |