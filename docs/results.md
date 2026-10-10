# Paper results

The exact Figure 1 supplied with the paper is
[`method_pipeline.png`](../figures/method_pipeline.png). It is the only stored figure
image. Figure 2 and 3 data can be regenerated from the checked-in per-document scores
with `python scripts/make_paper_figures.py`; their source CSVs are in
[`results/figure_data/`](../results/figure_data/).

## Main AUROC

| Setting | Method | C4 | PubMed | SQuAD | Writing | XSum | Macro |
|---|---|---:|---:|---:|---:|---:|---:|
| White-box | Likelihood | 0.5707 | 0.5727 | 0.5770 | 0.6714 | 0.5537 | 0.5891 |
| White-box | Entropy | 0.5500 | 0.5575 | 0.5474 | 0.5303 | 0.5761 | 0.5523 |
| White-box | Rank | 0.5929 | 0.5919 | 0.5872 | 0.6193 | 0.5777 | 0.5938 |
| White-box | LogRank | 0.5951 | 0.6014 | 0.5984 | 0.6909 | 0.5779 | 0.6127 |
| White-box | LRR | 0.6553 | 0.6635 | 0.6473 | 0.7134 | 0.6215 | 0.6602 |
| White-box | Fast-DetectGPT | 0.7681 | 0.7502 | 0.7591 | 0.8589 | 0.7289 | 0.7730 |
| White-box | AdaDetectGPT | 0.8253 | 0.7987 | 0.8157 | 0.9026 | 0.7797 | 0.8244 |
| White-box | Ours | **0.8447** | **0.8087** | **0.8261** | **0.9120** | **0.7956** | **0.8374** |
| Black-box | Fast-DetectGPT | 0.7034 | 0.5763 | 0.6939 | 0.7993 | 0.6477 | 0.6841 |
| Black-box | AdaDetectGPT | 0.7623 | 0.6492 | 0.7444 | **0.8419** | 0.7129 | 0.7421 |
| Black-box | Ours | **0.7776** | **0.7176** | **0.7773** | 0.8404 | **0.7377** | **0.7701** |

The 320 per-configuration entries underlying Tables 1–4 are in
[`results/paper_auroc.csv`](../results/paper_auroc.csv); rebuild with
`python scripts/build_paper_tables.py`.

Ours beats AdaDetectGPT in 20/20 white-box and 17/20 black-box configurations. The
paired configuration-level mean differences have 95% bootstrap intervals [0.0101,
0.0160] and [0.0169, 0.0411], respectively. One-sided Wilcoxon signed-rank p-values are
9.54e-7 and 2.38e-5.

## Fixed-FPR evaluation

In white-box evaluation, empirical powers at 1%/5% FPR are 21.22%/43.06% for Ours,
17.47%/38.35% for AdaDetectGPT, and 10.77%/27.52% for Fast-DetectGPT. After independent
human calibration, Ours reaches 17.72%/40.02% power at realized FPRs 0.84%/4.40%.

Paper Table 5 reports the following macro means for the primary `T_theta` score:

| Setting | pAUC@5% | FPR@1% | Power@1% | FPR@5% | Power@5% |
|---|---:|---:|---:|---:|---:|
| White-box | 0.6424 | 0.0084 | 0.1772 | 0.0440 | 0.4002 |
| Black-box | 0.1807 | 0.0105 | 0.0999 | 0.0460 | 0.2714 |

Paper Table 6 reports the white-box source-model averages (each over five datasets):

| Source model | Method | pAUC@5% | FPR@1% | Power@1% | FPR@5% | Power@5% |
|---|---|---:|---:|---:|---:|---:|
| LLaMA-2-7B | Fast-DetectGPT | 0.5458 | 0.0056 | 0.0516 | 0.0540 | 0.2000 |
| LLaMA-2-7B | AdaDetectGPT | 0.5866 | 0.0076 | 0.0864 | 0.0532 | 0.3008 |
| LLaMA-2-7B | Ours | 0.6031 | 0.0088 | 0.1252 | 0.0460 | 0.3104 |
| OPT-2.7B | Fast-DetectGPT | 0.5962 | 0.0056 | 0.1144 | 0.0512 | 0.3400 |
| OPT-2.7B | AdaDetectGPT | 0.6455 | 0.0064 | 0.1712 | 0.0456 | 0.4252 |
| OPT-2.7B | Ours | 0.6760 | 0.0084 | 0.2348 | 0.0468 | 0.4772 |
| OPT-6.7B | Fast-DetectGPT | 0.5884 | 0.0088 | 0.1072 | 0.0452 | 0.2916 |
| OPT-6.7B | AdaDetectGPT | 0.6269 | 0.0080 | 0.1336 | 0.0420 | 0.3720 |
| OPT-6.7B | Ours | 0.6525 | 0.0096 | 0.1984 | 0.0376 | 0.4184 |
| Falcon-7B | Fast-DetectGPT | 0.5698 | 0.0092 | 0.0792 | 0.0432 | 0.2532 |
| Falcon-7B | AdaDetectGPT | 0.6141 | 0.0084 | 0.1260 | 0.0404 | 0.3388 |
| Falcon-7B | Ours | 0.6379 | 0.0068 | 0.1504 | 0.0456 | 0.3948 |

## Ablations and diagnostics

- `results/whitebox/factorial/` and `results/blackbox/factorial/` contain all eight H/S/C
  variants and reproduce the paper's component factorial tables.
- `results/gaussian_diagnostics/` contains all 20 configurations, normality tests,
  held-out power predictions, operating points, and the final run manifest.
- Mean absolute Gaussian power-prediction errors are 0.0393 at 1% FPR and 0.0288 at 5%
  FPR; mean absolute predicted-versus-empirical AUROC error is 0.0043.

## Deployment score: paper Table 8

All values below are the macro mean of `T_theta - S_theta` across 20 configurations.
They are recalculated from per-cell CSVs, including the corrected LLaMA independent
white-box cells and both fresh confirmation settings. The exact unrounded values are
in [`results/deployment_ablation/table_8.csv`](../results/deployment_ablation/table_8.csv);
rebuild with `python scripts/build_table_8.py`.

| Setting | Split | AUROC | PR-AUC | pAUC@5% | Power@1% | Power@5% |
|---|---|---:|---:|---:|---:|---:|
| White | Independent | +0.0034 | +0.0076 | +0.0123 | +0.0366 | +0.0239 |
| White | Fresh | +0.0045 | +0.0106 | +0.0257 | +0.0308 | +0.0105 |
| Black | Independent | +0.0033 | +0.0062 | +0.0111 | +0.0181 | +0.0242 |
| Black | Fresh | +0.0038 | +0.0077 | +0.0111 | +0.0201 | +0.0229 |
