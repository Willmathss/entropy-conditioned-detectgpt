# Artifact map and remaining gaps

Only experiments reported in the paper belong in the main reproduction path. The
historical HeavyWater, E0/E1, Simplex, old 128-candidate comparison, and obsolete LLaMA
test outputs have been left out.

| Paper artifact | Code | Included output/input |
|---|---|---|
| Figure 1 | supplied final artwork | `figures/method_pipeline.png` |
| Independent text construction | `experiments/generate_independent_whitebox_test.py` | `data/{train,test}/{sources,text}/` |
| White-box cache and train-only fit | `src/build_whitebox_reference_cache.py`, `experiments/evaluate_whitebox_reference.py` | text inputs and `data/fits/whitebox/` |
| White-box Ours and 2³ factorial | `experiments/run_whitebox_factorial_ltheta.py` | `results/whitebox/factorial/` |
| White-box Fast/Ada | `experiments/evaluate_whitebox_reference.py` | `results/whitebox/conditional_baselines/` |
| White-box probability/rank baselines | `experiments/compute_whitebox_baseline_power.py` | `results/whitebox/probability_rank/` |
| Tables 1 and 3 | `scripts/build_paper_tables.py` | 20 cells × 8 methods under `results/whitebox/` and `results/paper_auroc.csv` |
| Black-box Fast/Ada and main detector | `experiments/run_blackbox_local.py`, `experiments/detect_tensor_rayleigh_blackbox.py` | `results/blackbox/conditional_baselines/` and text adapters under `data/` |
| Black-box Ours and 2³ factorial | `experiments/run_blackbox_factorial_ablation.py`, `experiments/run_blackbox_factorial_ltheta.py` | `results/blackbox/factorial/` |
| Black-box probability/rank baselines | `experiments/run_detectllm_blackbox_five.py` | `results/blackbox/probability_rank/` |
| Tables 2 and 4 | `scripts/build_paper_tables.py` | 20 cells × 8 methods under `results/blackbox/` and `results/paper_auroc.csv` |
| Tables 5 and 6 fixed-FPR operating points | frozen independent human calibration in the evaluation runners | per-cell metrics under `results/{whitebox,blackbox}/`; Table 5–6 summaries in `docs/results.md` and the paper |
| Figure 2 low-FPR ROC | `scripts/make_paper_figures.py` | `results/figure_data/figure_2_*.csv`; images are generated only on request with `--plot-dir` |
| Figure 3 and Table 7 Gaussian diagnostics | `experiments/gaussian_score_diagnostics.py`, `scripts/make_paper_figures.py` | `results/gaussian_diagnostics/`, `results/figure_data/figure_3_*.csv` |
| Tables 9 and 10 component factorials | `experiments/run_{whitebox,blackbox}_factorial_ltheta.py` | all eight H/S/C variants in `results/{whitebox,blackbox}/factorial/` |
| Table 8: `T_theta` versus `S_theta` | `experiments/compare_*_ltheta_stheta.py`, `scripts/build_table_8.py` | all four `results/deployment_ablation/{whitebox_independent,blackbox_independent,fresh/whitebox,fresh/blackbox}/` bundles and `table_8.csv` |
| Fresh confirmation source selection/generation | `experiments/{prepare_fresh_source_split,generate_fresh_confirmation,validate_fresh_confirmation}.py` | `data/fresh/{sources,text}/` and lock files |
| Fresh confirmation rescoring | `experiments/{evaluate_whitebox_ltheta_confirmatory,evaluate_blackbox_ltheta_confirmatory}.py` | `results/deployment_ablation/fresh/{whitebox,blackbox}/` |
| Paired bootstrap and Wilcoxon | `scripts/paired_comparisons.py` | script output from 20 configuration scores |

White-box conditional baselines use corrected LLaMA results from
`whitebox_independent_llama_fixed_20260821`; the other models use the independent
reference run. Select `Fast-DetectGPT-10000` and `AdaDetectGPT-exact`. Historical
`Fast-DetectGPT-analytic` and `Water-Full` fields remain inside original result files
for provenance and must not be substituted for the paper's rows.

Black-box baseline JSON files for LLaMA are the corrected run. The file
`results/blackbox/conditional_baselines/summary-uncorrected-llama.csv` is retained only
as historical provenance and must not be used for the final table. Generate the final
320-row Tables 1–4 per-cell AUROC table with `python scripts/build_paper_tables.py`.
Generate the four Table 8 macro rows from the 20 per-cell comparisons in each split
with `python scripts/build_table_8.py`.

## Still needed for a full model-level rerun

- Local source and proxy model checkpoints. The source models are LLaMA-2-7B,
  OPT-2.7B, OPT-6.7B, and Falcon-7B; black-box proxies are GPT-J-6B and
  GPT-Neo-2.7B.
- Model-derived white-box exact moment caches and the black-box full-vocabulary
  feature caches/frozen fit bundles. They can be regenerated, but the black-box bundle
  in the original working tree is about 22 GB.

The Table 8 White/Independent and Fresh per-document output bundles are included;
they were recovered from the original project after the initial repository draft.

The selected text data, per-document final scores, and Figure 2–3 figure data are
present. The separate original upstream corpus pools are needed only to repeat source
selection from scratch; the selected source documents are already included.
Only `figures/method_pipeline.png` is shipped as an image. Regenerable diagnostic
PNG/PDF files are intentionally omitted; their numeric CSV and score artifacts remain.
