# Paper data and cache preparation

This repository includes the exact selected text inputs for the paper's five datasets:
C4, PubMedQA, SQuAD, WritingPrompts, and XSum. The full upstream source pools and model
checkpoints are separate downloads. Run `python scripts/audit_paper_data.py` to validate
the checked-in files against [`paper_data_manifest.json`](../data/manifests/paper_data_manifest.json).

## Files actually read by the experiment code

| Role | Repository path | Contents |
|---|---|---|
| Development source selection | `data/train/sources/<dataset>.jsonl` | 500 paired sources and 500 independent calibration sources per dataset |
| Development model text | `data/train/text/<model>/<dataset>.jsonl` | 500 human–machine training pairs, with calibration text |
| Frozen test source selection | `data/test/sources/<dataset>.jsonl` | 500 sources excluded from development; adjacent `.audit.json` records selection checks |
| Independent test text, non-LLaMA | `data/test/text/<model>/<dataset>.jsonl` | 500 human–machine test pairs; adjacent `.metadata.json` records generation settings |
| Independent test text, corrected LLaMA | `data/test/text/llama-2-7b/<dataset>.jsonl` | Final LLaMA test pairs used by the paper |
| White-box frozen development fit | `data/fits/whitebox/<model>/<dataset>/results.json` | Train-only selected spline configuration and coefficients for locked evaluation |
| Black-box development adapters | `data/adapters/train/<model>/<dataset>.raw_data.json` | 500 train pairs and calibration texts in official baseline format |
| Black-box test adapters, non-LLaMA | `data/adapters/test/<model>/<dataset>.raw_data.json` | Independent test text in official baseline format |
| Black-box test adapters, corrected LLaMA | `data/adapters/test/llama-2-7b/<dataset>.raw_data.json` | Corrected LLaMA test text in official baseline format |
| Fresh confirmation sources | `data/fresh/sources/{test,calibration}/<dataset>.jsonl` | New held-out source selections, 500 test and 500 calibration passages per dataset |
| Fresh confirmation generated text | `data/fresh/text/<model>/<dataset>.jsonl` | 500 fresh human–machine pairs with independent calibration text per cell; adjacent metadata records generation |
| Fresh split locks | `data/fresh/source_split_lock.json`, `data/fresh/validation_lock.json` | Frozen fresh-confirmation selection and validation metadata |

The audit verifies 191 required files, all 20 model–dataset cells,
500 rows per split, disjoint development/test source indices, reused human calibration
passages after whitespace normalization, exact equality between each adapter and its
source JSONL, and the fresh text's source-record linkage.
The original development calibration text sometimes contains different whitespace;
the independent-test generator normalizes it before copying it. File hashes are
computed on the packaged files; the text fields themselves are unchanged.

Fresh confirmation has its own frozen source split and generated text. The included
`experiments/prepare_fresh_source_split.py`, `generate_fresh_confirmation.py`,
`validate_fresh_confirmation.py`, and `evaluate_*_ltheta_confirmatory.py` cover its
selection, generation, validation, and rescoring. We reran validation over all 20
fresh cells, including exact and 48-word-prefix overlap checks against earlier data.
All packaged metadata paths are repository-relative. The validator resolves the
corresponding files from `data/fresh/` and checks their hashes.
The fresh source-selection lock also records an excluded, pre-correction LLaMA test
set as `data/legacy/test-text/llama-2-7b/`; those historical files are intentionally
not shipped. The paper's corrected LLaMA passages are in `data/test/text/llama-2-7b/`.

The old top-level `generation_manifest.json` with a 64-token prompt and top-p 0.95 was
from an earlier experiment. Final development generation settings are recorded in the
four `data/train/text/<model>/generation_manifest.json` files; test settings are in the
per-cell metadata. The final protocol uses a 120-token prompt, 150–200 total tokens,
top-k 50, temperature 1, top-p 1, and at least 55 generated words. For some non-LLaMA
test metadata, `temperature` and `top_p` are stored as `null` because the generation
script did not pass explicit overrides; the model's effective defaults should be checked
when rerunning these cells. Corrected LLaMA metadata records both values explicitly.

## Cache and model inputs for new scoring runs

The text above is sufficient to audit splits and rebuild the model-dependent caches.
The paper's exact white-box cache runner is
`src/build_whitebox_reference_cache.py`. It reads `--generated-root` and writes
`--cache-root`; use `data/train/text` for train/calibration and `data/test/text` for
independent testing. The corrected LLaMA files are in the same test tree. Frozen fits
are already included.

The black-box runner `experiments/detect_tensor_rayleigh_blackbox.py` consumes the
`*.raw_data.json` adapters and local GPT-J-6B/GPT-Neo-2.7B checkpoints. The cache-first
factorial fit and final `T_theta` rescore are in
`experiments/run_blackbox_factorial_ablation.py` and
`experiments/run_blackbox_factorial_ltheta.py`. The latter needs the original large
black-box feature caches and frozen fit bundles, which are not in this Git repository.

Put local checkpoints under `models/` or pass their paths explicitly. The expected
source-model directories are `models/llama-2-7b`, `models/opt-2.7b`,
`models/opt-6.7b`, and `models/falcon-7b`; the proxy-model directories are
`models/gpt-j-6b` and `models/gpt-neo-2.7b`. Checkpoint licensing and access are
controlled by their upstream providers.

The official AdaDetectGPT baseline runner additionally expects its upstream code at
`third_party/AdaDetectGPT` or a path passed with `--official-repo-root`.

The original working tree included multi-gigabyte derived logits and feature caches. These are
omitted from the uploadable repository. Reproducing tables and figures from the
published per-document scores does not require them; recomputing all model logits and
the black-box `T_theta` rescore does. The outstanding large inputs are listed in
[repository-layout.md](repository-layout.md).

## Dataset identifiers

| Paper name | Code identifier |
|---|---|
| C4 | `c4` |
| PubMedQA | `pubmed` |
| SQuAD | `squad` |
| WritingPrompts | `writing` |
| XSum | `xsum` |

Historical code sometimes calls this field `domain`; public documentation uses
`dataset`.
