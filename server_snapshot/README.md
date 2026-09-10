# Server code snapshot (read-only provenance)

These files are byte-for-byte copies of the code that ran on the experiment
server. They are kept unmodified so a reviewer can confirm that the published
numbers were produced by this code and not by something rewritten afterwards.
Every file's SHA-256 is recorded in `MANIFEST.sha256` and was verified against
`sha256sum` output taken on the server itself.

**Do not edit anything in this directory.** The maintained, runnable code lives
in `src/rwxai/`.

## `v3/` — original run for UNSW-NB15, NSL-KDD and CIC-IDS-2017

Source: `/opt/ids_revision/v3_strengthened/scripts` (8 files).

This is the code that produced the per-seed metrics for the first three
datasets, published under `evidence/v3/per_seed/`.

## `v4/` — CIC-DDoS2019, the MLP control, and corrected aggregation

Source: `/opt/ids_revision/v4_deep_baseline/scripts` (11 files).

This is the code that produced `evidence/v4/per_seed/` (multilayer perceptron
control runs) and `evidence/ddos/per_seed/` (CIC-DDoS2019 runs).

## Why three files differ between `v3/` and `v4/`

`v3_data.py` is identical in both directories. Three others are not, and the
differences are deliberate:

| File | Difference |
|---|---|
| `run_v3_explainable.py` | `v4/` adds the `cicddos2019` dataset and its `--cicddos2019-dir` argument, and records `official_test_set_untouched: false` for that dataset because its test partition is a fixed stratified sample rather than a full official split. |
| `aggregate_v3_results.py` | `v4/` computes Spearman correlations with **average ranks**, so tied importance values share their mean rank. The earlier sort-position ranking silently broke ties and did not match the Spearman definition. |
| `make_v3_figures.py` | Superseded by the figure scripts in `src/rwxai/`, which draw the six figures in the published manuscript. |

The manuscript's reported values were recomputed with the corrected
average-rank aggregation. The earlier version is retained here only so the
change is visible rather than hidden.

## What is not here

Raw datasets, trained models, prediction arrays, SHAP arrays, caches and
checkpoints stay on the experiment server. This repository publishes code and
aggregate/per-seed statistics only.
