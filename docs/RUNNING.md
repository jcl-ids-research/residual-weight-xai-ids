# Running the experiments

`README.md` covers offline verification, which needs no data. This document
covers the other direction: rerunning the experiments that produced the
evidence.

Every command below is the one that actually ran on the experiment server. The
paths in the examples are the server's; substitute your own. Argument defaults
come from the scripts themselves, so a command with no explicit path falls back
to the server layout and will fail elsewhere — pass the paths.

## Before you start

Obtain the four datasets (see `DATASETS.md`), then check your setup against the
recorded run environment:

```
Linux 5.15 · Python 3.10.12 · numpy 1.26.4 · pandas 2.3.3
scikit-learn 1.7.2 · imbalanced-learn 0.14.2 · XGBoost 3.2.0
CUDA · 24 CPU threads
```

Different library versions will produce different numbers. `pyproject.toml`
pins them for this reason.

Every script takes `--smoke` to run on a tiny subsample. Use it first: a smoke
run finishes in minutes and catches a wrong path before you spend hours.

## 1. Main experiment — the first three datasets

Produces `evidence/v3/per_seed/` and the aggregates behind Tables 3-5, 7, 8 and
Figures 2-5.

One run is one dataset and one seed:

```bash
python src/rwxai/run_v3_explainable.py \
    --dataset unsw \
    --seed 42 \
    --out results \
    --unsw-dir /opt/UNSW-NB15 \
    --device cuda \
    --n-jobs 24 \
    --shap-samples 1000
```

`--dataset` accepts `unsw`, `nslkdd`, `cicids2017` and `cicddos2019`. Use
`--nslkdd-dir`, `--cicids2017-dir` or `--cicddos2019-dir` for the others.

The reported matrix is 23 runs — ten seeds each for UNSW-NB15 and NSL-KDD,
three for CIC-IDS-2017:

```bash
for seed in 42 123 456 789 1001 2024 31415 65537 77777 99991; do
    python src/rwxai/run_v3_explainable.py --dataset unsw   --seed "$seed" --out results --device cuda
    python src/rwxai/run_v3_explainable.py --dataset nslkdd --seed "$seed" --out results --device cuda
done
for seed in 42 123 456; do
    python src/rwxai/run_v3_explainable.py --dataset cicids2017 --seed "$seed" --out results --device cuda
done
```

The server ran two jobs at a time and skipped any run whose `metrics.json`
already existed, which makes the queue resumable. `server_snapshot/v3/run_v3_queue.sh`
is that queue, kept verbatim.

Then aggregate:

```bash
python src/rwxai/aggregate_v3_results.py --root results --output aggregate
```

## 2. CIC-DDoS2019

The raw release is roughly 70 million flows, so a cache is built once and
reused by all three seeds instead of re-parsing 30 GB per run:

```bash
python src/rwxai/prepare_cicddos2019_cache.py \
    --source /opt/CIC-DDoS2019 \
    --out cache/cicddos2019.npz \
    --train-rows 250000 \
    --test-rows 250000 \
    --workers 16
```

This writes `cicddos2019.npz` plus `cicddos2019.audit.json`, which records the
per-file sampling rates and class proportions. The published copy of that audit
is in `evidence/v4/aggregate/`.

Then run the three seeds against the cache and analyse them:

```bash
for seed in 42 123 456; do
    python src/rwxai/run_v3_explainable.py \
        --dataset cicddos2019 \
        --seed "$seed" \
        --cicddos2019-dir cache \
        --out ddos_results \
        --device cuda
done

python src/rwxai/analyze_ddos2019.py --results ddos_results --out ddos2019_summary.json
```

Two supporting scripts document the sampling itself:

```bash
python src/rwxai/diagnose_ddos_sampling.py --day /opt/CIC-DDoS2019/CSV-03-11 \
    --target-rows 250000 --out sampling_diag_train.json

python src/rwxai/collect_ddos2019_evidence.py \
    --source /opt/CIC-DDoS2019 --results ddos_results \
    --cache cache/cicddos2019.npz --out ddos2019_evidence.json
```

`collect_ddos2019_evidence.py` samples raw `Timestamp` values from each CSV.
That is how the capture dates were established rather than assumed — see the
direction caveat in `DATASETS.md`.

## 3. Multilayer perceptron control

Tests whether the residual-weight effect is specific to XGBoost. Same protocol,
same seeds, different model — 23 runs producing `evidence/v4/per_seed/` and
Table 6:

```bash
for seed in 42 123 456 789 1001 2024 31415 65537 77777 99991; do
    python src/rwxai/run_v4_deep_baseline.py --dataset unsw   --seed "$seed" --out results --device cuda --n-jobs 12
    python src/rwxai/run_v4_deep_baseline.py --dataset nslkdd --seed "$seed" --out results --device cuda --n-jobs 12
done
for seed in 42 123 456; do
    python src/rwxai/run_v4_deep_baseline.py --dataset cicids2017 --seed "$seed" --out results --device cuda --n-jobs 12
done

python src/rwxai/aggregate_v4_results.py --results results --out v4_summary.json
```

`server_snapshot/v4/run_v4_queue.sh` is the queue that ran this.

## 4. Weight mass and cross-layer accounting

Figure 6 needs the per-class total weight before and after cleaning:

```bash
python src/rwxai/extract_weight_mass.py --results results --out weight_mass_summary.json
```

The cross-layer check reconciles raw files, cache, runs and the manuscript:

```bash
python src/rwxai/crosscheck_layers.py --mode manuscript \
    --facts crosscheck_layers.json \
    --docx <manuscript.docx> \
    --out crosscheck_layers.json
```

## 5. Figures

```bash
python src/rwxai/make_v4_figures.py \
    --tables evidence/v3/aggregate \
    --ddos evidence/v4/aggregate/ddos2019_summary.json \
    --runs evidence/v3/per_seed \
    --out figures

python src/rwxai/make_workflow_figure.py --out figures

python src/rwxai/make_weight_mass_figure.py \
    --mass evidence/v4/aggregate/weight_mass_summary.json \
    --ddos evidence/v4/aggregate/ddos2019_summary.json \
    --out figures
```

These three run against the committed evidence, so they work without the
datasets. `python -m rwxai verify --all` runs exactly these commands into a
scratch directory and compares the PNG output byte for byte.

## Order

```
datasets ─┬─> run_v3_explainable ────> aggregate_v3_results ─┐
          │                                                  │
          ├─> prepare_cicddos2019_cache ─> run_v3_explainable │
          │        └─> analyze_ddos2019 ─────────────────────┤──> figures
          │                                                  │
          └─> run_v4_deep_baseline ────> aggregate_v4_results ┘
                                          extract_weight_mass
```

## Cost

The full matrix is 49 runs. On the server — one A100, 24 threads, two
concurrent jobs — the main experiment took hours, not minutes, and building the
CIC-DDoS2019 cache is itself a long single-pass scan over about 30 GB.

If you only want to check the paper's numbers, do not run any of this. Use
`python -m rwxai verify --all`; it finishes in under a minute.
