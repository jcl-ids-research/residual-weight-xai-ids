# Running the experiments / 实验运行手册

`README.md` covers offline verification, which needs no data. This document
covers the other direction: rerunning the experiments that produced the
evidence.

`README.md` 讲的是离线核验，不需要数据。本文讲另一件事：重跑产生这些证据的实验。

Every command below is the one that actually ran on the experiment server. The
paths in the examples are the server's; substitute your own. Argument defaults
come from the scripts themselves, so a command with no explicit path falls back
to the server layout and will fail elsewhere — pass the paths.

以下命令都是服务器上真实执行过的。示例中的路径是服务器路径，请替换为自己的。
参数默认值取自脚本本身，因此不显式传路径就会回退到服务器目录结构、在别处必然失败
——请把路径写全。

## Before you start / 开始之前

Obtain the four datasets (see `DATASETS.md`), then check your setup against the
recorded run environment:

先获取四个数据集（见 `DATASETS.md`），再核对运行环境：

```
Linux 5.15 · Python 3.10.12 · numpy 1.26.4 · pandas 2.3.3
scikit-learn 1.7.2 · imbalanced-learn 0.14.2 · XGBoost 3.2.0
CUDA · 24 CPU threads
```

Different library versions will produce different numbers. `pyproject.toml`
pins them for this reason.

库版本不同会得到不同的数值。`pyproject.toml` 因此锁定了版本。

Every script takes `--smoke` to run on a tiny subsample. Use it first: a smoke
run finishes in minutes and catches a wrong path before you spend hours.

每个脚本都支持 `--smoke`，在极小子样本上运行。**建议先跑它**：几分钟即可完成，
能在你耗费数小时之前发现路径写错。

Install verification and test dependencies with:

安装验证与测试依赖：

```bash
pip install -e ".[dev]"
```

For full model training, also install `.[training]`. Select the appropriate
CUDA-enabled PyTorch wheel for the target machine when needed.

完整模型训练还需安装 `.[training]`；如使用 CUDA，请按目标机器选择对应的 PyTorch
CUDA 安装包。

## Offline verification / 离线核验

Rebuild all six public aggregates, validate every table value and redraw all
six figures:

重建 6 个公开聚合文件、逐值核对全部表格并重绘 6 幅图：

```bash
python -m rwxai verify --all --report verification_report.json
```

If the submitted DOCX is available, bind the verification to that exact file:

如持有投稿 DOCX，可将核验绑定到该文件的精确版本：

```bash
python -m rwxai verify --all --manuscript /path/to/manuscript.docx
```

To rebuild only the V3 public CSV aggregates without unpublished prediction or
SHAP arrays:

如只需在没有预测和 SHAP 数组的条件下重建 V3 公开 CSV：

```bash
python src/rwxai/aggregate_v3_results.py \
    --root evidence/v3/per_seed \
    --output rebuilt-v3 \
    --metrics-only
```

## 1. Main experiment — the first three datasets / 主实验：前三个数据集

Produces `evidence/v3/per_seed/` and the aggregates behind Tables 3-5, 7, 8 and
Figures 2-5.

产生 `evidence/v3/per_seed/` 及其聚合结果，对应表3-5、表7、表8和图2-5。

One run is one dataset and one seed / 一次运行对应一个数据集和一个种子：

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

`--dataset` 可取 `unsw`、`nslkdd`、`cicids2017`、`cicddos2019`；其余数据集分别用
`--nslkdd-dir`、`--cicids2017-dir`、`--cicddos2019-dir` 指定目录。

The reported matrix is 23 runs — ten seeds each for UNSW-NB15 and NSL-KDD,
three for CIC-IDS-2017:

论文报告的是 23 次运行——UNSW-NB15 和 NSL-KDD 各十个种子，CIC-IDS-2017 三个种子：

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

服务器每次并发两个任务，并跳过 `metrics.json` 已存在的运行，因此队列可断点续跑。
`server_snapshot/v3/run_v3_queue.sh` 就是那份队列脚本，原样保留。

Then aggregate / 然后聚合：

```bash
python src/rwxai/aggregate_v3_results.py --root results --output aggregate
```

## 2. CIC-DDoS2019

The raw release is roughly 70 million flows, so a cache is built once and
reused by all three seeds instead of re-parsing 30 GB per run:

原始数据约 7 000 万条流量，因此先构建一次缓存供三个种子复用，避免每次运行都重新
解析 30 GB：

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

这会生成 `cicddos2019.npz` 和 `cicddos2019.audit.json`，后者记录逐文件抽样率与
类别占比。该审计文件的公开副本在 `evidence/v4/aggregate/`。

Then run the three seeds against the cache and analyse them:

然后基于缓存运行三个种子并分析：

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

另有两个脚本用于记录抽样本身：

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

`collect_ddos2019_evidence.py` 从每个 CSV 抽取原始 `Timestamp` 值。采集日期是**这样
查证出来的，而不是假定的**——方向问题的说明见 `DATASETS.md`。

## 3. Multilayer perceptron control / 多层感知机对照

Tests whether the residual-weight effect is specific to XGBoost. Same protocol,
same seeds, different model — 23 runs producing `evidence/v4/per_seed/` and
Table 6:

用于检验残差权重效应是否只在 XGBoost 上成立。协议相同、种子相同、模型不同，
共 23 次运行，产生 `evidence/v4/per_seed/` 和表6：

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

`server_snapshot/v4/run_v4_queue.sh` 是执行这批任务的队列脚本。

## 4. Weight mass and cross-layer accounting / 权重质量与跨层核对

Figure 6 needs the per-class total weight before and after cleaning:

图6 需要清理前后的逐类总权重：

```bash
python src/rwxai/extract_weight_mass.py --results results --out weight_mass_summary.json
```

The cross-layer check reconciles raw files, cache, runs and the manuscript:

跨层核对用于对齐原始文件、缓存、运行结果与论文正文：

```bash
python src/rwxai/crosscheck_layers.py --mode manuscript \
    --facts crosscheck_layers.json \
    --docx <manuscript.docx> \
    --out crosscheck_layers.json
```

## 5. Figures / 绘图

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
datasets. `python -m rwxai verify --all` runs them in a scratch directory and
checks that each PNG is non-blank and retains the published structure. The
published files themselves remain protected by their SHA-256 manifest.

这三条命令基于已提交的证据运行，**不需要数据集**。`python -m rwxai verify --all`
会在临时目录执行这些命令，确认每幅 PNG 非空且保持发布版结构；已发布文件本身仍由
SHA-256 清单严格保护。

## Order / 执行顺序

```
datasets ─┬─> run_v3_explainable ────> aggregate_v3_results ─┐
          │                                                  │
          ├─> prepare_cicddos2019_cache ─> run_v3_explainable │
          │        └─> analyze_ddos2019 ─────────────────────┤──> figures
          │                                                  │
          └─> run_v4_deep_baseline ────> aggregate_v4_results ┘
                                          extract_weight_mass
```

## Cost / 代价

The full matrix is 49 runs. On the server — one A100, 24 threads, two
concurrent jobs — the main experiment took hours, not minutes, and building the
CIC-DDoS2019 cache is itself a long single-pass scan over about 30 GB.

全套共 49 次运行。在服务器上（单张 A100、24 线程、双并发），主实验以**小时**计而非
分钟；构建 CIC-DDoS2019 缓存本身就是对约 30 GB 数据的一次长时间单遍扫描。

If you only want to check the paper's numbers, do not run any of this. Use
`python -m rwxai verify --all`; it finishes in under a minute.

**如果只是想核对论文数值，不要执行以上任何命令**，用 `python -m rwxai verify --all`，
一分钟内即可完成。
