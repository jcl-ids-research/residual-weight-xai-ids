# residual-weight-xai-ids

Verification package for the manuscript *基于类别不平衡优化的可解释网络入侵检测方法研究*
(Explainable network intrusion detection with class-imbalance optimization:
residual weighting and its applicability boundaries).

The paper argues that stacking SMOTE(-NC), Tomek cleaning and the original
class weights double-compensates the minority classes, and that recomputing the
weights from the **cleaned** training distribution corrects it. This repository
lets a reader check that claim against the actual run evidence.

本文提出：SMOTE(-NC)、Tomek 清理与原始类别权重叠加使用，会对少数类造成**重复
补偿**；改用**清理后**训练分布重新计算权重可校正这一问题。本仓库让读者能对照真实
运行证据核查这一结论。

## What you can check in one minute / 一分钟内可以核查什么

```bash
pip install -e ".[dev]"
python -m rwxai verify --all --report verification_report.json
```

No dataset, no GPU, no network. The command checks:

不需要数据集、不需要 GPU、不需要联网。该命令检查：

| Check | What it proves |
|---|---|
| `manifest:evidence` | Every published result file still hashes to its recorded SHA-256 |
| `manifest:figures` | The published figures have not been edited |
| `manifest:server_snapshot` | The archived server code is unmodified |
| `evidence:completeness` | All 49 per-seed runs are present, none quietly dropped |
| `table1:protocol` | Table 1's record counts, feature counts and seed counts are recomputed from the runs |
| `table2:configurations` | Every Table 2 field matches the runner semantics |
| `aggregates:recomputed` | Six aggregate files are rebuilt from the 49 public per-seed metrics |
| `tables:results` | All 272 result cells in Tables 3-8 match the rebuilt evidence |
| `figures:regenerated` | All six figures redraw as non-blank images with the published structure |

Exit code is `0` only if every check passes.

只有全部检查通过，退出码才为 `0`。

If you also have the submitted DOCX, bind the check to that exact version:

如同时持有投稿 DOCX，可将核验绑定到该文件的精确版本：

```bash
python -m rwxai verify --all --manuscript /path/to/manuscript.docx
```

### The checks can fail / 这些检查是会失败的

A verifier that has never failed proves nothing. `tests/test_calibration.py`
corrupts one input at a time — a metrics value, a deleted seed, a table cell, a
figure byte, a snapshot file — and requires the matching check to fail and to
name what broke.

**从未失败过的检查器什么也证明不了。** `tests/test_calibration.py` 每次篡改一处
输入——某个指标值、删掉一个种子、改一个表格单元、翻转图片一个字节、动一个快照
文件——并要求对应的检查**必须失败**且指出坏在哪里。

```bash
python -m pytest tests/ -q
```

## Verify vs. rerun — an honest distinction / 核验与重跑：必须分清

Two different things, and this repository is explicit about which is which:

这是两件不同的事，本仓库明确区分：

**Verification (what this repository supports).** Everything above runs offline
from the committed evidence. It rebuilds the public aggregates, checks every
result-table cell and validates that all figure scripts render successfully.

**核验（本仓库支持）。** 以上全部基于已提交证据离线运行：重建公开聚合、逐值核对
结果表，并确认全部绘图脚本能够成功生成有效图像。

**Full rerun (what it does not ship).** Retraining from raw traffic needs the
four public datasets, a CUDA machine and many hours. The code that does it is
here, and `docs/RUNNING.md` gives the exact command for every stage — the same
commands that ran on the experiment server — but the datasets, model weights,
prediction arrays, SHAP arrays and caches are not.
See `docs/DATASETS.md` for how to obtain the data.

**完整重跑（本仓库不提供数据）。** 从原始流量重新训练需要四个公开数据集、一台
CUDA 机器和数小时时间。执行代码在此，`docs/RUNNING.md` 给出每个阶段的**确切命令**
（即服务器上真实执行过的命令），但数据集、模型权重、预测数组、SHAP 数组和缓存
不在其中。数据获取方式见 `docs/DATASETS.md`。

## Layout

```text
src/rwxai/          runnable code: experiment scripts, figure scripts, verification
server_snapshot/    read-only copies of the code that ran on the server
evidence/           aggregates and 49 per-seed metrics files
figures/            the six published figures (PNG / PDF / SVG)
paper_claims.json   table values extracted from the manuscript, with its SHA-256
tools/              helper that regenerates paper_claims.json
tests/              unit tests and the tamper-detection controls
docs/               dataset sources, provenance, coverage matrix
```

`docs/COVERAGE.md` maps every figure and table to the script and evidence file
behind it. `docs/RUNNING.md` is the command reference for rerunning the
experiments themselves.

`docs/COVERAGE.md` 将每张图、每张表映射到产生它的脚本和证据文件；
`docs/RUNNING.md` 是重跑实验本身的命令手册。

## Provenance / 代码溯源

`server_snapshot/` holds byte-exact copies of the experiment code, with
per-file SHA-256 values verified against the server. `v3/` produced the results
for UNSW-NB15, NSL-KDD and CIC-IDS-2017; `v4/` added CIC-DDoS2019, the
multilayer-perceptron control, and a correction to the Spearman computation
(tied importances now receive average ranks). The differences are listed in
`server_snapshot/README.md` rather than smoothed over.

`server_snapshot/` 保存实验代码的逐字节副本，每个文件的 SHA-256 均已与服务器核对。
`v3/` 产生了 UNSW-NB15、NSL-KDD 和 CIC-IDS-2017 的结果；`v4/` 增加了 CIC-DDoS2019、
多层感知机对照，并修正了 Spearman 的计算（并列重要度现取平均秩）。这些差异在
`server_snapshot/README.md` 中**逐条列出，而非抹平**。

## Environment / 运行环境

Results were produced on Linux 5.15, Python 3.10.12, scikit-learn 1.7.2,
imbalanced-learn 0.14.2, XGBoost 3.2.0, CUDA, 24 CPU threads. Core numerical
versions are pinned; figure verification tolerates platform font rasterisation.

结果产生于 Linux 5.15、Python 3.10.12、scikit-learn 1.7.2、imbalanced-learn 0.14.2、
XGBoost 3.2.0、CUDA、24 CPU 线程。核心数值依赖已锁定；图像核验允许不同平台的
字体栅格化差异。

## What is deliberately absent / 刻意不包含的内容

Raw datasets, trained models, prediction arrays, SHAP arrays, caches,
checkpoints, and the manuscript itself. The evidence published here is what the
tables and figures are computed from — roughly 7 MB, not 3 GB.

原始数据集、训练好的模型、预测数组、SHAP 数组、缓存、检查点，以及论文本身。
此处公开的证据即表格与图的计算来源——约 7 MB，而非 3 GB。

## License

MIT — see [LICENSE](LICENSE).
