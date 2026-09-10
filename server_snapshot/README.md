# Server code snapshot (read-only provenance) / 服务器代码快照（只读溯源）

These files are byte-for-byte copies of the code that ran on the experiment
server. They are kept unmodified so a reviewer can confirm that the published
numbers were produced by this code and not by something rewritten afterwards.
Every file's SHA-256 is recorded in `MANIFEST.sha256` and was verified against
`sha256sum` output taken on the server itself.

以下文件是服务器上实际运行代码的**逐字节副本**，保持未修改状态，以便审阅者确认
公开数值确由该代码产生，而非事后改写的版本。每个文件的 SHA-256 记录在
`MANIFEST.sha256`，并已与服务器上 `sha256sum` 的输出逐一核对。

**Do not edit anything in this directory.** The maintained, runnable code lives
in `src/rwxai/`.

**请勿修改本目录下任何文件。** 可运行、可维护的代码在 `src/rwxai/`。

## `v3/` — original run for UNSW-NB15, NSL-KDD and CIC-IDS-2017 / 前三个数据集的原始运行

Source: `/opt/ids_revision/v3_strengthened/scripts` (8 files).

来源：`/opt/ids_revision/v3_strengthened/scripts`（8 个文件）。

This is the code that produced the per-seed metrics for the first three
datasets, published under `evidence/v3/per_seed/`.

这是产生前三个数据集逐种子指标的代码，其结果公开在 `evidence/v3/per_seed/`。

## `v4/` — CIC-DDoS2019, the MLP control, and corrected aggregation / DDoS、感知机对照与修正后的聚合

Source: `/opt/ids_revision/v4_deep_baseline/scripts` (11 files).

来源：`/opt/ids_revision/v4_deep_baseline/scripts`（11 个文件）。

This is the code that produced `evidence/v4/per_seed/` (multilayer perceptron
control runs) and `evidence/ddos/per_seed/` (CIC-DDoS2019 runs).

这是产生 `evidence/v4/per_seed/`（多层感知机对照运行）和 `evidence/ddos/per_seed/`
（CIC-DDoS2019 运行）的代码。

## Why three files differ between `v3/` and `v4/` / 为什么有三个文件在两版之间不同

`v3_data.py` is identical in both directories. Three others are not, and the
differences are deliberate:

`v3_data.py` 在两个目录中完全相同。另外三个文件不同，且差异是**有意为之**：

| File | Difference / 差异 |
|---|---|
| `run_v3_explainable.py` | `v4/` adds the `cicddos2019` dataset and its `--cicddos2019-dir` argument, and records `official_test_set_untouched: false` for that dataset because its test partition is a fixed stratified sample rather than a full official split.<br>`v4/` 增加了 `cicddos2019` 数据集及 `--cicddos2019-dir` 参数，并对该数据集记录 `official_test_set_untouched: false`，因为其测试集是固定分层抽样而非完整官方划分。 |
| `aggregate_v3_results.py` | `v4/` computes Spearman correlations with **average ranks**, so tied importance values share their mean rank. The earlier sort-position ranking silently broke ties and did not match the Spearman definition.<br>`v4/` 用**平均秩**计算 Spearman 相关，使并列的重要度取其均值秩。此前按排序位置定秩会**静默打破并列**，不符合 Spearman 定义。 |
| `make_v3_figures.py` | Superseded by the figure scripts in `src/rwxai/`, which draw the six figures in the published manuscript.<br>已由 `src/rwxai/` 中的绘图脚本取代，后者绘制论文最终采用的六张图。 |

The manuscript's reported values were recomputed with the corrected
average-rank aggregation. The earlier version is retained here only so the
change is visible rather than hidden.

论文所报数值均已用修正后的平均秩聚合**重新计算**。此处保留旧版本，只是为了让这处
改动**可见**，而不是被掩盖。

## What is not here / 此处没有什么

Raw datasets, trained models, prediction arrays, SHAP arrays, caches and
checkpoints stay on the experiment server. This repository publishes code and
aggregate/per-seed statistics only.

原始数据集、训练好的模型、预测数组、SHAP 数组、缓存和检查点均保留在实验服务器。
本仓库只公开代码与聚合/逐种子统计量。
