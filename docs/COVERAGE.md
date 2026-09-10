# Coverage matrix / 覆盖对照表

Every figure and table in the manuscript, the code that produces it, the
evidence it reads, and the check that verifies it.

论文中每一张图和表，对应的生成代码、依据的证据文件，以及验证它的检查项。

## Figures / 图

| Figure | Script | Evidence | Check |
|---|---|---|---|
| 图1 研究流程 | `src/rwxai/make_workflow_figure.py` | — (schematic / 示意图) | `figures:regenerated` |
| 图2 Macro-F1 对比 | `src/rwxai/make_v4_figures.py` | `evidence/v3/aggregate/performance_summary.csv`, `baseline_summary.csv` | `figures:regenerated` |
| 图3 残差权重消融 | `src/rwxai/make_v4_figures.py` | `evidence/v3/aggregate/performance_summary.csv`, `evidence/v4/aggregate/ddos2019_summary.json` | `figures:regenerated` |
| 图4 告警预算工作点 | `src/rwxai/make_v4_figures.py` | `evidence/v3/aggregate/performance_summary.csv` | `figures:regenerated` |
| 图5 TreeSHAP 解释稳定性 | `src/rwxai/make_v4_figures.py` | `evidence/v3/aggregate/explanation_summary.csv`, `evidence/v3/per_seed/**/metrics.json` | `figures:regenerated` |
| 图6 权重质量不变性 | `src/rwxai/make_weight_mass_figure.py` | `evidence/v4/aggregate/weight_mass_summary.json`, `ddos2019_summary.json` | `figures:regenerated` |

Figures are verified by redrawing them from the committed evidence and
comparing PNG bytes. PDF and SVG are not byte-compared because Matplotlib
embeds a creation timestamp in those formats.

图的验证方式是从证据重新绘制并比对 PNG 字节。PDF 与 SVG 不做字节比对，因为
Matplotlib 会在其中写入生成时间戳。

## Tables / 表

| Table | Source of truth | Evidence | Check |
|---|---|---|---|
| 表1 数据集与测试协议 | recomputed from runs / 由运行结果重算 | `evidence/*/per_seed/**/metrics.json` | `table1:protocol` (20 values) |
| 表2 七种不平衡配置 | runner's `VARIANT_ORDER` | `server_snapshot/v4/run_v3_explainable.py` | `table2:configurations` |
| 表3 强基线对比 | aggregated results | `evidence/v3/aggregate/baseline_summary.csv` | `tables:results` |
| 表4 主要结果 | aggregated results | `evidence/v3/aggregate/performance_summary.csv` | `tables:results` |
| 表5 配对差值与区间 | paired bootstrap | `evidence/v3/aggregate/performance_summary.csv`, `evidence/v4/aggregate/ddos2019_summary.json` | `tables:results` |
| 表6 多层感知机对照 | MLP control runs | `evidence/v4/aggregate/v4_performance_summary.csv`, `evidence/v4/per_seed/**` | `tables:results` |
| 表7 关键少数类检测变化 | per-class results | `evidence/v3/aggregate/per_class_summary.csv`, `evidence/v3/per_seed/**` | `tables:results` |
| 表8 TreeSHAP 稳定性 | explanation aggregates | `evidence/v3/aggregate/explanation_summary.csv` | `tables:results` |

Table 1 was previously unverified: nothing recomputed its record counts,
feature counts or seed counts from the runs. It is now checked value by value,
including the one-hot width, which legitimately varies across seeds
(UNSW-NB15: 192-194) because it depends on which categorical values appear in
the sub-training split.

表1此前没有任何数值核验。现在逐值重算，包括独热编码宽度——该宽度会随种子变化
（UNSW-NB15 为 192–194），因为它取决于子训练集中出现的类别取值。

## Per-seed evidence / 逐种子证据

| Layer | Datasets | Seeds | Files |
|---|---|---|---|
| `evidence/v3/per_seed` | unsw, nslkdd, cicids2017 | 10 / 10 / 3 | 23 |
| `evidence/v4/per_seed` | unsw, nslkdd, cicids2017 (MLP control) | 10 / 10 / 3 | 23 |
| `evidence/ddos/per_seed` | cicddos2019 | 3 | 3 |

Total 49 runs. `evidence:completeness` fails if any is missing.

共 49 次运行。缺少任何一次，`evidence:completeness` 即失败。
