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

Published figure files are protected by the figure SHA-256 manifest. The
verification command also redraws every PNG and checks that all six render,
remain non-blank, and retain the published dimensions within a 2% portability
tolerance. Pixel hashes are not compared because font rasterisation varies by
operating system.

已发布图文件由图像 SHA-256 清单保护。验证命令还会重新绘制全部 PNG，确认六图均能
生成、内容非空，且尺寸相对发布图的偏差不超过 2%。不同操作系统的字体栅格化结果
不同，因此不再比较像素哈希。

## Tables / 表

| Table | Source of truth | Evidence | Check |
|---|---|---|---|
| 表1 数据集与测试协议 | recomputed from runs / 由运行结果重算 | `evidence/*/per_seed/**/metrics.json` | `table1:protocol` (20 values) |
| 表2 七种不平衡配置 | runner's `VARIANT_ORDER` | `server_snapshot/v4/run_v3_explainable.py` | `table2:configurations` |
| 表3 强基线对比 | aggregated results | `evidence/v3/aggregate/baseline_summary.csv` | `tables:results` (mean and SD) |
| 表4 主要结果 | aggregated results | `evidence/v3/aggregate/performance_summary.csv` | `tables:results` (mean and SD) |
| 表5 配对差值与区间 | paired bootstrap | `evidence/v3/aggregate/performance_summary.csv`, `evidence/v4/aggregate/ddos2019_summary.json` | `tables:results` |
| 表6 多层感知机对照 | MLP control runs | `evidence/v4/aggregate/v4_performance_summary.csv`, `evidence/v4/per_seed/**` | `tables:results` |
| 表7 关键少数类检测变化 | per-class results | `evidence/v3/aggregate/per_class_summary.csv`, `evidence/v3/per_seed/**` | `tables:results` |
| 表8 TreeSHAP 稳定性 | explanation aggregates | `evidence/v3/aggregate/explanation_summary.csv` | `tables:results` |

`tables:results` compares every published value, including means, sample
standard deviations, paired differences, confidence bounds, support counts and
TreeSHAP additivity errors. It checks 272 result cells; it is not a row-count or
presence check.

`tables:results` 会逐值比较均值、样本标准差、配对差值、置信区间边界、测试支持数和
TreeSHAP 加和误差，共检查 272 个结果单元；它不是行数或存在性检查。

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

## Which script produced which evidence / 证据由哪个脚本产生

| Evidence | Produced by | Rerun command |
|---|---|---|
| `evidence/v3/per_seed/**` | `run_v3_explainable.py` | `docs/RUNNING.md` §1 |
| `evidence/v3/aggregate/*` | `aggregate_v3_results.py` | `docs/RUNNING.md` §1 |
| `evidence/ddos/per_seed/**` | `prepare_cicddos2019_cache.py` → `run_v3_explainable.py` | `docs/RUNNING.md` §2 |
| `evidence/v4/aggregate/ddos2019_summary.json` | `analyze_ddos2019.py` | `docs/RUNNING.md` §2 |
| `evidence/v4/aggregate/ddos2019_evidence.json` | `collect_ddos2019_evidence.py` | `docs/RUNNING.md` §2 |
| `evidence/v4/aggregate/sampling_diag_*.json` | `diagnose_ddos_sampling.py` | `docs/RUNNING.md` §2 |
| `evidence/v4/per_seed/**` | `run_v4_deep_baseline.py` | `docs/RUNNING.md` §3 |
| `evidence/v4/aggregate/v4_summary.json` | `aggregate_v4_results.py` | `docs/RUNNING.md` §3 |
| `evidence/v4/aggregate/weight_mass_summary.json` | `extract_weight_mass.py` | `docs/RUNNING.md` §4 |
| `evidence/v4/aggregate/crosscheck_layers.json` | `crosscheck_layers.py` | `docs/RUNNING.md` §4 |
| `figures/*.png` | the three figure scripts | `docs/RUNNING.md` §5 |

`python -m rwxai verify --all` rebuilds four V3 CSVs, the V4 summary and the
CIC-DDoS2019 summary from the 49 public per-seed metric files before checking
the manuscript claim snapshot and redrawing the figures.

`python -m rwxai verify --all` 会先从 49 个公开逐种子指标文件重建 4 个 V3 CSV、
V4 汇总和 CIC-DDoS2019 汇总，再核对论文表格快照并重新绘图。
