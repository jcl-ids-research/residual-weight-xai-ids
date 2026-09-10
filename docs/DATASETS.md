# Datasets / 数据集

All four datasets are public. None is redistributed here; each must be obtained
from its publisher.

四个数据集均为公开数据集。本仓库不转发任何数据，请各自从发布方获取。

| Dataset | Publisher | Files this project reads |
|---|---|---|
| UNSW-NB15 | UNSW Canberra, ADFA | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| NSL-KDD | Canadian Institute for Cybersecurity (CIC) | `KDDTrain+.txt`, `KDDTest+.txt` |
| CIC-IDS-2017 | CIC | `MachineLearningCVE/*.csv` (Monday–Friday) |
| CIC-DDoS2019 | CIC | `CSV-03-11/*.csv`, `CSV-01-12/*.csv` |

## Partitions / 数据划分

**UNSW-NB15, NSL-KDD** — the publisher's official train/test split, unmodified.
Ten percent of the official training set is held out as an internal validation
partition; the official test set is never touched during preprocessing,
sampling, feature ranking, threshold selection or early stopping.

**UNSW-NB15、NSL-KDD** —— 使用发布方的官方训练/测试划分，未作改动。官方训练集中
留出 10% 作为内部验证集；预处理、采样、特征排序、阈值选择和早停**均不接触**官方
测试集。

**CIC-IDS-2017** — cross-day: Monday–Thursday for training, Friday as the
held-out test day. The training period is capped at a fixed stratified sample
of 250,000 flows; the Friday test set is kept whole.

**CIC-IDS-2017** —— 跨日划分：周一至周四训练，周五作为留出测试日。训练期固定分层
抽样 250 000 条；周五测试集**完整保留**。

**CIC-DDoS2019** — cross-day: `CSV-03-11` (captured 2018-11-03) for training,
`CSV-01-12` (captured 2018-12-01) as the later held-out day. Both days are
sampled to 250,000 flows because the full release is roughly 70 million flows.

**CIC-DDoS2019** —— 跨日划分：`CSV-03-11`（采集于 2018-11-03）训练，`CSV-01-12`
（采集于 2018-12-01）作为较晚的留出日。因完整数据约 7 000 万条，两天**均**抽样至
250 000 条。

Two caveats are stated in the manuscript and repeated here:

论文中已说明、此处重申两点：

1. Because the CIC-DDoS2019 test partition is a sample rather than a complete
   official split, the reported intervals do not include the uncertainty
   introduced by that sampling.

   由于 CIC-DDoS2019 测试集是抽样而非完整官方划分，所报区间**不包含**抽样引入的
   不确定性。

2. The capture dates were read from the raw `Timestamp` field of the CSV files.
   The directory named `CSV-03-11` contains 2018-11-03 traffic and `CSV-01-12`
   contains 2018-12-01 traffic, so the chronological order runs `CSV-03-11` →
   `CSV-01-12`. This is the **opposite** of the direction the publisher's
   naming suggests, so these results are not directly comparable to work that
   follows the publisher's stated protocol.

   采集日期读自 CSV 的原始 `Timestamp` 字段。目录 `CSV-03-11` 实为 2018-11-03 的
   流量，`CSV-01-12` 实为 2018-12-01，故时间顺序是 `CSV-03-11` → `CSV-01-12`。
   这与发布方命名所暗示的方向**相反**，因此本文结果不能与遵循发布方既定协议的
   工作直接比较。

The date evidence — raw timestamp samples per file — is published in
`evidence/v4/aggregate/ddos2019_evidence.json`.

日期证据（逐文件的原始时间戳样本）见
`evidence/v4/aggregate/ddos2019_evidence.json`。

## Class imbalance direction / 不平衡方向

UNSW-NB15, NSL-KDD and CIC-IDS-2017 have a normal-traffic majority. In
CIC-DDoS2019 attack traffic is the majority class, inverting the imbalance
direction. The manuscript treats this as a boundary condition rather than a
uniform benchmark.

UNSW-NB15、NSL-KDD 和 CIC-IDS-2017 以正常流量为多数类；CIC-DDoS2019 中攻击流量
才是多数类，**不平衡方向相反**。论文将其作为适用边界而非同质基准来处理。

## Preprocessing / 预处理

Splitting happens before any fitting. Scalers, encoders, resamplers and class
weights are all fit on the sub-training partition only. Identity-like columns
(flow identifiers, addresses, timestamps) are dropped; the dropped column list
for CIC-DDoS2019 is recorded in `evidence/v4/aggregate/cicddos2019.audit.json`.

先划分，后拟合。标准化器、编码器、重采样器和类别权重**一律只在子训练集上拟合**。
标识类字段（流标识、地址、时间戳）予以剔除；CIC-DDoS2019 的剔除字段清单记录在
`evidence/v4/aggregate/cicddos2019.audit.json`。

## Reproduction boundary / 复现边界

The repository ships 49 per-seed metric files, aggregate summaries, figures and
the exact server code snapshot. These are sufficient for offline aggregation,
table checking and figure regeneration. Raw traffic, trained models,
predictions, SHAP arrays and caches are intentionally not redistributed.

仓库提供 49 个逐种子指标文件、聚合摘要、图件和服务器代码快照，足以离线重算聚合、
核对表格并重新绘图。原始流量、训练模型、预测结果、SHAP 数组和缓存不予转发。

Full retraining requires obtaining all four datasets from their publishers and
installing the training extra with `pip install -e ".[training]"`. A CUDA build
of PyTorch may require the platform-specific installation command from the
official PyTorch documentation.

完整重训练需从发布方取得四个数据集，并执行 `pip install -e ".[training]"` 安装训练
依赖。CUDA 版 PyTorch 可能还需按 PyTorch 官方文档使用对应平台的安装命令。
