# Datasets

All four datasets are public. None is redistributed here; each must be obtained
from its publisher.

| Dataset | Publisher | Files this project reads |
|---|---|---|
| UNSW-NB15 | UNSW Canberra, ADFA | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| NSL-KDD | Canadian Institute for Cybersecurity (CIC) | `KDDTrain+.txt`, `KDDTest+.txt` |
| CIC-IDS-2017 | CIC | `MachineLearningCVE/*.csv` (Monday–Friday) |
| CIC-DDoS2019 | CIC | `CSV-03-11/*.csv`, `CSV-01-12/*.csv` |

## Partitions

**UNSW-NB15, NSL-KDD** — the publisher's official train/test split, unmodified.
Ten percent of the official training set is held out as an internal validation
partition; the official test set is never touched during preprocessing,
sampling, feature ranking, threshold selection or early stopping.

**CIC-IDS-2017** — cross-day: Monday–Thursday for training, Friday as the
held-out test day. The training period is capped at a fixed stratified sample
of 250,000 flows; the Friday test set is kept whole.

**CIC-DDoS2019** — cross-day: `CSV-03-11` (captured 2018-11-03) for training,
`CSV-01-12` (captured 2018-12-01) as the later held-out day. Both days are
sampled to 250,000 flows because the full release is roughly 70 million flows.

Two caveats are stated in the manuscript and repeated here:

1. Because the CIC-DDoS2019 test partition is a sample rather than a complete
   official split, the reported intervals do not include the uncertainty
   introduced by that sampling.
2. The capture dates were read from the raw `Timestamp` field of the CSV files.
   The directory named `CSV-03-11` contains 2018-11-03 traffic and `CSV-01-12`
   contains 2018-12-01 traffic, so the chronological order runs `CSV-03-11` →
   `CSV-01-12`. This is the **opposite** of the direction the publisher's
   naming suggests, so these results are not directly comparable to work that
   follows the publisher's stated protocol.

The date evidence — raw timestamp samples per file — is published in
`evidence/v4/aggregate/ddos2019_evidence.json`.

## Class imbalance direction

UNSW-NB15, NSL-KDD and CIC-IDS-2017 have a normal-traffic majority. In
CIC-DDoS2019 attack traffic is the majority class, inverting the imbalance
direction. The manuscript treats this as a boundary condition rather than a
uniform benchmark.

## Preprocessing

Splitting happens before any fitting. Scalers, encoders, resamplers and class
weights are all fit on the sub-training partition only. Identity-like columns
(flow identifiers, addresses, timestamps) are dropped; the dropped column list
for CIC-DDoS2019 is recorded in `evidence/v4/aggregate/cicddos2019.audit.json`.
