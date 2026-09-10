# residual-weight-xai-ids

Verification package for the manuscript *基于类别不平衡优化的可解释网络入侵检测方法研究*
(Explainable network intrusion detection with class-imbalance optimization:
residual weighting and its applicability boundaries).

The paper argues that stacking SMOTE(-NC), Tomek cleaning and the original
class weights double-compensates the minority classes, and that recomputing the
weights from the **cleaned** training distribution corrects it. This repository
lets a reader check that claim against the actual run evidence.

## What you can check in one minute

```bash
pip install -e .
python -m rwxai verify --all
```

No dataset, no GPU, no network. The command checks:

| Check | What it proves |
|---|---|
| `manifest:evidence` | Every published result file still hashes to its recorded SHA-256 |
| `manifest:figures` | The published figures have not been edited |
| `manifest:server_snapshot` | The archived server code is unmodified |
| `evidence:completeness` | All 49 per-seed runs are present, none quietly dropped |
| `table1:protocol` | Table 1's record counts, feature counts and seed counts are recomputed from the runs |
| `table2:configurations` | Table 2's seven configurations are the ones the runner implements |
| `tables:results` | Tables 3-8 are present and complete in the claim snapshot |
| `figures:regenerated` | All six figures are redrawn and match the published PNGs **byte-for-byte** |

Exit code is `0` only if every check passes.

### The checks can fail

A verifier that has never failed proves nothing. `tests/test_calibration.py`
corrupts one input at a time — a metrics value, a deleted seed, a table cell, a
figure byte, a snapshot file — and requires the matching check to fail and to
name what broke.

```bash
python -m pytest tests/ -q
```

## Verify vs. rerun — an honest distinction

Two different things, and this repository is explicit about which is which:

**Verification (what this repository supports).** Everything above runs offline
from the committed evidence. It confirms the paper's numbers and figures follow
from the recorded runs.

**Full rerun (what it does not ship).** Retraining from raw traffic needs the
four public datasets, a CUDA machine and many hours. The code that does it is
here, and `docs/RUNNING.md` gives the exact command for every stage — the same
commands that ran on the experiment server — but the datasets, model weights,
prediction arrays, SHAP arrays and caches are not.
See `docs/DATASETS.md` for how to obtain the data.

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

## Provenance

`server_snapshot/` holds byte-exact copies of the experiment code, with
per-file SHA-256 values verified against the server. `v3/` produced the results
for UNSW-NB15, NSL-KDD and CIC-IDS-2017; `v4/` added CIC-DDoS2019, the
multilayer-perceptron control, and a correction to the Spearman computation
(tied importances now receive average ranks). The differences are listed in
`server_snapshot/README.md` rather than smoothed over.

## Environment

Results were produced on Linux 5.15, Python 3.10.12, scikit-learn 1.7.2,
imbalanced-learn 0.14.2, XGBoost 3.2.0, CUDA, 24 CPU threads. `pyproject.toml`
pins those library versions so verification recomputes the same values.

## What is deliberately absent

Raw datasets, trained models, prediction arrays, SHAP arrays, caches,
checkpoints, and the manuscript itself. The evidence published here is what the
tables and figures are computed from — roughly 7 MB, not 3 GB.

## License

MIT — see [LICENSE](LICENSE).
