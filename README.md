# Fake Account & Bot Network Detection via Graph Structure — demo

Working prototype of the REVA University Research Activity 2 proposal.

## Run
```
pip install networkx streamlit plotly matplotlib scikit-learn pandas numpy scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu
streamlit run app.py        # or double-click run_demo.bat
```
First load takes ~30 s (builds graph, computes features, trains 8 detectors). Everything after is cached.

## Files
- `botlab/core.py` – the whole pipeline (graph generator, 17 features, 8 detectors, robustness, ablation, metrics)
- `app.py` – Streamlit demo UI (10 tabs: 7 pipeline tabs + Why this matters, Text vs structure, Future work)
- Headless check: `python -W ignore -m botlab.core`

## Mapping to the proposal
| Proposal item | Where |
|---|---|
| Synthetic NetworkX graph, planted bot clusters, simulated timestamps | `build_graph` |
| Structural + temporal features (Sec. 6) | `compute_features` |
| LogReg, RF, spectral / community-outlier anomaly detection, GraphSAGE, GAT | `DetectorBank` |
| Separate content-based baseline | "Content baseline (text only)" |
| Robustness: camouflage, sparse groups, reduced sync, timing, AI text | `SCENARIOS`, tab 5 |
| Metrics: P, R, F1, PR-AUC, ROC-AUC, ΔF1, group-level, inference time | `evaluate`, `robustness`, `timing` |
| Feature ablation | `ablation`, tab 6 |

## Using real data (TwiBot-20/22, Cresci)
Convert to an edge list `source,target,timestamp` and labels, then build a `GraphData` from it
(replace `build_graph`; `compute_features` and `DetectorBank` are dataset-agnostic).

## Honest limitations
- Data is synthetic; numbers show the *method*, not real-world accuracy.
- GNNs are a dense-matrix implementation (fine up to ~2k nodes). Use PyTorch Geometric for TwiBot-22 scale.
- Group-level metrics are computed over all nodes (train nodes included); node metrics use the held-out test split only.
