# LLM-Assisted Explainable Log Anomaly Detection for Security Monitoring
*with cross-dataset evaluation and a synthetic attack-simulation test*

A college mini project (proof of concept, CPU only, built within ~3 hours).
Raw logs -> Drain templates -> event sequences -> anomaly models (Isolation Forest, DeepLog-style LSTM) ->
explanations (event-contribution ranking + local LLM) -> Streamlit dashboard.

> **Read the Limitations section.** The datasets are small samples and several results are weak or noisy.
> This README reports what the code actually produced, including the bad numbers.

## Architecture

```mermaid
flowchart LR
    A[Raw logs<br/>HDFS 100k lines / BGL 2k lines / SYNTHETIC auth.log] --> B[Drain parsing<br/>src/parse.py]
    B --> C[Templates + event IDs]
    C --> D[Sequences<br/>HDFS: per block<br/>BGL: sliding windows<br/>src/sequences.py]
    D --> E[Chronological split<br/>70% train / 30% test]
    C --> F[Template text embeddings<br/>MiniLM 384-d<br/>src/embed.py]
    E --> G1[Baselines on event IDs<br/>Isolation Forest + LSTM]
    E --> G2[Embedding models<br/>EmbIForest + EmbLSTM]
    F --> G2
    G2 --> H[Cross-dataset tests<br/>HDFS to BGL, BGL to HDFS]
    G2 --> I[Flagged sequences]
    I --> J[Leave-one-event-out contribution ranking<br/>src/explain.py]
    J --> K[Ollama llama3.2:1b explanation<br/>disk cache + fallback]
    I --> L[Severity from score percentiles]
    K --> M[Streamlit dashboard app.py]
    L --> M
    H --> M
    N[SYNTHETIC SSH auth.log] --> O[Same pipeline: src/attack_eval.py] --> M
```

## Setup (Windows, PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt          # pinned versions that were actually used
ollama pull llama3.2:1b                  # for LLM explanations (Ollama must be running)
```
Notes: `all-MiniLM-L6-v2` (~90 MB) downloads automatically the first time embeddings are computed
(code falls back to TF-IDF only if that fails; it did not fail here).
Data needed in `references/` (clone of loghub + loglizer; gitignored): `loglizer/data/HDFS/HDFS_100k.log_structured.csv`,
`loglizer/data/HDFS/anomaly_label.csv`, `loghub/BGL/BGL_2k.log`. Copy them into `data/raw/` (see `config.yaml`).

## How to run each step

| Step | Command | Output |
|---|---|---|
| A. parse + sequences | `python src/parse.py` then `python src/sequences.py` | `data/parsed/`, `data/sequences/` |
| B. baselines | `python src/evaluate.py` | `results/metrics.csv`, `confusion_baselines.png` |
| C. cross-dataset | `python src/evaluate.py cross` | `results/cross_dataset.csv/.png`, threshold sensitivity CSV |
| D. save models + dashboard | `python src/pipeline.py` then `streamlit run app.py` | `data/models/*.pkl`, dashboard on localhost:8501 |
| E. synthetic attack test | `python src/attack_eval.py` | `data/synthetic/SYNTHETIC_auth.log`, `results/attack_*` |
| F. collect tables | `python src/make_report.py` | `results/results_tables.md` |

All paths and hyperparameters are in `config.yaml`; seeds are fixed (42).

## Data actually used

| Dataset | Subset | Sequences | Train (normal/anomalous) | Test (normal/anomalous) |
|---|---|---|---|---|
| HDFS | first 104,815 lines (loglizer sample), 25 templates | 7,940 blocks | 5,366 / 192 | 2,261 / 121 |
| BGL | loghub 2,000-line sample, 105 templates | 199 windows (size 20, step 10) | 111 / 28 | 35 / 25 |

Split is chronological (first 70% train, last 30% test). Anomaly models train on **normal** training sequences only.
HDFS labels are per block; a BGL window is anomalous if any line in it is.

## Results

Full tables: `results/results_tables.md` (generated from the CSVs). Key results:

### In-dataset baselines, event-ID features
| Dataset | Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| HDFS | Isolation Forest | 0.196 | 0.182 | 0.189 | 0.568 |
| HDFS | LSTM (DeepLog) | 0.933 | 0.463 | 0.619 | 0.660 |
| BGL | Isolation Forest | 0.500 | 0.880 | 0.638 | 0.799 |
| BGL | LSTM (DeepLog) | 0.556 | 1.000 | 0.714 | 0.611 |

### Embedding models, in-dataset and cross-dataset (F1 / ROC-AUC)
| Model | HDFS->HDFS | BGL->BGL | HDFS->BGL | BGL->HDFS |
|---|---|---|---|---|
| EmbIForest | 0.000 / 0.795 | 0.269 / 0.342 | 0.560 / 0.656 | 0.614 / 0.636 |
| EmbLSTM | 0.586 / 0.659 | 0.138 / 0.585 | 0.238 / 0.461 | 0.282 / 0.358 |

![cross-dataset](results/cross_dataset.png)

**Threshold calibration.** The threshold for the target dataset is the 95th percentile of the source model's scores on the
target's *normal training* sequences (no anomaly labels). `results/cross_dataset_threshold_sensitivity.csv` shows F1 at the 80/90/95/99th
percentile; F1 changes a lot (e.g. HDFS->HDFS EmbIForest: 0.37 at the 90th percentile, 0.00 at the 95th), so single F1 numbers are fragile. ROC-AUC is threshold-free.

**Honest analysis.** Cross-dataset transfer is poor: several cells have ROC-AUC at or below 0.5. Likely reasons:
(1) the vocabularies differ (HDFS = block replication, BGL = kernel/hardware errors; nearest-template cosine similarity was only 0.30-0.43);
(2) the anomaly definitions differ (workflow deviation per block vs. any alert line in a window);
(3) the log styles differ (short regular sequences vs. 20 mixed lines);
(4) the BGL sample is tiny (60 test windows) and its train/test periods differ, so BGL numbers are noisy.
Embeddings did not improve in-dataset results on HDFS (EmbLSTM F1 0.586 vs. ID-based LSTM 0.619).
I make no claim that the embedding approach solves cross-dataset detection.

### Synthetic SSH brute-force test (**SYNTHETIC DATA, not a real attack**)
The log `data/synthetic/SYNTHETIC_auth.log` is generated by `src/attack_eval.py` (3,132 lines; 185 injected failed-password lines:
a 160-line fast burst and a 25-line slow drip; attacker IPs are RFC 5737 documentation addresses).
Test portion: 93 windows, 35 contain an injected attack, 58 are normal.

| Model | Attack windows flagged (of 35) | Fast burst | Slow drip | False alarms (of 58) | ROC-AUC |
|---|---|---|---|---|---|
| EmbLSTM trained on SSH-normal | 17 | 17/18 | 0/17 | 0 | 0.755 |
| EmbIForest trained on SSH-normal | 28 | 17/18 | 11/17 | 9 | 0.898 |
| EmbLSTM trained on HDFS | 0 | 0/18 | 0/17 | 1 | 0.340 |
| EmbIForest trained on HDFS | 7 | 1/18 | 6/17 | 0 | 0.800 |
| EmbIForest trained on BGL | 15 | 15/18 | 0/17 | 4 | 0.553 |
| EmbLSTM trained on BGL | 35 | 18/18 | 17/17 | 0 | 1.000 (**artifact, see below**) |

The BGL-trained EmbLSTM row looks perfect but is **not** evidence of detection: its predicted embeddings are almost identical for all inputs
(mean pairwise cosine 0.98) and almost orthogonal to the true next event (0.08), so its score is just distance to a near-constant vector, and the attack
template happens to be slightly farther from that vector. The HDFS-trained twin gets AUC 0.34 for the same reason in the opposite direction.
What the test supports: the pipeline runs end-to-end on a different log type, and models trained on that log's own normal traffic catch the fast burst.
Neither SSH-trained model reliably catches the slow drip. Single seed, tiny test set, synthetic data with only 10 templates: optimistic and not generalisable.

![attack scores](results/attack_scores.png)

> **Future work: real attack simulation.** Tools such as `hydra` must only be run against **your own isolated VMs** that you own and control,
> never against other people's systems or networks. I did not do this; the attack data in this project is entirely synthetic.

## Explainability and dashboard
* **Event-contribution ranking** (`src/explain.py`): remove each event from a flagged sequence, re-score, rank events by score drop.
  Works well when one event dominates (e.g. a Java `IOException` line in an HDFS block: drop 0.66 vs. about 0.02 for the others);
  it can miss anomalies caused by *missing* events.
* **LLM explanation:** `llama3.2:1b` via Ollama gets only the top contributing lines plus a short prompt. Log text is fenced in `<logs>` tags and the model
  is told to ignore instructions inside it (a mitigation, not a guarantee). Responses are cached in `.ollama_cache/`; if Ollama is down a fallback message is shown.
  A 1B model's explanations are often vague or partly wrong; treat them as a readability aid and verify against the log lines.
* **Severity:** percentile of a flagged score among all flagged scores (top 10% high, next 40% medium, rest low). It is relative, not an absolute risk level.
* **Dashboard** (`streamlit run app.py`): explorer (dataset/model/threshold slider, flagged table, detail view), model comparison page, and log-file upload.
  Tested headlessly (page rendering, no exceptions); the upload page's file-upload widget was not exercised in a browser.

## Limitations
* **Small samples:** HDFS = first ~105k lines (about 13 minutes of logs); BGL = only 2,000 lines (60 test windows). The 200k-line cap in the brief was not reached. Metrics, especially BGL, are noisy and not comparable to published full-dataset numbers.
* **CPU only, tiny models** (LSTM hidden 64, 1 layer, 3 epochs); BGL LSTMs train on about 2,100 samples and are likely undertrained.
* HDFS blocks still open when the sample was cut may be truncated.
* Count-vector Isolation Forest cannot see most HDFS anomalies here (over 75% of anomalous test blocks have the same count vector as normal ones).
* Thresholds are a weak point (see sensitivity table). One IForest threshold (event-ID version) was chosen by best F1 on labelled *train* data.
* Single random seed; no confidence intervals. Synthetic SSH data is easy compared with real attacks.
* Proof of concept; not a production security tool. No Transformer model was built (skipped by plan).
* `shap` is installed but unused: explanations use leave-one-event-out instead, so it is not in `requirements.txt`.

## Ideas from papers and repos, and what I changed
All code is my own and was written without copying from the reference repos (I did not read their code; I only used their data files).
* **Drain** (He et al., ICWS 2017) via the `drain3` library: template mining. I added masking rules for block IDs, IPs, hex and numbers.
* **DeepLog** (Du et al., CCS 2017): next-event prediction with top-k rule. My version adds left-padding for short sequences, an "unseen event is always a miss" rule, and a score (number of misses plus a small surprise tiebreaker) so ROC-AUC can be computed.
* **Isolation Forest** (Liu et al., 2008) on event-count vectors, the usual loglizer-style baseline. I added a count of events unseen in training.
* **Loghub / loglizer** (He et al.): source of the HDFS sample with block labels and of the BGL 2k sample. Only the *data* was used.
* **Template embeddings** (sentence-embedding idea, as in semantic log-anomaly work): MiniLM (Reimers & Gurevych, Sentence-BERT) vectors replace event IDs. My EmbLSTM predicts the next event's *embedding* (cosine loss), not an event ID, so one model can score another dataset.
* Own additions: chronological splits everywhere, target-normal threshold calibration with a sensitivity table, leave-one-event-out explanation, Ollama caching and prompt-injection guard, synthetic SSH generator with labelled injected attacks and a train-contamination assertion.

## Viva preparation
See [VIVA.md](VIVA.md) (10 likely questions with short answers based on this code).
