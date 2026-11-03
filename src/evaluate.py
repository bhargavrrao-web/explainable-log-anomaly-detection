"""Evaluation helpers + Step B baseline run (in-dataset only).
Run:  python src/evaluate.py
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.iforest import IForestDetector
from src.models.lstm import LSTMDetector
from src.utils import load_config, p, set_seed


def load_split(name, cfg):
    seqs = pd.read_pickle(p(cfg["paths"]["seq_dir"]) / f"{name}_sequences.pkl")
    return seqs[seqs.split == "train"], seqs[seqs.split == "test"]


def metrics(y_true, y_pred, scores):
    """Precision/recall/F1 for the anomaly class + confusion matrix + threshold-free ROC-AUC."""
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else float("nan")
    return dict(precision=pr, recall=rc, f1=f1, roc_auc=auc, TP=tp, FP=fp, FN=fn, TN=tn)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    rows = []
    for name in ["HDFS", "BGL"]:
        train, test = load_split(name, cfg)
        # anomaly models learn from NORMAL training sequences only, capped for speed
        normal = train[train.label == 0].events.tolist()[: cfg["data"]["max_train_sequences"]]
        y = test.label.values
        print(f"\n##### {name}: train on {len(normal)} normal seqs, test on {len(test)} ({y.sum()} anomalous)")

        iso_cfg, l = cfg["iforest"], cfg["lstm"]
        models = {
            "IsolationForest": IForestDetector(iso_cfg["n_estimators"], iso_cfg["contamination"], cfg["seed"],
                                               iso_cfg.get("threshold_percentile", 95)),
            "LSTM(DeepLog)": LSTMDetector(l["hidden_size"], l["num_layers"], l["batch_size"], l["epochs"],
                                          l["lr"], l["window"], l["top_k"], cfg["seed"]),
        }
        for mname, m in models.items():
            t0 = time.time()
            m.fit(normal)
            if mname == "IsolationForest":   # threshold calibrated on labelled TRAIN data only
                m.calibrate(train.events.tolist(), train.label.values)
            fit_s = time.time() - t0
            r = metrics(y, m.predict(test.events.tolist()), m.score(test.events.tolist()))
            r.update(dataset=name, model=mname, train_seconds=round(fit_s, 1))
            rows.append(r)
            print(f"  {mname:16s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
                  f"AUC={r['roc_auc']:.3f} | TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']} | train {fit_s:.1f}s")
            if r["f1"] > 0.99:
                print("  !! suspiciously perfect: check for leakage")

    df = pd.DataFrame(rows)[["dataset", "model", "precision", "recall", "f1", "roc_auc", "TP", "FP", "FN", "TN", "train_seconds"]]
    out = p(cfg["paths"]["results_dir"])
    out.mkdir(exist_ok=True)
    df.round(4).to_csv(out / "metrics.csv", index=False)

    # confusion-matrix grid: one panel per (dataset, model)
    fig, axes = plt.subplots(1, len(df), figsize=(3.3 * len(df), 3.2))
    for ax, (_, r) in zip(axes, df.iterrows()):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        ax.set_xticks([0, 1], ["pred normal", "pred anom"], fontsize=8)
        ax.set_yticks([0, 1], ["true normal", "true anom"], fontsize=8)
        ax.set_title(f"{r.dataset} / {r.model}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "confusion_baselines.png", dpi=130)
    print("\nSaved results/metrics.csv and results/confusion_baselines.png")


if __name__ == "__main__":
    main()
