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


def cross_main():
    """STEP C: train on SOURCE dataset, test on TARGET (HDFS->HDFS, BGL->BGL, HDFS->BGL, BGL->HDFS).
    Threshold = 95th percentile of the source model's scores on the TARGET's normal TRAIN sequences."""
    from src.embed import embed_dataset
    from src.models.emb_models import EmbIForest, EmbLSTM
    cfg = load_config()
    set_seed(cfg["seed"])
    l, pct = cfg["lstm"], cfg["iforest"]["threshold_percentile"]
    emb = {n: embed_dataset(n, cfg) for n in ["HDFS", "BGL"]}
    split = {n: load_split(n, cfg) for n in ["HDFS", "BGL"]}
    rows, sens = [], []
    for src in ["HDFS", "BGL"]:
        normal = split[src][0]
        normal = normal[normal.label == 0].events.tolist()[: cfg["data"]["max_train_sequences"]]
        models = {"EmbIForest": EmbIForest(cfg["iforest"]["n_estimators"], cfg["seed"]),
                  "EmbLSTM": EmbLSTM(l["hidden_size"], l["num_layers"], l["batch_size"], l["epochs"], l["lr"], l["window"], cfg["seed"])}
        for mname, m in models.items():
            t0 = time.time()
            m.fit(normal, emb[src])
            fit_s = time.time() - t0
            for tgt in ["HDFS", "BGL"]:
                tr, te = split[tgt]
                tr_normal = tr[tr.label == 0].events.tolist()[: cfg["data"]["max_train_sequences"]]
                m.calibrate(tr_normal, emb[tgt], pct)            # target-normal calibration (no anomaly labels)
                ev = te.events.tolist()
                r = metrics(te.label.values, m.predict(ev, emb[tgt]), m.score(ev, emb[tgt]))
                r.update(model=mname, experiment=f"{src}->{tgt}", train_seconds=round(fit_s, 1))
                rows.append(r)
                for q in [80, 90, 99]:                           # sensitivity of F1 to the calibration percentile
                    m.calibrate(tr_normal, emb[tgt], q)
                    sens.append(dict(model=mname, experiment=f"{src}->{tgt}", percentile=q,
                                     f1=metrics(te.label.values, m.predict(ev, emb[tgt]), m.score(ev, emb[tgt]))["f1"]))
                print(f"  {mname:10s} {src}->{tgt}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} AUC={r['roc_auc']:.3f} "
                      f"| TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']}")
    df = pd.DataFrame(rows)[["model", "experiment", "precision", "recall", "f1", "roc_auc", "TP", "FP", "FN", "TN", "train_seconds"]]
    out = p(cfg["paths"]["results_dir"])
    df.round(4).to_csv(out / "cross_dataset.csv", index=False)
    sens.extend({"model": r["model"], "experiment": r["experiment"], "percentile": pct, "f1": r["f1"]} for r in rows)
    sd = pd.DataFrame(sens)
    sd.round(4).to_csv(out / "cross_dataset_threshold_sensitivity.csv", index=False)
    print("\n F1 vs calibration percentile:\n", sd.pivot_table(index=["model", "experiment"], columns="percentile", values="f1").round(3).to_string())
    print("\n", df.pivot(index="model", columns="experiment", values="f1").round(3).to_string())
    print("\n AUC:\n", df.pivot(index="model", columns="experiment", values="roc_auc").round(3).to_string())

    # bar chart: F1 (left) and ROC-AUC (right), grouped by experiment
    exps = ["HDFS->HDFS", "BGL->BGL", "HDFS->BGL", "BGL->HDFS"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, col, ttl in zip(axes, ["f1", "roc_auc"], ["F1 (anomaly class)", "ROC-AUC"]):
        for k, (mn, color) in enumerate([("EmbIForest", "#4C78A8"), ("EmbLSTM", "#F58518")]):
            v = [df[(df.model == mn) & (df.experiment == e)][col].iloc[0] for e in exps]
            bars = ax.bar(np.arange(4) + (k - 0.5) * 0.38, v, 0.38, label=mn, color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(range(4), exps, fontsize=8)
        ax.set_title(ttl)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "cross_dataset.png", dpi=130)
    print("\nSaved results/cross_dataset.csv and results/cross_dataset.png")


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
    cross_main() if "cross" in sys.argv[1:] else main()
