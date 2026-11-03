"""Step 2: build event sequences from parsed logs + chronological train/test split.

HDFS: one sequence per block id (all events that touch blk_xxx), label from anomaly_label.csv
BGL : sliding windows (size/step from config); window is anomalous if ANY line in it is anomalous.
Output (pickle) columns: seq_id, events (list of event_ids), label (0/1), start_time, split
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, p, set_seed


def hdfs_sequences(cfg):
    df = pd.read_csv(p(cfg["paths"]["parsed_dir"]) / "HDFS_parsed.csv", parse_dates=["timestamp"])
    df["block"] = df["raw"].str.extract(r"(blk_-?\d+)")[0]
    df = df.dropna(subset=["block"])
    labels = pd.read_csv(p(cfg["paths"]["hdfs_labels"])).set_index("BlockId")["Label"]
    # lines are already in time order; groupby keeps that order inside each block
    g = df.groupby("block").agg(events=("event_id", list), start_time=("timestamp", "min"))
    g["label"] = (g.index.map(labels) == "Anomaly").astype(int)
    g = g.reset_index().rename(columns={"block": "seq_id"})
    return g


def bgl_sequences(cfg):
    df = pd.read_csv(p(cfg["paths"]["parsed_dir"]) / "BGL_parsed.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    w, s = cfg["bgl"]["window_size"], cfg["bgl"]["step"]
    rows = []
    for start in range(0, len(df) - w + 1, s):
        chunk = df.iloc[start:start + w]
        rows.append({"seq_id": f"win_{start}", "events": chunk["event_id"].tolist(),
                     "label": int(chunk["label"].max() > 0), "start_time": chunk["timestamp"].iloc[0],
                     "first_line": start})
    return pd.DataFrame(rows)


def chrono_split(seqs, frac):
    """First `frac` of sequences (by start time) = train, the rest = test. Never random."""
    seqs = seqs.sort_values("start_time", kind="stable").reset_index(drop=True)
    cut = int(len(seqs) * frac)
    seqs["split"] = ["train"] * cut + ["test"] * (len(seqs) - cut)
    return seqs


def balance(seqs, name):
    print(f"\n=== {name}: {len(seqs)} sequences, mean length {seqs['events'].map(len).mean():.1f} ===")
    for sp in ["train", "test"]:
        d = seqs[seqs.split == sp]
        print(f"  {sp:5s}: {len(d):6d} seqs | normal={int((d.label == 0).sum()):6d} anomalous={int((d.label == 1).sum()):5d} "
              f"({100 * d.label.mean():.1f}% anomalous) | {d.start_time.min()} -> {d.start_time.max()}")


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    out = p(cfg["paths"]["seq_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in [("HDFS", hdfs_sequences), ("BGL", bgl_sequences)]:
        seqs = chrono_split(fn(cfg), cfg["data"]["train_fraction"])
        seqs.to_pickle(out / f"{name}_sequences.pkl")
        balance(seqs, name)


if __name__ == "__main__":
    main()
