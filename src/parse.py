"""Step 1: turn raw log lines into templates with Drain (drain3 library).

Drain groups similar lines into a "template", e.g.
  "Receiving block blk_123 src: /10.0.0.1:5000"  ->  "Receiving block <BLK> src: <IP>"
Each template gets an integer event_id. Output CSV columns:
  raw, timestamp, template, event_id, label (label = 1 anomalous / 0 normal / -1 unknown)
"""
import re
import sys

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.utils import load_config, p, set_seed


def make_miner(cfg):
    """Create a Drain miner. Masking replaces volatile tokens BEFORE clustering."""
    c = TemplateMinerConfig()
    c.drain_depth = cfg["drain"]["depth"]
    c.drain_sim_th = cfg["drain"]["sim_th"]
    c.drain_max_children = cfg["drain"]["max_children"]
    c.profiling_enabled = False
    c.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+", "BLK"),                      # HDFS block ids
        MaskingInstruction(r"\d+\.\d+\.\d+\.\d+(:\d+)?", "IP"),       # IPv4 (+port)
        MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),                 # hex numbers
        MaskingInstruction(r"\b\d+\b", "NUM"),                        # plain numbers
    ]
    return TemplateMiner(config=c)


def drain_parse(contents, cfg):
    """Run Drain over a list of message strings. Returns (event_ids, id->template dict).
    Templates evolve as more lines arrive, so we read final templates at the end."""
    miner = make_miner(cfg)
    ids = []
    for msg in contents:
        res = miner.add_log_message(msg)
        ids.append(res["cluster_id"])
    templates = {cl.cluster_id: cl.get_template() for cl in miner.drain.clusters}
    return ids, templates


def parse_hdfs(cfg):
    """HDFS: use the Content column of the loglizer 100k sample (it has date/time but no
    labels per line; labels are per block and applied later in sequences.py)."""
    df = pd.read_csv(p(cfg["paths"]["hdfs_raw"]), nrows=cfg["data"]["max_lines"], dtype=str)
    # Rebuild the raw line, and parse the timestamp "081109 203518" -> datetime
    raw = df["Date"] + " " + df["Time"] + " " + df["Pid"] + " " + df["Level"] + " " + df["Component"] + ": " + df["Content"]
    ts = pd.to_datetime(df["Date"] + df["Time"], format="%y%m%d%H%M%S")
    ids, templates = drain_parse(df["Content"].tolist(), cfg)
    out = pd.DataFrame({
        "raw": raw, "timestamp": ts, "content": df["Content"],
        "template": [templates[i] for i in ids], "event_id": ids,
        "label": -1,  # per-line labels do not exist for HDFS (block-level only)
    })
    return out, templates


# BGL line format (space separated):
# Label Timestamp Date Node Time NodeRepeat Type Component Level Content...
def parse_bgl(cfg):
    rows = []
    with open(p(cfg["paths"]["bgl_raw"]), "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= cfg["data"]["max_lines"]:
                break
            parts = line.rstrip("\n").split(" ", 9)
            if len(parts) < 10:
                continue
            rows.append({"raw": line.rstrip("\n"),
                         "timestamp": pd.to_datetime(int(parts[1]), unit="s"),
                         "label": 0 if parts[0] == "-" else 1,   # '-' = normal, else an alert category
                         "content": parts[9]})
    df = pd.DataFrame(rows)
    ids, templates = drain_parse(df["content"].tolist(), cfg)
    df["event_id"] = ids
    df["template"] = [templates[i] for i in ids]
    return df[["raw", "timestamp", "content", "template", "event_id", "label"]], templates


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    out_dir = p(cfg["paths"]["parsed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in [("HDFS", parse_hdfs), ("BGL", parse_bgl)]:
        df, templates = fn(cfg)
        df.to_csv(out_dir / f"{name}_parsed.csv", index=False)
        # template table (used later for embeddings): event_id -> template text
        pd.DataFrame({"event_id": list(templates), "template": list(templates.values())}) \
            .sort_values("event_id").to_csv(out_dir / f"{name}_templates.csv", index=False)
        print(f"\n=== {name}: {len(df)} lines -> {len(templates)} templates ===")
        for eid, t in list(templates.items())[:5]:
            print(f"  E{eid}: {t[:110]}")


if __name__ == "__main__":
    main()
