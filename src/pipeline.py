"""Reusable end-to-end pipeline: raw log lines -> Drain templates -> embeddings -> windows -> scores.
Used by the dashboard (upload page) and by the attack simulation (Step E).
Also: `python src/pipeline.py` trains and SAVES the embedding models for the dashboard.
"""
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # MiniLM is already cached; avoid slow network checks

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.embed import clean_template, embed_dataset
from src.models.emb_models import EmbIForest, EmbLSTM
from src.parse import drain_parse
from src.utils import load_config, p, set_seed

MODEL_DIR = "data/models"


def model_path(model_name, source):
    return p(MODEL_DIR) / f"{model_name}_{source}.pkl"


def load_model(model_name, source):
    with open(model_path(model_name, source), "rb") as f:
        return pickle.load(f)


import re
# syslog prefix "Nov  3 10:15:01 hostname " (timestamp + host) carries no event meaning -> strip before Drain
_SYSLOG_PREFIX = re.compile(r"^[A-Z][a-z]{2}\s+\d+\s+[\d:]{8}\s+\S+\s+")


def strip_prefix(line):
    return _SYSLOG_PREFIX.sub("", line.strip())


def parse_and_window(lines, cfg, window=None, step=None):
    """Raw log lines (message text) -> windows of event ids + template->embedding dict.
    A NEW Drain run is used for the new file; embeddings make the result model-compatible."""
    from sentence_transformers import SentenceTransformer
    window, step = window or cfg["bgl"]["window_size"], step or cfg["bgl"]["step"]
    ids, templates = drain_parse(lines, cfg)
    st = SentenceTransformer(cfg["embedding"]["model_name"])
    tids = sorted(templates)
    vecs = st.encode([clean_template(templates[t]) for t in tids], normalize_embeddings=True, show_progress_bar=False)
    emb = dict(zip(tids, vecs.astype(np.float32)))
    wins = [(s, ids[s:s + window]) for s in range(0, max(len(ids) - window, 0) + 1, step)]
    return ids, templates, emb, wins


def train_and_save(cfg):
    """Train EmbIForest + EmbLSTM on each dataset's normal TRAIN sequences; calibrate; pickle."""
    import pandas as pd
    set_seed(cfg["seed"])
    l, pct = cfg["lstm"], cfg["iforest"]["threshold_percentile"]
    p(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    for src in ["HDFS", "BGL"]:
        seqs = pd.read_pickle(p(cfg["paths"]["seq_dir"]) / f"{src}_sequences.pkl")
        tr = seqs[(seqs.split == "train") & (seqs.label == 0)].events.tolist()[: cfg["data"]["max_train_sequences"]]
        emb = embed_dataset(src, cfg)
        for name, m in [("EmbIForest", EmbIForest(cfg["iforest"]["n_estimators"], cfg["seed"])),
                        ("EmbLSTM", EmbLSTM(l["hidden_size"], l["num_layers"], l["batch_size"], l["epochs"], l["lr"], l["window"], cfg["seed"]))]:
            m.fit(tr, emb)
            m.calibrate(tr, emb, pct)
            with open(model_path(name, src), "wb") as f:
                pickle.dump(m, f)
            print(f"saved {name}_{src}  threshold={m.threshold:.4f}")


if __name__ == "__main__":
    train_and_save(load_config())
