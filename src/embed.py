"""Turn Drain templates into dense vectors so different datasets share ONE feature space.

Event IDs mean different things in HDFS and BGL (E1 is unrelated), but the TEXT of a template
("Receiving block <BLK> ...") can be embedded with a sentence model. Similar meaning -> nearby vectors.
Primary: all-MiniLM-L6-v2 (384-dim, L2-normalised). Fallback: TF-IDF (+SVD) if the model is unavailable.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, p

DIM = 384


def clean_template(t):
    """Drop Drain placeholders like <NUM>, <BLK>, <*> and punctuation noise; keep the words."""
    t = re.sub(r"<[A-Z*]+>", " ", t)
    t = re.sub(r"[^A-Za-z0-9_.:\- ]", " ", t)
    return re.sub(r"\s+", " ", t).strip() or "empty"


def _minilm(texts, cfg):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["embedding"]["model_name"])
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _tfidf_fallback(texts):
    """Fallback only: TF-IDF fitted on the given texts, projected to DIM dims. NOT comparable across
    datasets unless fitted on both together, so cross-dataset scores with it are less meaningful."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer().fit_transform(texts)
    k = max(1, min(DIM, X.shape[1] - 1, X.shape[0] - 1))
    Z = TruncatedSVD(k, random_state=0).fit_transform(X)
    Z = np.pad(Z, ((0, 0), (0, DIM - k)))
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def embed_dataset(name, cfg=None):
    """Returns dict event_id -> 384-dim vector for dataset `name`. Cached to disk."""
    cfg = cfg or load_config()
    cache = p(cfg["paths"]["seq_dir"]) / f"{name}_template_emb.npz"
    tpl = pd.read_csv(p(cfg["paths"]["parsed_dir"]) / f"{name}_templates.csv")
    if cache.exists():
        z = np.load(cache)
        return dict(zip(z["ids"].tolist(), z["vecs"]))
    texts = [clean_template(t) for t in tpl["template"]]
    try:
        vecs, how = _minilm(texts, cfg), "MiniLM"
    except Exception as e:  # e.g. model not downloadable
        print(f"!! MiniLM unavailable ({e}); falling back to TF-IDF")
        vecs, how = _tfidf_fallback(texts), "TF-IDF"
    np.savez(cache, ids=tpl["event_id"].values, vecs=vecs.astype(np.float32))
    print(f"  embedded {len(texts)} {name} templates with {how} -> {vecs.shape}")
    return dict(zip(tpl["event_id"].tolist(), vecs.astype(np.float32)))


if __name__ == "__main__":
    cfg = load_config()
    embs = {n: embed_dataset(n, cfg) for n in ["HDFS", "BGL"]}
    # sanity check: which BGL template is closest to each of a few HDFS templates?
    for n in embs:
        print(n, len(embs[n]), "templates")
    H = pd.read_csv(p(cfg["paths"]["parsed_dir"]) / "HDFS_templates.csv").set_index("event_id")["template"]
    B = pd.read_csv(p(cfg["paths"]["parsed_dir"]) / "BGL_templates.csv").set_index("event_id")["template"]
    bid = list(embs["BGL"]); Bm = np.stack([embs["BGL"][i] for i in bid])
    for hid in list(embs["HDFS"])[:4]:
        j = int(np.argmax(Bm @ embs["HDFS"][hid]))
        print(f"HDFS: {H[hid][:60]!r}\n   nearest BGL: {B[bid[j]][:60]!r} (cos={float(Bm[j] @ embs['HDFS'][hid]):.2f})")
