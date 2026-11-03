"""Streamlit dashboard.  Run:  streamlit run app.py
Pages: Anomaly explorer | Model comparison | Upload a log file
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.embed import embed_dataset
from src.explain import contribution_ranking, llm_explain, severity
from src.pipeline import load_model, model_path, parse_and_window, strip_prefix
from src.utils import load_config, p

cfg = load_config()
st.set_page_config(page_title="Log Anomaly Detection", layout="wide")
SEV_ICON = {"high": "ðŸ”´ high", "medium": "ðŸŸ  medium", "low": "ðŸŸ¡ low"}


# ---------- cached loaders ----------
@st.cache_resource
def get_model(name, source):
    return load_model(name, source)


@st.cache_resource
def get_emb(dataset):
    return embed_dataset(dataset, cfg)


@st.cache_data
def get_test(dataset):
    s = pd.read_pickle(p(cfg["paths"]["seq_dir"]) / f"{dataset}_sequences.pkl")
    return s[s.split == "test"].reset_index(drop=True)


@st.cache_data
def get_lines(dataset):
    """Raw log lines per sequence id (HDFS: per block; BGL: per window start line)."""
    df = pd.read_csv(p(cfg["paths"]["parsed_dir"]) / f"{dataset}_parsed.csv", parse_dates=["timestamp"])
    if dataset == "HDFS":
        df["block"] = df["raw"].str.extract(r"(blk_-?\d+)")[0]
        return {b: g["raw"].tolist() for b, g in df.dropna(subset=["block"]).groupby("block")}
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)   # same order as sequences.py
    return {"BGL": df["raw"].tolist()}


def lines_for(dataset, row):
    L = get_lines(dataset)
    if dataset == "HDFS":
        return L[row.seq_id]
    return L["BGL"][int(row.first_line): int(row.first_line) + cfg["bgl"]["window_size"]]


@st.cache_data
def score_test(dataset, model_name, source):
    m, emb = get_model(model_name, source), get_emb(dataset)
    return m.score(get_test(dataset).events.tolist(), emb)


def contribution_chart(contribs, n_lines):
    drops = np.zeros(n_lines)
    for pos, _, d in contribs:
        drops[pos] = d
    fig, ax = plt.subplots(figsize=(7, max(2, 0.28 * n_lines)))
    ax.barh(range(n_lines), drops, color=["#D55E00" if d > 0 else "#9aa5b1" for d in drops])
    ax.set_yticks(range(n_lines), [f"line {i + 1}" for i in range(n_lines)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("score drop when this line is removed (bigger = more responsible)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def detail_view(model, emb, events, lines, key):
    """Shared by the explorer and the upload page: lines + contribution chart + LLM explanation."""
    contribs = contribution_ranking(model, events, emb)
    drop = {pos: d for pos, _, d in contribs}
    st.dataframe(pd.DataFrame({"#": range(1, len(lines) + 1), "contribution": [round(drop.get(i, 0), 4) for i in range(len(lines))],
                               "log line": [l[:220] for l in lines]}), hide_index=True, width="stretch")
    st.pyplot(contribution_chart(contribs, len(lines)))
    top = [lines[pos] for pos, _, d in contribs[:3] if d > 0] or [lines[contribs[0][0]]]
    st.markdown("**Lines sent to the LLM (top contributors):**")
    st.code("\n".join(top), language="text")
    if st.button("Explain with local LLM (llama3.2:1b)", key=f"llm_{key}"):
        with st.spinner("Asking Ollama (first call ~10-30 s, cached afterwards)..."):
            text, cached = llm_explain(top, cfg)
        st.info(text)
        st.caption("served from disk cache" if cached else "generated now (a 1B model can be vague - verify against the lines above)")


# ---------- page 1: explorer ----------
def page_explorer():
    st.title("Anomaly explorer")
    ds = st.sidebar.selectbox("Dataset to monitor", ["HDFS", "BGL"])
    model_name = st.sidebar.selectbox("Model", ["EmbLSTM", "EmbIForest"])
    source = st.sidebar.selectbox("Model trained on", ["HDFS", "BGL"], index=["HDFS", "BGL"].index(ds))
    if not model_path(model_name, source).exists():
        st.error("Saved models missing. Run:  python src/pipeline.py")
        return
    model, test = get_model(model_name, source), get_test(ds)
    scores = score_test(ds, model_name, source)
    lo, hi = float(scores.min()), float(scores.max())
    thr = st.sidebar.slider("Anomaly threshold (score)", lo, hi + 1e-6, float(np.clip(model.threshold, lo, hi)), step=(hi - lo) / 200 or 0.01)
    if source != ds:
        st.warning(f"Cross-dataset: model trained on {source}, scoring {ds}. Threshold default comes from {source}'s own normal data, "
                   f"so expect poor results (see Model comparison page).")
    flagged = scores > thr
    fs = scores[flagged]
    df = pd.DataFrame({"sequence": test.seq_id, "start": test.start_time, "score": scores.round(4),
                       "severity": [SEV_ICON[severity(fs, s)] if f else "" for s, f in zip(scores, flagged)],
                       "ground truth": np.where(test.label == 1, "anomalous", "normal"), "flagged": flagged})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test sequences", len(test))
    c2.metric("Flagged", int(flagged.sum()))
    tp = int((flagged & (test.label == 1)).sum())
    c3.metric("Flagged & truly anomalous", tp, f"precision {tp / max(1, flagged.sum()):.2f}")
    c4.metric("Recall", f"{tp / max(1, (test.label == 1).sum()):.2f}")
    st.subheader("Flagged sequences (highest score first)")
    view = df[df.flagged].sort_values("score", ascending=False).drop(columns="flagged").head(500)
    st.dataframe(view, hide_index=True, width="stretch")
    if view.empty:
        st.info("Nothing flagged at this threshold - lower it in the sidebar.")
        return
    st.subheader("Detail")
    pick = st.selectbox("Sequence", view["sequence"].tolist()[:200])
    i = int(test.index[test.seq_id == pick][0])
    row = test.iloc[i]
    st.write(f"**{pick}** | score {scores[i]:.3f} | severity {severity(fs, scores[i])} | ground truth: "
             f"{'anomalous' if row.label else 'normal'}")
    detail_view(model, get_emb(ds), list(row.events), lines_for(ds, row), f"{ds}_{pick}")


# ---------- page 2: comparison ----------
def page_comparison():
    st.title("Model comparison and cross-dataset results")
    res = p(cfg["paths"]["results_dir"])
    st.subheader("In-dataset baselines (event-ID features)")
    if (res / "metrics.csv").exists():
        st.dataframe(pd.read_csv(res / "metrics.csv"), hide_index=True, width="stretch")
        st.image(str(res / "confusion_baselines.png"))
    st.subheader("Embedding models: in-dataset and cross-dataset")
    if (res / "cross_dataset.csv").exists():
        st.dataframe(pd.read_csv(res / "cross_dataset.csv"), hide_index=True, width="stretch")
        st.image(str(res / "cross_dataset.png"))
    sens = res / "cross_dataset_threshold_sensitivity.csv"
    if sens.exists():
        st.subheader("F1 vs threshold-calibration percentile")
        st.dataframe(pd.read_csv(sens).pivot_table(index=["model", "experiment"], columns="percentile", values="f1").round(3))
    st.caption("BGL results use only a 2,000-line sample (60 test windows) and are noisy. Cross-dataset scores are mostly poor - see README.")


# ---------- page 3: upload ----------
def page_upload():
    st.title("Upload a log file")
    st.write("Each line is parsed with Drain, embedded with MiniLM, grouped into windows "
             f"({cfg['bgl']['window_size']} lines, step {cfg['bgl']['step']}) and scored by a saved model. Max 20,000 lines.")
    model_name = st.sidebar.selectbox("Model", ["EmbLSTM", "EmbIForest"])
    source = st.sidebar.selectbox("Model trained on", ["HDFS", "BGL"], index=1)
    up = st.file_uploader("Log file (.log / .txt)", type=["log", "txt"])
    if not up:
        return
    lines = [strip_prefix(l) for l in up.getvalue().decode("utf-8", "ignore").splitlines() if l.strip()][:20000]
    if len(lines) < cfg["bgl"]["window_size"]:
        st.error("Need at least one window of lines.")
        return
    with st.spinner("Parsing and embedding..."):
        ids, templates, emb, wins = parse_and_window(lines, cfg)
    model = get_model(model_name, source)
    scores = model.score([w for _, w in wins], emb)
    st.write(f"{len(lines)} lines -> {len(templates)} templates -> {len(wins)} windows")
    thr = st.slider("Threshold", float(scores.min()), float(scores.max()) + 1e-6, float(np.clip(model.threshold, scores.min(), scores.max())))
    flagged = np.where(scores > thr)[0]
    fs = scores[flagged]
    st.dataframe(pd.DataFrame({"window": flagged, "first line": [wins[i][0] + 1 for i in flagged], "score": scores[flagged].round(4),
                               "severity": [SEV_ICON[severity(fs, s)] for s in fs]}).sort_values("score", ascending=False),
                 hide_index=True, width="stretch")
    if len(flagged):
        k = st.selectbox("Window", flagged.tolist())
        s0, w = wins[k]
        detail_view(model, emb, w, lines[s0:s0 + len(w)], f"up_{k}")


page = st.sidebar.radio("Page", ["Anomaly explorer", "Model comparison", "Upload a log file"])
{"Anomaly explorer": page_explorer, "Model comparison": page_comparison, "Upload a log file": page_upload}[page]()

