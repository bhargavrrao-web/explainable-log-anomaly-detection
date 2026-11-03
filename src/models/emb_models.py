"""Embedding-based detectors that work across datasets (no dataset-specific event vocabulary).

1. EmbIForest : Isolation Forest on the MEAN template embedding of a sequence (384-dim).
2. EmbLSTM    : DeepLog-style, but input AND output are embeddings. Given the previous events'
                embeddings, predict the NEXT event's embedding; anomaly score = 1 - cosine(pred, true).
                (We can't use a softmax over event IDs, because IDs differ between datasets.)

Both take an `emb` dict (event_id -> vector) at fit/score time, so the SAME trained model can score
another dataset by passing that dataset's embeddings.
Threshold: calibrate(target_normal_seqs, target_emb) -> percentile of the model's scores on a
sample of NORMAL sequences from the dataset being monitored (no anomaly labels needed).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

from src.models.lstm import make_samples  # (history, target, owner) windows, PAD = 0


def pooled(seqs, emb):
    return np.stack([np.mean([emb[e] for e in s], axis=0) for s in seqs])


class _Base:
    def calibrate(self, seqs, emb, percentile=95):
        self.threshold = float(np.percentile(self.score(seqs, emb), percentile))
        return self

    def predict(self, seqs, emb):
        return (self.score(seqs, emb) > self.threshold).astype(int)


class EmbIForest(_Base):
    def __init__(self, n_estimators=100, seed=42):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, seqs, emb):
        self.model.fit(pooled(seqs, emb))
        return self

    def score(self, seqs, emb):
        return -self.model.score_samples(pooled(seqs, emb))


class _Net(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h[:, -1, :])


def _table(emb):
    """Event-id -> row index (1..V); row 0 is all zeros for padding."""
    ids = sorted(emb)
    idx = {e: i + 1 for i, e in enumerate(ids)}
    tab = torch.zeros(len(ids) + 1, len(next(iter(emb.values()))))
    for e, i in idx.items():
        tab[i] = torch.tensor(emb[e])
    return idx, tab


class EmbLSTM(_Base):
    def __init__(self, hidden=64, layers=1, batch_size=64, epochs=3, lr=1e-3, window=10, seed=42, dim=384):
        self.h, self.l, self.bs, self.epochs, self.lr, self.window, self.seed, self.dim = \
            hidden, layers, batch_size, epochs, lr, window, seed, dim

    def fit(self, seqs, emb):
        torch.manual_seed(self.seed)
        idx, tab = _table(emb)
        X, y, _ = make_samples([[idx[e] for e in s] for s in seqs], self.window)
        self.net = _Net(self.dim, self.h, self.l)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for ep in range(self.epochs):
            perm, tot = torch.randperm(len(X)), 0.0
            for b in range(0, len(X), self.bs):
                i = perm[b:b + self.bs]
                opt.zero_grad()
                # loss = 1 - cosine similarity between predicted and true next-event embedding
                loss = (1 - nn.functional.cosine_similarity(self.net(tab[X[i]]), tab[y[i]])).mean()
                loss.backward()
                opt.step()
                tot += loss.item() * len(i)
            print(f"    EmbLSTM epoch {ep + 1}/{self.epochs} loss={tot / len(X):.4f} ({len(X)} samples)")
        return self

    @torch.no_grad()
    def score(self, seqs, emb):
        """Sequence score = WORST position's (1 - cosine) -- one very surprising event is enough."""
        self.net.eval()
        idx, tab = _table(emb)
        X, y, owner = make_samples([[idx[e] for e in s] for s in seqs], self.window)
        out = np.zeros(len(seqs))
        if len(X):
            pred = torch.cat([self.net(tab[X[b:b + 512]]) for b in range(0, len(X), 512)])
            d = (1 - nn.functional.cosine_similarity(pred, tab[y])).numpy()
            np.maximum.at(out, owner, d)
        return out
