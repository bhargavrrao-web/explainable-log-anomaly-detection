"""Isolation Forest on event-count vectors (one count per template, per sequence).

Idea: anomalies are "few and different", so random trees isolate them in fewer splits.
Score = higher means more anomalous. Threshold = a percentile of the TRAIN (normal) scores.
"""
import numpy as np
from sklearn.ensemble import IsolationForest


def count_vectors(seqs, vocab):
    """seqs: list of event-id lists. vocab: dict event_id -> column.
    One EXTRA last column counts events never seen in training (a strong anomaly hint)."""
    X = np.zeros((len(seqs), len(vocab) + 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        for e in s:
            X[i, vocab.get(e, len(vocab))] += 1
    return X


class IForestDetector:
    def __init__(self, n_estimators=100, contamination="auto", seed=42, percentile=95):
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=seed)
        self.percentile = percentile

    def fit(self, normal_seqs):
        # vocabulary = all event ids seen in the (normal) training data
        self.vocab = {e: i for i, e in enumerate(sorted({e for s in normal_seqs for e in s}))}
        X = count_vectors(normal_seqs, self.vocab)
        self.model.fit(X)
        # threshold chosen so ~(100-percentile)% of normal training sequences would be flagged
        self.threshold = np.percentile(self.score(normal_seqs), self.percentile)
        return self

    def calibrate(self, seqs, labels):
        """Pick the threshold with the best F1 on LABELLED TRAIN sequences (never on test data)."""
        sc, labels = self.score(seqs), np.asarray(labels)
        best = (-1, self.threshold)
        for t in np.unique(np.percentile(sc, np.arange(50, 100, 0.5))):
            pred = sc > t
            tp, fp, fn = (pred & (labels == 1)).sum(), (pred & (labels == 0)).sum(), (~pred & (labels == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
            if f1 > best[0]:
                best = (f1, t)
        self.threshold = best[1]
        return self

    def score(self, seqs):
        # sklearn: score_samples is higher for NORMAL points, so negate it
        return -self.model.score_samples(count_vectors(seqs, self.vocab))

    def predict(self, seqs):
        return (self.score(seqs) > self.threshold).astype(int)
