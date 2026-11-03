"""DeepLog-style LSTM on event IDs.

Train (normal data only): given the previous `window` events, predict the next event.
Detect: a sequence is anomalous if, at ANY position, the true next event is not among
the model's top-k predictions. Unseen events (not in train vocab) always count as a miss.
"""
import numpy as np
import torch
import torch.nn as nn

PAD, UNK = 0, 1  # special tokens; real events start at index 2


class NextEventLSTM(nn.Module):
    def __init__(self, vocab_size, hidden=64, layers=1, emb_dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x):                     # x: (batch, window) of token ids
        h, _ = self.lstm(self.emb(x))
        return self.out(h[:, -1, :])          # logits for the next event


def make_samples(seqs, window):
    """For every position i>=1 in every sequence: history = previous `window` tokens
    (left-padded with PAD), target = token at i. Also returns which sequence each sample is from."""
    X, y, owner = [], [], []
    for sid, s in enumerate(seqs):
        for i in range(1, len(s)):
            hist = s[max(0, i - window):i]
            X.append([PAD] * (window - len(hist)) + hist)
            y.append(s[i])
            owner.append(sid)
    return (torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long), np.array(owner))


class LSTMDetector:
    def __init__(self, hidden=64, layers=1, batch_size=64, epochs=3, lr=1e-3, window=10, top_k=9, seed=42):
        self.hidden, self.layers, self.bs, self.epochs = hidden, layers, batch_size, epochs
        self.lr, self.window, self.top_k, self.seed = lr, window, top_k, seed

    def encode(self, seqs):
        return [[self.vocab.get(e, UNK) for e in s] for s in seqs]

    def fit(self, normal_seqs):
        torch.manual_seed(self.seed)
        self.vocab = {e: i + 2 for i, e in enumerate(sorted({e for s in normal_seqs for e in s}))}
        X, y, _ = make_samples(self.encode(normal_seqs), self.window)
        self.model = NextEventLSTM(len(self.vocab) + 2, self.hidden, self.layers)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(len(X))
            total = 0.0
            for b in range(0, len(X), self.bs):
                idx = perm[b:b + self.bs]
                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]), y[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            print(f"    LSTM epoch {ep + 1}/{self.epochs}  loss={total / len(X):.4f}  ({len(X)} samples)")
        return self

    @torch.no_grad()
    def _per_sample(self, seqs):
        """Returns per-sample (is_miss, surprise=-log p(true)) and the owner sequence ids."""
        self.model.eval()
        X, y, owner = make_samples(self.encode(seqs), self.window)
        if len(X) == 0:
            return np.zeros(0, bool), np.zeros(0), owner
        logits = torch.cat([self.model(X[b:b + 512]) for b in range(0, len(X), 512)])
        topk = logits.topk(min(self.top_k, logits.shape[1]), dim=1).indices
        miss = ~(topk == y[:, None]).any(dim=1)
        miss |= (y == UNK)                                   # unseen event = always a miss
        surprise = -torch.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
        return miss.numpy(), surprise.numpy(), owner

    def score(self, seqs):
        """Sequence anomaly score = number of top-k misses (0 = looks normal).
        Ties are broken by the largest surprise, so the score is also usable for ROC-AUC."""
        miss, surprise, owner = self._per_sample(seqs)
        n_miss, max_s = np.zeros(len(seqs)), np.zeros(len(seqs))
        np.add.at(n_miss, owner, miss.astype(float))
        np.maximum.at(max_s, owner, surprise)
        return n_miss + max_s / (max_s.max() + 1e-9) * 0.5   # misses dominate; surprise only breaks ties

    def predict(self, seqs):
        miss, _, owner = self._per_sample(seqs)
        flagged = np.zeros(len(seqs), int)
        flagged[np.unique(owner[miss])] = 1
        return flagged
