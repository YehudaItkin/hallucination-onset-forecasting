"""A2 (part 1): ForwardGRU hazard head vs the logistic floor.

Does modeling the feature *trajectory* with a causal recurrent net beat the
pointwise logistic hazard of A1 — and does it rescue the LM signal (the A1 caveat
was that logistic is linear)?

ForwardGRU is the paper's exact unidirectional GRU (run_directional_ablation.py:
2-layer, h=64, dropout 0.1, bidirectional=False) — causal by construction, so with
causal-24 features the whole pipeline is streaming-safe. Head adapted to emit one
hazard logit per horizon k in {1,3,5,10}. Trained with masked multi-horizon BCE
over the risk set (faithful tokens).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, average_precision_score

import hazard_data as H

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOR = H.HORIZONS


class ForwardGRUHazard(nn.Module):
    """Paper's ForwardGRU trunk; head emits one hazard logit per horizon."""
    def __init__(self, dim, h=64, n_hor=len(HOR)):
        super().__init__()
        self.gru = nn.GRU(dim, h, num_layers=2, batch_first=True,
                          bidirectional=False, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, n_hor))

    def forward(self, x):
        return self.head(self.gru(x)[0])          # (B, T, n_hor) logits


def to_sequences(d):
    """Split the flat token cache back into per-example sequences by group id."""
    groups = d["groups"]
    bounds = np.searchsorted(groups, np.arange(groups[-1] + 2))
    seqs = []
    for gi in range(groups[-1] + 1):
        s, e = bounds[gi], bounds[gi + 1]
        if e <= s:
            continue
        seqs.append({
            "gi": gi,                                    # original example id (A3 alignment)
            "X": d["X"][s:e],
            "risk": d["risk"][s:e].astype(bool),
            "Y": np.stack([d[f"y{k}"][s:e] for k in HOR], axis=1).astype(np.float32),
        })
    return seqs


def standardize(seqs, mean, std, idx):
    for s in seqs:
        s["Xs"] = ((s["X"][:, idx] - mean) / std).astype(np.float32)


def batches(seqs, bs, shuffle):
    order = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        chunk = [seqs[j] for j in order[i:i + bs]]
        X = pad_sequence([torch.from_numpy(s["Xs"]) for s in chunk], batch_first=True)
        Y = pad_sequence([torch.from_numpy(s["Y"]) for s in chunk], batch_first=True)
        M = pad_sequence([torch.from_numpy(s["risk"]) for s in chunk], batch_first=True)
        yield X.to(DEVICE), Y.to(DEVICE), M.to(DEVICE)


def train_eval(tr, va, te, idx, pos_weight, epochs=15, bs=32, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ForwardGRUHazard(len(idx)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor(pos_weight, device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    best_auc, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train()
        for X, Y, M in batches(tr, bs, True):
            opt.zero_grad()
            logit = model(X)
            m = M.unsqueeze(-1).float()
            loss = (lossf(logit, Y) * m).sum() / m.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        va_auc = evaluate(model, va)[f"auc@{HOR[1]}"]   # early-stop on k=3 AUROC
        if va_auc > best_auc:
            best_auc, best_state, patience = va_auc, {k: v.cpu().clone()
                                                      for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return model, evaluate(model, te)     # return trained model (A3 reuses it)


@torch.no_grad()
def evaluate(model, seqs, bs=64):
    model.eval()
    ys = {k: [] for k in HOR}; ps = {k: [] for k in HOR}
    for X, Y, M in batches(seqs, bs, False):
        logit = model(X)
        prob = torch.sigmoid(logit).cpu().numpy()
        Yc, Mc = Y.cpu().numpy(), M.cpu().numpy().astype(bool)
        for hi, k in enumerate(HOR):
            ys[k].append(Yc[..., hi][Mc]); ps[k].append(prob[..., hi][Mc])
    out = {}
    for k in HOR:
        y = np.concatenate(ys[k]); p = np.concatenate(ps[k])
        out[f"auc@{k}"] = roc_auc_score(y, p)
        out[f"ap@{k}"] = average_precision_score(y, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fset", choices=["causal-24", "full-33"], default="causal-24")
    a = ap.parse_args()
    idx = H.CAUSAL_IDX if a.fset == "causal-24" else list(range(33))

    print(f"[A2 ForwardGRU-hazard]  device={DEVICE}  fset={a.fset} (dim={len(idx)})",
          flush=True)
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    tr_all = to_sequences(trd); te = to_sequences(ted)
    rng = np.random.RandomState(0); rng.shuffle(tr_all)
    n_val = int(0.15 * len(tr_all)); va, tr = tr_all[:n_val], tr_all[n_val:]

    # standardize on train risk tokens; pos_weight per horizon from train risk set
    rmask = trd["risk"].astype(bool)
    Xr = trd["X"][rmask][:, idx]
    mean, std = Xr.mean(0), Xr.std(0) + 1e-6
    standardize(tr, mean, std, idx); standardize(va, mean, std, idx); standardize(te, mean, std, idx)
    pos_weight = []
    for k in HOR:
        yk = trd[f"y{k}"][rmask]
        pos_weight.append((yk == 0).sum() / max((yk == 1).sum(), 1))

    _, res = train_eval(tr, va, te, idx, pos_weight)

    print(f"\n  {'k':>3} | {'ForwardGRU AUROC/PR':>22} | {'A1 logistic (causal-24)':>24}")
    a1 = {1: (.683, .008), 3: (.657, .020), 5: (.647, .032), 10: (.639, .058)}
    print("-" * 58)
    for k in HOR:
        g = f"{res[f'auc@{k}']:.3f}/{res[f'ap@{k}']:.3f}"
        print(f"  {k:>3} | {g:>22} | {a1[k][0]:.3f}/{a1[k][1]:.3f}")
    print(f"\nGATE-A2  did GRU beat logistic @k=3?  "
          f"{res['auc@3']:.3f} vs 0.657  "
          f"-> {'YES' if res['auc@3'] > 0.657 else 'no gain'}")


if __name__ == "__main__":
    main()
