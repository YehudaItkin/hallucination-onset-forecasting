"""Theory (task 1): do feature DYNAMICS forecast onset beyond the label chain?

Reviewer-corrected design. The naive 2-state label-Markov "null" is ~0.5 on the
risk set (the only covariate — current label = faithful — is constant there), so
it is not an informative baseline. The honest, MATCHED-CAPACITY null is a label-
history GRU N1': the SAME ForwardGRU trunk fed ONLY causal label/clock features
(within-example index, time-since-last-onset, #prior-onsets, running hallu-rate).
If the feature-GRU (causal-24 text+LM) beats N1' by a paired, document-clustered
bootstrap margin, then feature *dynamics* carry onset-forecasting information
beyond the label history — the claim.

Also: re-estimate the label Markov chain on TRAIN (no hardcoded constants),
assert lambda2 ~ 0.896, and show the analytic onset-within-k rate 1-P_FF^k matches
the empirical base rate (the null is honest).

GO: (N0) N1' bootstrap-CI lower bound > 0.5 at every k (label history alone beats
chance); (MARGIN) paired feature-GRU minus N1' CI lower bound > 0 at k=3 (primary)
and k=5.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score

import hazard_data as H
from run_hazard_a2 import ForwardGRUHazard, DEVICE

HOR = H.HORIZONS
RNG = np.random.RandomState(0)


def estimate_chain(d):
    g = d["groups"]; hallu = (~d["risk"].astype(bool)).astype(int)
    bounds = np.searchsorted(g, np.arange(g[-1] + 2))
    c = np.zeros((2, 2))
    for gi in range(g[-1] + 1):
        s, e = bounds[gi], bounds[gi + 1]
        if e - s < 2:
            continue
        seq = hallu[s:e]
        for a, b in zip(seq[:-1], seq[1:]):
            c[a, b] += 1
    T = c / c.sum(1, keepdims=True).clip(min=1)
    lam = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    return T, lam[1]


def label_features(d):
    """Causal label/clock features known at time t on the risk set."""
    g = d["groups"]; onset = d["onset"].astype(int); hallu = (~d["risk"].astype(bool)).astype(int)
    bounds = np.searchsorted(g, np.arange(g[-1] + 2))
    tsl = np.zeros(len(g), np.float32); npo = np.zeros(len(g), np.float32); hfr = np.zeros(len(g), np.float32)
    posn = np.zeros(len(g), np.float32)
    for gi in range(g[-1] + 1):
        s, e = bounds[gi], bounds[gi + 1]
        if e <= s:
            continue
        last = -1; cnt = 0; run = 0
        for j in range(s, e):
            t = j - s
            posn[j] = t
            tsl[j] = min(t - last, 50) if last >= 0 else 50.0
            npo[j] = cnt
            hfr[j] = run / max(t, 1)
            if onset[j]:           # update AFTER emitting (causal)
                last = t; cnt += 1
            run += hallu[j]
    return np.stack([posn / 50.0, tsl / 50.0, npo, hfr], axis=1).astype(np.float32)


def seqs_from(Xmat, d):
    g = d["groups"]; bounds = np.searchsorted(g, np.arange(g[-1] + 2))
    out = []
    for gi in range(g[-1] + 1):
        s, e = bounds[gi], bounds[gi + 1]
        if e <= s:
            continue
        out.append({"gi": gi, "X": Xmat[s:e], "risk": d["risk"][s:e].astype(bool),
                    "Y": np.stack([d[f"y{k}"][s:e] for k in HOR], 1).astype(np.float32)})
    return out


def std_fit(seqs):
    X = np.concatenate([s["X"][s["risk"]] for s in seqs])
    return X.mean(0), X.std(0) + 1e-6


def batch_iter(seqs, bs, shuffle):
    order = RNG.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(seqs), bs):
        ch = [seqs[j] for j in order[i:i + bs]]
        X = pad_sequence([torch.from_numpy(s["Xs"]) for s in ch], batch_first=True).to(DEVICE)
        Y = pad_sequence([torch.from_numpy(s["Y"]) for s in ch], batch_first=True).to(DEVICE)
        M = pad_sequence([torch.from_numpy(s["risk"]) for s in ch], batch_first=True).to(DEVICE)
        yield ch, X, Y, M


def train_predict(tr, va, te, mean, std, pos_weight, epochs=15, bs=32, seed=42):
    torch.manual_seed(seed)
    for grp in (tr, va, te):
        for s in grp:
            s["Xs"] = ((s["X"] - mean) / std).astype(np.float32)
    model = ForwardGRUHazard(tr[0]["X"].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(reduction="none",
                                 pos_weight=torch.tensor(pos_weight, device=DEVICE))

    def eval_auc(grp):
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for ch, X, Y, M in batch_iter(grp, 64, False):
                p = torch.sigmoid(model(X)).cpu().numpy()
                Yc = Y.cpu().numpy(); Mc = M.cpu().numpy().astype(bool)
                ys.append(Yc[..., 1][Mc]); ps.append(p[..., 1][Mc])
        return roc_auc_score(np.concatenate(ys), np.concatenate(ps))

    best, best_state, pat = -1, None, 0
    for ep in range(epochs):
        model.train()
        for ch, X, Y, M in batch_iter(tr, bs, True):
            opt.zero_grad(); m = M.unsqueeze(-1).float()
            loss = (lossf(model(X), Y) * m).sum() / m.sum().clamp(min=1)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        a = eval_auc(va)
        if a > best:
            best, best_state, pat = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 4:
                break
    model.load_state_dict(best_state)

    model.eval(); P = {k: [] for k in HOR}; Yt = {k: [] for k in HOR}; G = []
    with torch.no_grad():
        for ch, X, Y, M in batch_iter(te, 64, False):
            p = torch.sigmoid(model(X)).cpu().numpy()
            Yc = Y.cpu().numpy(); Mc = M.cpu().numpy().astype(bool)
            for bi, s in enumerate(ch):
                mk = Mc[bi]
                for hi, k in enumerate(HOR):
                    P[k].append(p[bi, :, hi][mk]); Yt[k].append(Yc[bi, :, hi][mk])
                G.append(np.full(int(mk.sum()), s["gi"], np.int32))
    return {k: np.concatenate(P[k]) for k in HOR}, {k: np.concatenate(Yt[k]) for k in HOR}, np.concatenate(G)


def clustered_ci(y, p, groups, B=2000, p2=None):
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(B):
        gs = uniq[RNG.randint(0, len(uniq), len(uniq))]
        take = np.concatenate([idx[g] for g in gs])
        yy = y[take]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, p[take])
        vals.append(a if p2 is None else a - roc_auc_score(yy, p2[take]))
    return np.percentile(vals, [2.5, 50, 97.5])


def main():
    print(f"[theory]  device={DEVICE}", flush=True)
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    assert int(trd["onset"].sum()) == int(trd["n_onsets"][0])
    assert int(ted["onset"].sum()) == int(ted["n_onsets"][0])

    T, lam2 = estimate_chain(trd); P_FF = T[0, 0]
    print(f"\nlabel chain (train): P_FF={P_FF:.4f} P_HH={T[1,1]:.4f} lambda2={lam2:.4f}")
    assert abs(lam2 - 0.896) < 0.02, f"lambda2 {lam2} off"
    print(f"{'k':>3} | {'analytic 1-P_FF^k':>18} | {'empirical base':>15}")
    for k in HOR:
        emp = ted[f"y{k}"][ted["risk"].astype(bool)].mean()
        print(f"{k:>3} | {1 - P_FF**k:>18.4f} | {emp:>15.4f}")

    Xtr_f = trd["X"][:, H.CAUSAL_IDX]; Xte_f = ted["X"][:, H.CAUSAL_IDX]
    Xtr_l = label_features(trd); Xte_l = label_features(ted)
    pw = [((trd[f"y{k}"][trd["risk"].astype(bool)] == 0).sum() /
           max((trd[f"y{k}"][trd["risk"].astype(bool)] == 1).sum(), 1)) for k in HOR]

    def make(mtr, mte):
        tr_all = seqs_from(mtr, trd); te = seqs_from(mte, ted)
        r = np.random.RandomState(1); r.shuffle(tr_all)
        nv = int(0.15 * len(tr_all)); return tr_all[nv:], tr_all[:nv], te

    print("\ntraining feature-GRU (causal-24)...", flush=True)
    tr, va, te = make(Xtr_f, Xte_f); Pf, Yf, Gf = train_predict(tr, va, te, *std_fit(tr), pw)
    print("training N1' label-history GRU (matched capacity)...", flush=True)
    tr, va, te = make(Xtr_l, Xte_l); Pn, Yn, Gn = train_predict(tr, va, te, *std_fit(tr), pw)
    assert np.array_equal(Gf, Gn) and all(np.array_equal(Yf[k], Yn[k]) for k in HOR), "unpaired"

    print(f"\n{'k':>3} | {'feat-GRU':>9} | {'N1 label':>9} | {'margin 95% CI':>24} | {'N1>0.5 CI':>16}")
    print("-" * 78)
    ok_m = True; ok_n0 = True
    for k in HOR:
        af = roc_auc_score(Yf[k], Pf[k]); an = roc_auc_score(Yn[k], Pn[k])
        mlo, mmed, mhi = clustered_ci(Yf[k], Pf[k], Gf, p2=Pn[k])
        n0lo, _, n0hi = clustered_ci(Yn[k], Pn[k], Gn)
        print(f"{k:>3} | {af:>9.3f} | {an:>9.3f} | [{mlo:+.3f},{mhi:+.3f}] m={mmed:+.3f} | [{n0lo:.3f},{n0hi:.3f}]")
        if k in (3, 5) and mlo <= 0:
            ok_m = False
        if n0lo <= 0.5:
            ok_n0 = False
    print(f"\nGATE-theory: N1'>chance(all k)={ok_n0}; feature-dynamics>label-history(k3,k5 CI>0)={ok_m} "
          f"-> {'PASS: features forecast beyond the label chain' if (ok_m and ok_n0) else 'partial/negative'}")


if __name__ == "__main__":
    main()
