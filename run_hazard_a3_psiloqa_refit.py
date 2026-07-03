"""A3(ii) disambiguation: is onset forecastable IN PsiloQA at all?

The zero-shot RAGTruth->PsiloQA transfer failed (AUROC 0.32-0.43, inverting). Two
explanations: (a) the anticipatory signal is DOMAIN-SPECIFIC (exists in PsiloQA but
does not transfer), or (b) onset is simply not forecastable in PsiloQA. Refit
IN-DOMAIN: train the causal ForwardGRU-hazard on PsiloQA-train (its OWN lm_medians
+ standardization) and evaluate on PsiloQA-test.

GATE: in-domain PsiloQA AUROC@3 >= 0.60 -> signal EXISTS but is domain-specific
(confirms transfer-limit, not signal-absence). ~0.5 -> onset not forecastable in
PsiloQA (e.g. 96%-positive short QA leaves too little faithful runway).
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import hazard_data as H
from run_hazard_a2 import to_sequences, standardize, train_eval, batches, HOR

IDX = H.CAUSAL_IDX


def guarded(model, seqs):
    model.eval(); ys = {k: [] for k in HOR}; ps = {k: [] for k in HOR}
    with torch.no_grad():
        for X, Y, M in batches(seqs, 64, False):
            p = torch.sigmoid(model(X)).cpu().numpy()
            Yc, Mc = Y.cpu().numpy(), M.cpu().numpy().astype(bool)
            for hi, k in enumerate(HOR):
                ys[k].append(Yc[..., hi][Mc]); ps[k].append(p[..., hi][Mc])
    out = {}
    for k in HOR:
        y = np.concatenate(ys[k]); p = np.concatenate(ps[k])
        out[k] = (float("nan"), y.mean()) if y.min() == y.max() else (roc_auc_score(y, p), y.mean())
    return out


def main():
    print("[A3-psiloqa-refit] in-domain PsiloQA forecasting (own lm_medians)...", flush=True)
    pqtr, pqmed = H.build("psiloqa_train", cache_tag="_pq")
    pqte, _ = H.build("psiloqa_test", lm_medians=pqmed, cache_tag="_pq")
    tr_all = to_sequences(pqtr); te = to_sequences(pqte)
    rng = np.random.RandomState(0); rng.shuffle(tr_all)
    nv = int(0.15 * len(tr_all)); va, tr = tr_all[:nv], tr_all[nv:]
    rmask = pqtr["risk"].astype(bool); Xr = pqtr["X"][rmask][:, IDX]
    mean, std = Xr.mean(0), Xr.std(0) + 1e-6
    standardize(tr, mean, std, IDX); standardize(va, mean, std, IDX); standardize(te, mean, std, IDX)
    pw = [((pqtr[f"y{k}"][rmask] == 0).sum() / max((pqtr[f"y{k}"][rmask] == 1).sum(), 1)) for k in HOR]
    print(f"PsiloQA risk tokens: train {rmask.sum()}, test {pqte['risk'].sum()}", flush=True)

    model, _ = train_eval(tr, va, te, IDX, pw)
    res = guarded(model, te)

    print(f"\n  {'k':>3} | {'PsiloQA in-domain':>17} | {'base%':>6} | {'(RAGTruth 0.78, zero-shot):':>28}")
    zs = {1: 0.434, 3: 0.383, 5: 0.349, 10: 0.319}
    rid = {1: 0.810, 3: 0.783, 5: 0.775, 10: 0.767}
    for k in HOR:
        auc, base = res[k]
        s = "single-class" if np.isnan(auc) else f"{auc:.3f}"
        print(f"  {k:>3} | {s:>17} | {100*base:>5.1f}% | RAGTruth {rid[k]:.3f} / zeroshot {zs[k]:.3f}")
    a3 = res[3][0]
    verdict = ("signal EXISTS -> transfer-limited (domain-specific)" if (not np.isnan(a3) and a3 >= 0.60)
               else "onset not forecastable in PsiloQA (signal absent)")
    print(f"\nGATE-A3-refit: PsiloQA in-domain AUROC@3 = {a3:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
