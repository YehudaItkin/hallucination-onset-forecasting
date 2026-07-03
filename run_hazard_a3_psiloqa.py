"""A3 (ii): PsiloQA zero-shot transfer of the causal hazard forecaster.

Train the A2 causal ForwardGRU-hazard on RAGTruth (causal-24), then evaluate
hazard AUROC@k on PsiloQA-English with NO refit: same weights, same RAGTruth-train
standardization (mean/std over train risk tokens) AND same RAGTruth-train
lm_medians used to build the LM features — otherwise it is not zero-shot.
PsiloQA is ~96% positive at the example level, so the faithful risk set is small;
report AUROC (prevalence-robust) + base rate, guarded against single-class horizons.

GATE: zero-shot AUROC@3 on PsiloQA >= 0.60 (A1 feasibility bar) OR within 0.10 of
in-domain 0.783. AUROC ~0.5 => forecasting signal is RAGTruth-specific (reported
as an honest transfer-limit negative, not hidden).
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import hazard_data as H
from run_hazard_a2 import (to_sequences, standardize, train_eval, evaluate,
                           batches, HOR)

IDX = H.CAUSAL_IDX


def guarded_eval(model, seqs):
    """AUROC@k / PR@k over the risk set, tolerant of single-class horizons."""
    model.eval()
    ys = {k: [] for k in HOR}; ps = {k: [] for k in HOR}
    with torch.no_grad():
        for X, Y, M in batches(seqs, 64, False):
            prob = torch.sigmoid(model(X)).cpu().numpy()
            Yc, Mc = Y.cpu().numpy(), M.cpu().numpy().astype(bool)
            for hi, k in enumerate(HOR):
                ys[k].append(Yc[..., hi][Mc]); ps[k].append(prob[..., hi][Mc])
    out = {}
    for k in HOR:
        y = np.concatenate(ys[k]); p = np.concatenate(ps[k])
        base = y.mean()
        if y.min() == y.max():
            out[k] = (float("nan"), float("nan"), base, len(y))
        else:
            out[k] = (roc_auc_score(y, p), average_precision_score(y, p), base, len(y))
    return out


def main():
    print("[A3-psiloqa] training RAGTruth forecaster (causal-24)...", flush=True)
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    tr_all = to_sequences(trd); te = to_sequences(ted)
    rng = np.random.RandomState(0); rng.shuffle(tr_all)
    nv = int(0.15 * len(tr_all)); va, tr = tr_all[:nv], tr_all[nv:]
    rmask = trd["risk"].astype(bool)
    Xr = trd["X"][rmask][:, IDX]
    mean, std = Xr.mean(0), Xr.std(0) + 1e-6
    standardize(tr, mean, std, IDX); standardize(va, mean, std, IDX); standardize(te, mean, std, IDX)
    pw = [((trd[f"y{k}"][rmask] == 0).sum() / max((trd[f"y{k}"][rmask] == 1).sum(), 1)) for k in HOR]

    model, rag = train_eval(tr, va, te, IDX, pw)   # trained model + RAGTruth-test metrics

    # PsiloQA zero-shot: RAGTruth-train lm_medians (med) + RAGTruth-train mean/std
    print("building PsiloQA (RAGTruth-train lm_medians -> zero-shot)...", flush=True)
    pq, _ = H.build("psiloqa_test", lm_medians=med)
    pq_seqs = to_sequences(pq)
    standardize(pq_seqs, mean, std, IDX)
    pr = guarded_eval(model, pq_seqs)

    print(f"\n  {'k':>3} | {'RAGTruth in-domain':>18} | {'PsiloQA zero-shot':>18} | {'PsiloQA base%':>13} | {'risk n':>8}")
    print("-" * 74)
    for k in HOR:
        rid = f"{rag[f'auc@{k}']:.3f}/{rag[f'ap@{k}']:.3f}"
        auc, appr, base, n = pr[k]
        pqs = "single-class" if np.isnan(auc) else f"{auc:.3f}/{appr:.3f}"
        print(f"  {k:>3} | {rid:>18} | {pqs:>18} | {100*base:>12.2f}% | {n:>8}")
    a3 = pr[3][0]
    print(f"\nGATE-A3-psiloqa: zero-shot AUROC@3 = {a3:.3f}  -> "
          f"{'PASS' if (not np.isnan(a3) and (a3 >= 0.60 or a3 >= 0.683)) else 'transfer-limited (honest negative)'}")


if __name__ == "__main__":
    main()
