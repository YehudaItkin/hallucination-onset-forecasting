"""Within-document forecastability: isolate WHEN from WHICH-document.

The pooled AUROC@k is inflated by a between-document component -- an ORACLE
per-document constant (perfect "does this doc hallucinate?" label) scores ~0.82,
above the causal forecaster's ~0.77, because most positives live in hallucination
docs. That between-doc signal is NOT the paper's claim; the claim is timing (WHEN
within a document onset is imminent).

The clean test: restrict to each hallucination document's own risk set and compute
AUROC of the hazard ranking pre-onset-window tokens (y^k=1) above earlier faithful
tokens (y^k=0) WITHIN that document, then aggregate over documents. By construction
any per-document-constant predictor (incl. the oracle doc label) scores exactly 0.5
here, so anything the forecaster gets above 0.5 is timing signal no prompt-level
model can provide. A position-only within-doc baseline controls for "later tokens
are simply closer to onset."

GATE: mean within-document AUROC@3 > 0.5 (bootstrap CI over docs excludes 0.5) and
exceeds the position-only within-doc baseline.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import hazard_data as H
from run_hazard_a2 import to_sequences, standardize, train_eval, HOR
from run_hazard_a3_leadtime import forecaster_paths

KS = [1, 3, 5, 10]


def within_doc_auc(scores_by_doc, labels_by_doc):
    """per-doc AUROC where both classes present; return array over docs."""
    aucs = []
    for gi in scores_by_doc:
        y = labels_by_doc[gi]; s = scores_by_doc[gi]
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, s))
    return np.array(aucs)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    ms = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(ms, 2.5), np.percentile(ms, 97.5)


def main():
    print("[within-doc] training forecaster (causal-24, seed 0)...", flush=True)
    idx = H.CAUSAL_IDX
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    tr_all = to_sequences(trd); te = to_sequences(ted)
    rng = np.random.RandomState(0); rng.shuffle(tr_all)
    nv = int(0.15 * len(tr_all)); va, tr = tr_all[:nv], tr_all[nv:]
    rmask = trd["risk"].astype(bool); Xr = trd["X"][rmask][:, idx]
    mean, std = Xr.mean(0), Xr.std(0) + 1e-6
    standardize(tr, mean, std, idx); standardize(va, mean, std, idx); standardize(te, mean, std, idx)
    pw = [((trd[f"y{k}"][rmask] == 0).sum() / max((trd[f"y{k}"][rmask] == 1).sum(), 1)) for k in HOR]
    model, _ = train_eval(tr, va, te, idx, pw, seed=0)

    # per-doc position (within risk set) baseline
    te_by_gi = {s["gi"]: s for s in te}

    print(f"\n  {'k':>3} | {'within-doc AUROC (forecaster)':>30} | {'position-only':>13} | {'ndoc':>5}")
    print("-" * 66)
    for k in KS:
        fpaths = forecaster_paths(model, te, mean, std, idx, HOR.index(k))
        fsc, psc, lab = {}, {}, {}
        for gi, s in te_by_gi.items():
            r = s["risk"].astype(bool)
            if r.sum() < 2:
                continue
            y = s["Y"][:, HOR.index(k)][r]
            fsc[gi] = fpaths[gi][r]
            psc[gi] = np.arange(len(s["risk"]))[r].astype(np.float64)  # position within doc
            lab[gi] = y
        fa = within_doc_auc(fsc, lab); pa = within_doc_auc(psc, lab)
        lo, hi = boot_ci(fa)
        print(f"  {k:>3} | {fa.mean():.3f}  [95% {lo:.3f},{hi:.3f}]      | {pa.mean():>13.3f} | {len(fa):>5}")
    print("\nNote: any per-document-constant predictor (incl. the ORACLE doc label) = 0.5 "
          "within-doc by construction. Forecaster > 0.5 = timing signal no prompt-level model has.")


if __name__ == "__main__":
    main()
