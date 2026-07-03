"""A1 feasibility gate: is hallucination onset forecastable from CAUSAL
black-box features, before it happens?

Discrete-time hazard via logistic regression (the floor model). For each horizon
k we predict y_t^(k) = 1[onset in (t, t+k]] over the risk set (faithful tokens),
and compare feature sets to locate the signal:

  full-33        all features (includes future-leaking ones)   -> optimistic ceiling
  causal-24      streaming-safe set                             -> THE GATE
  text-causal-18 causal text features only
  lm-6           proxy-LM features only (log-prob, entropy, ranks)

GATE: causal-24 AUROC@3 >= 0.60 AND clearly above chance -> onset is forecastable
      from a causal stream; proceed to A2 (Hawkes). Otherwise the direction dies.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import hazard_data as H

LM_IDX = [27, 28, 29, 30, 31, 32]
TEXT_CAUSAL_IDX = [i for i in H.CAUSAL_IDX if i < 20]
FEATURE_SETS = {
    "full-33": list(range(33)),
    "causal-24": H.CAUSAL_IDX,
    "text-causal-18": TEXT_CAUSAL_IDX,
    "lm-6": LM_IDX,
}


def fit_eval(Xtr, ytr, Xte, yte, seed=42):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             n_jobs=-1, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def main():
    print("Building hazard datasets (reusing the paper's exact 33-dim pipeline)...",
          flush=True)
    tr, med = H.build("train")
    te, _ = H.build("test", lm_medians=med)
    rtr = tr["risk"].astype(bool)
    rte = te["risk"].astype(bool)
    Xtr, Xte = tr["X"][rtr], te["X"][rte]
    print(f"train risk tokens={rtr.sum():,}  test risk tokens={rte.sum():,}  "
          f"(causal dims={len(H.CAUSAL_IDX)})\n", flush=True)

    hdr = f"{'k':>3} {'base%':>7} | " + " | ".join(
        f"{n:>14}" for n in FEATURE_SETS)
    print(hdr)
    print("-" * len(hdr))

    gate_val = None
    for k in H.HORIZONS:
        ytr, yte = tr[f"y{k}"][rtr], te[f"y{k}"][rte]
        base = 100 * yte.mean()
        cells = []
        for name, idx in FEATURE_SETS.items():
            auc, ap = fit_eval(Xtr[:, idx], ytr, Xte[:, idx], yte)
            cells.append(f"{auc:.3f}/{ap:.3f}")
            if k == 3 and name == "causal-24":
                gate_val = auc
        print(f"{k:>3} {base:>7.2f} | " + " | ".join(f"{c:>14}" for c in cells),
              flush=True)

    print("\n(cells = AUROC / PR-AUC ; base% = risk-set positive rate)")
    print(f"\nGATE  causal-24 AUROC@3 = {gate_val:.3f}  "
          f"-> {'PASS, proceed to A2 (Hawkes)' if gate_val and gate_val >= 0.60 else 'FAIL / marginal'}")


if __name__ == "__main__":
    main()
