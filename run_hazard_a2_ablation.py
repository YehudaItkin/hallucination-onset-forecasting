"""A2 rigor pass: multi-seed CIs + GRU feature dissociation + static baselines.

Closes three gaps in the single-seed A2:
  1. Multi-seed CIs on the headline forecaster (5 seeds) -- the 0.78 must not rest
     on one seed.
  2. GRU feature dissociation: causal-24 vs text-causal-18 vs lm-6 vs full-33, all
     with the recurrent model. A1 showed text >> lm only for the LINEAR logistic;
     this establishes the dissociation for the model the paper actually uses.
  3. Static baselines for RQ1 ("beats a static prompt-level predictor"): a
     position-only score and an ORACLE per-document constant (perfect doc-level
     hallucination label broadcast to every token). If the token-level forecaster
     beats an oracle doc-level label, within-document timing carries real signal.

Same paired train/val/test split across all feature sets and seeds.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import hazard_data as H
from run_hazard_a2 import to_sequences, standardize, train_eval, HOR

SEEDS = [0, 1, 2, 3, 4]
TEXT18 = [i for i in range(20) if i not in (4, 19)]   # causal text features
LM6 = list(range(27, 33))                              # LM block
FSETS = {"causal-24": H.CAUSAL_IDX, "text-18": TEXT18, "lm-6": LM6, "full-33": list(range(33))}


def static_baselines(ted):
    """position-only and oracle doc-constant AUROC@k on the test risk set."""
    groups = ted["groups"]; pos = ted["pos"]; onset = ted["onset"]
    bounds = np.searchsorted(groups, np.arange(groups[-1] + 2))
    posnorm = np.zeros(len(groups), dtype=np.float64)
    dochas = np.zeros(len(groups), dtype=np.float64)
    for gi in range(groups[-1] + 1):
        s, e = bounds[gi], bounds[gi + 1]
        if e <= s:
            continue
        posnorm[s:e] = pos[s:e] / max(e - s - 1, 1)
        dochas[s:e] = 1.0 if onset[s:e].sum() > 0 else 0.0
    r = ted["risk"].astype(bool)
    out = {}
    for k in HOR:
        y = ted[f"y{k}"][r]
        out[k] = (roc_auc_score(y, posnorm[r]), roc_auc_score(y, dochas[r]))
    return out


def main():
    print("[A2-ablation] building data...", flush=True)
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    tr_all = to_sequences(trd); te = to_sequences(ted)
    rng = np.random.RandomState(0); rng.shuffle(tr_all)
    nv = int(0.15 * len(tr_all)); va, tr = tr_all[:nv], tr_all[nv:]
    rmask = trd["risk"].astype(bool)

    # ---- static baselines (no training) ----
    base = static_baselines(ted)
    print(f"\nStatic baselines (AUROC@k on test risk set):")
    print(f"  {'k':>3} | {'position-only':>13} | {'oracle doc-constant':>19}")
    for k in HOR:
        print(f"  {k:>3} | {base[k][0]:>13.3f} | {base[k][1]:>19.3f}")

    # ---- GRU feature sets x seeds ----
    results = {name: {k: [] for k in HOR} for name in FSETS}
    for name, idx in FSETS.items():
        Xr = trd["X"][rmask][:, idx]
        mean, std = Xr.mean(0), Xr.std(0) + 1e-6
        standardize(tr, mean, std, idx); standardize(va, mean, std, idx); standardize(te, mean, std, idx)
        pw = [((trd[f"y{k}"][rmask] == 0).sum() / max((trd[f"y{k}"][rmask] == 1).sum(), 1)) for k in HOR]
        for sd in SEEDS:
            _, m = train_eval(tr, va, te, idx, pw, seed=sd)
            for k in HOR:
                results[name][k].append(m[f"auc@{k}"])
            print(f"  {name:>9} seed {sd}: " +
                  " ".join(f"@{k}={m[f'auc@{k}']:.3f}" for k in HOR), flush=True)

    print(f"\n=== GRU AUROC@k, mean ± std over {len(SEEDS)} seeds ===")
    header = "  {:>9} |".format("fset") + "".join(f" {'@'+str(k):>13} |" for k in HOR)
    print(header); print("-" * len(header))
    for name in FSETS:
        row = f"  {name:>9} |"
        for k in HOR:
            v = np.array(results[name][k]); row += f" {v.mean():.3f}±{v.std():.3f} |"
        print(row)
    c = {k: np.mean(results["causal-24"][k]) for k in HOR}
    t = {k: np.mean(results["text-18"][k]) for k in HOR}
    l = {k: np.mean(results["lm-6"][k]) for k in HOR}
    print(f"\nDISSOCIATION (GRU, mean@3): text-18={t[3]:.3f}  lm-6={l[3]:.3f}  causal-24={c[3]:.3f} "
          f"-> {'text >> lm confirmed for the recurrent model' if t[3] - l[3] > 0.02 else 'weak'}")
    print(f"RQ1 baseline: causal-24@3={c[3]:.3f} vs oracle doc-constant {base[3][1]:.3f} "
          f"-> {'token-level timing beats prompt-level' if c[3] > base[3][1] else 'no gain over doc-level'}")


if __name__ == "__main__":
    main()
