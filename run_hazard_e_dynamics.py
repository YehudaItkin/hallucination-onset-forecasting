"""Phase E: what the forecaster's hazard does around an onset, and where it holds.

Three analyses off one trained causal-24 forecaster, answering the three questions a
reader of Sections 5.1-5.3 is left with.

E1  Onset-aligned dynamics. The paper claims the hazard rises BEFORE the onset because
    novelty drifts up first. That claim is currently supported only by aggregate
    pre-onset hazards. Here we align every held-out onset at d=0 and average the
    hazard (and the top novelty features) over d in [-30, +5], with a
    document-clustered bootstrap. A rise toward d=0 could also be a position effect,
    so we carry a CONTROL curve: the same absolute positions read from clean
    documents, which have no onset. The gap between the two curves is the precursor.

E2  Strata. RAGTruth spans three task types and six generators. A precursor that only
    exists for one of them is a different (and weaker) finding than one that holds
    across them. Pooled and within-document AUROC@3 per stratum.

E3  Calibration. The alarm thresholds a hazard "posterior", so whether that posterior
    is calibrated matters: reliability over ten predicted-hazard bins, plus ECE and
    the Brier score.

Emits ready-to-paste TikZ coordinates for the E1 figure, so the plotted curve cannot
drift from the measured one.

    tmux new -s dynamics
    python run_hazard_e_dynamics.py 2>&1 | tee results/e_dynamics.log
"""
import collections

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import hazard_data as H
import run_extended as R
from run_hazard_a2 import to_sequences, standardize, train_eval, HOR
from run_hazard_a3_leadtime import forecaster_paths
from run_hazard_b_taxonomy import NAMES

K = 3                       # horizon used throughout the paper's headline numbers
LO, HI = -30, 5             # onset-aligned window, d = t - theta
MIN_RUNWAY = 10             # onsets with less runway cannot fill the left half
BOOT = 1000
NOVELTY = ["window_5_novelty", "window_10_novelty", "window_20_novelty"]


def doc_bounds(d):
    g = d["groups"]
    return np.searchsorted(g, np.arange(g[-1] + 2))


def aligned(paths, bounds, gis, onsets_by_doc):
    """Matrix (n_onsets x window) of hazard values aligned at the onset."""
    W = HI - LO + 1
    rows, owner = [], []
    for gi in gis:
        p = paths[gi]
        L = len(p)
        for th in onsets_by_doc[gi]:
            if th < MIN_RUNWAY:
                continue
            r = np.full(W, np.nan)
            for j, d in enumerate(range(LO, HI + 1)):
                t = th + d
                if 0 <= t < L:
                    r[j] = p[t]
            rows.append(r)
            owner.append(gi)
    return np.array(rows), np.array(owner)


def control(paths, clean_gis, positions, rng):
    """Same absolute positions, read from clean documents: the no-onset counterfactual."""
    W = HI - LO + 1
    rows = []
    for th in positions:
        for _ in range(1):
            gi = clean_gis[rng.randint(len(clean_gis))]
            p = paths[gi]
            if len(p) <= th:
                continue
            r = np.full(W, np.nan)
            for j, d in enumerate(range(LO, HI + 1)):
                t = th + d
                if 0 <= t < len(p):
                    r[j] = p[t]
            rows.append(r)
    return np.array(rows)


def clustered_ci(rows, owner, rng, boot=BOOT):
    """Mean curve with a document-clustered bootstrap CI."""
    mean = np.nanmean(rows, axis=0)
    docs = np.unique(owner)
    idx_by_doc = {g: np.where(owner == g)[0] for g in docs}
    draws = np.empty((boot, rows.shape[1]))
    for b in range(boot):
        pick = docs[rng.randint(len(docs), size=len(docs))]
        sel = np.concatenate([idx_by_doc[g] for g in pick])
        draws[b] = np.nanmean(rows[sel], axis=0)
    return mean, np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def tikz(name, xs, ys):
    pts = " ".join(f"({x},{y:.4f})" for x, y in zip(xs, ys))
    print(f"% {name}\n\\addplot coordinates {{{pts}}};")


def auroc_subset(paths, bounds, gis, ted, within=False):
    """Pooled or within-document AUROC@K over the risk set of the given documents."""
    ys, ps, gs = [], [], []
    riskm = ted["risk"].astype(bool)
    yk = ted[f"y{K}"]
    for gi in gis:
        s, e = bounds[gi], bounds[gi + 1]
        m = riskm[s:e]
        if m.sum() == 0:
            continue
        ys.append(yk[s:e][m]); ps.append(paths[gi][m]); gs.append(np.full(m.sum(), gi))
    if not ys:
        return float("nan"), 0
    y = np.concatenate(ys); p = np.concatenate(ps); g = np.concatenate(gs)
    if not within:
        if y.min() == y.max():
            return float("nan"), len(y)
        return roc_auc_score(y, p), len(y)
    # within-document: average per-document AUROC over documents that have both classes
    aucs = []
    for gi in np.unique(g):
        m = g == gi
        if y[m].min() == y[m].max():
            continue
        aucs.append(roc_auc_score(y[m], p[m]))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def main():
    rng = np.random.RandomState(0)
    print("[Phase E] training forecaster (causal-24)...", flush=True)
    idx = H.CAUSAL_IDX
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    tr_all = to_sequences(trd); te = to_sequences(ted)
    r0 = np.random.RandomState(0); r0.shuffle(tr_all)
    nv = int(0.15 * len(tr_all)); va, tr = tr_all[:nv], tr_all[nv:]
    rmask = trd["risk"].astype(bool); Xr = trd["X"][rmask][:, idx]
    mean, std = Xr.mean(0), Xr.std(0) + 1e-6
    standardize(tr, mean, std, idx); standardize(va, mean, std, idx); standardize(te, mean, std, idx)
    pw = [((trd[f"y{k}"][rmask] == 0).sum() / max((trd[f"y{k}"][rmask] == 1).sum(), 1)) for k in HOR]
    model, _ = train_eval(tr, va, te, idx, pw)
    paths = forecaster_paths(model, te, mean, std, idx, HOR.index(K))

    bounds = doc_bounds(ted)
    onset = ted["onset"]
    ndocs = int(ted["groups"][-1]) + 1
    onsets_by_doc, hallu_gis, clean_gis = {}, [], []
    for gi in range(ndocs):
        s, e = bounds[gi], bounds[gi + 1]
        if e <= s:
            continue
        o = np.where(onset[s:e] == 1)[0]
        onsets_by_doc[gi] = o
        (hallu_gis if len(o) else clean_gis).append(gi)
    print(f"documents: {len(hallu_gis)} hallucinating, {len(clean_gis)} clean")

    # ---------------- E1: onset-aligned dynamics ----------------
    rows, owner = aligned(paths, bounds, hallu_gis, onsets_by_doc)
    xs = list(range(LO, HI + 1))
    m, lo, hi = clustered_ci(rows, owner, rng)
    first = [th for gi in hallu_gis for th in onsets_by_doc[gi] if th >= MIN_RUNWAY]
    crows = control(paths, clean_gis, first, rng)
    cm = np.nanmean(crows, axis=0)

    print(f"\n(E1) onset-aligned hazard, {rows.shape[0]} onsets with runway >= {MIN_RUNWAY}, "
          f"control from {crows.shape[0]} clean-document positions")
    print(f"{'d':>4} {'hazard':>8} {'95% CI':>17} {'control':>8} {'gap':>7}")
    for j, d in enumerate(xs):
        if d % 2 == 0 or d in (-1, 1):
            print(f"{d:>+4} {m[j]:>8.3f} [{lo[j]:.3f},{hi[j]:.3f}] {cm[j]:>8.3f} {m[j]-cm[j]:>+7.3f}")
    pre = [j for j, d in enumerate(xs) if d < 0]
    print(f"  pre-onset rise: h({LO}) = {m[0]:.3f} -> h(-1) = {m[pre[-1]]:.3f} "
          f"(+{m[pre[-1]]-m[0]:.3f}); control moves {cm[0]:.3f} -> {cm[pre[-1]]:.3f}")
    sep = [d for j, d in enumerate(xs) if d < 0 and lo[j] > cm[j]]
    print(f"  hazard CI clears the control curve from d = {min(sep) if sep else 'never'} onward")

    # novelty features on the same alignment (standardized, as the model sees them)
    Xs = (ted["X"][:, idx] - mean) / std
    cn = [NAMES[i] for i in idx]
    print("\n(E1b) onset-aligned novelty features (standardized units)")
    feat_curves = {}
    for fname in NOVELTY:
        col = cn.index(fname)
        fp = {gi: Xs[bounds[gi]:bounds[gi + 1], col] for gi in hallu_gis}
        frows, fowner = aligned(fp, bounds, hallu_gis, onsets_by_doc)
        fm = np.nanmean(frows, axis=0)
        feat_curves[fname] = fm
        print(f"  {fname:<20} d={LO}: {fm[0]:+.3f}   d=-5: {fm[xs.index(-5)]:+.3f}   "
              f"d=-1: {fm[xs.index(-1)]:+.3f}   rise {fm[xs.index(-1)]-fm[0]:+.3f}")

    print("\n% ---- TikZ coordinates for the onset-aligned figure ----")
    tikz("hazard (mean)", xs, m)
    tikz("hazard CI lower", xs, lo)
    tikz("hazard CI upper", xs, hi)
    tikz("clean-document control", xs, cm)
    for fname, fm in feat_curves.items():
        tikz(fname, xs, fm)

    # ---------------- E2: strata ----------------
    _b, _l, ex = R.load_base_features("test")
    by_task = collections.defaultdict(list)
    by_model = collections.defaultdict(list)
    for gi in range(ndocs):
        if gi not in onsets_by_doc:
            continue
        by_task[ex[gi]["task_type"]].append(gi)
        by_model[ex[gi]["model"]].append(gi)

    for label, groups in (("task type", by_task), ("generator", by_model)):
        print(f"\n(E2) AUROC@{K} by {label}")
        print(f"{'stratum':<24} {'docs':>5} {'pooled':>8} {'within-doc':>11} {'n docs w/':>10}")
        for key in sorted(groups):
            gis = groups[key]
            pooled, npool = auroc_subset(paths, bounds, gis, ted, within=False)
            wd, ndoc = auroc_subset(paths, bounds, gis, ted, within=True)
            print(f"{key:<24} {len(gis):>5} {pooled:>8.3f} {wd:>11.3f} {ndoc:>10}")

    # ---------------- E3: calibration ----------------
    riskm = ted["risk"].astype(bool)
    allp, ally = [], []
    for gi in range(ndocs):
        if gi not in onsets_by_doc:
            continue
        s, e = bounds[gi], bounds[gi + 1]
        mk = riskm[s:e]
        allp.append(paths[gi][mk]); ally.append(ted[f"y{K}"][s:e][mk])
    p = np.concatenate(allp); y = np.concatenate(ally).astype(float)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    print(f"\n(E3) calibration of the hazard posterior over {len(p)} risk-set tokens")
    print(f"{'bin':>4} {'n':>8} {'mean pred':>10} {'empirical':>10}")
    ece = 0.0
    for b in range(10):
        mk = (p > edges[b]) & (p <= edges[b + 1])
        if mk.sum() == 0:
            continue
        pred, emp = p[mk].mean(), y[mk].mean()
        ece += mk.mean() * abs(pred - emp)
        print(f"{b+1:>4} {mk.sum():>8} {pred:>10.3f} {emp:>10.3f}")
    print(f"  ECE = {ece:.3f}   Brier = {np.mean((p - y) ** 2):.3f}   base rate = {y.mean():.3f}")
    print("  (the head is trained with per-horizon positive weights, so the posterior is "
          "deliberately not calibrated to the base rate; ranking, not level, is what the "
          "alarm threshold uses)")


if __name__ == "__main__":
    main()
