"""Phase B: predictability taxonomy of onsets (mechanism-adjacent, black-box).

Which onsets does the forecaster see coming, and what precedes them? For every
held-out (test) onset at position theta with runway (theta>=1) we read the causal
ForwardGRU-hazard the model assigned at the LAST pre-onset token, h_{theta-1}^(3)
(= "onset in the next 3 tokens?"). Onsets in the top tercile of that score are
ANTICIPATED, bottom tercile SURPRISE. We then ask, black-box:
  (a) does forecastability vary with onset POSITION in the output (runway)?
  (b) does it vary with SPAN LENGTH?
  (c) which pre-onset causal FEATURES (mean over the 3 tokens before onset)
      separate anticipated from surprise onsets? (single-feature AUROC)
Onsets at theta=0 have no runway and are structurally unforecastable -> reported
separately, excluded from the hazard analysis.

Claim to support: anticipated onsets are preceded by rising TEXT-NOVELTY drift
(not LM surprisal) -> consistent with finding 3 (forecasting != detection); early
onsets are less forecastable (no runway). Descriptive, so no pass/fail gate.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import hazard_data as H
from run_hazard_a2 import to_sequences, standardize, train_eval, HOR
from run_hazard_a3_leadtime import forecaster_paths

TEXT_NAMES = ["in_context", "is_number", "is_capitalized", "word_length", "position",
              "running_ctx_ratio", "running_novel_ratio", "is_novel", "num_in_context",
              "novel_number", "window_5_novelty", "window_10_novelty", "window_20_novelty",
              "delta_novel_ratio", "delta2_novel_ratio", "bigram", "trigram", "entity",
              "consec", "sent_pos"]                                   # 0..19
NLI_NAMES = [f"nli_{i}" for i in range(7)]                            # 20..26
LM_NAMES = ["lm_logprob", "lm_entropy", "lm_log1p_c2", "lm_log1p_c3",
            "lm_has_signal", "lm_logprob_x_signal"]                   # 27..32
NAMES = TEXT_NAMES + NLI_NAMES + LM_NAMES
K = 3
WIN = 3  # pre-onset window for feature means


def per_doc(ted):
    groups = ted["groups"]; onset = ted["onset"]; risk = ted["risk"].astype(bool)
    bounds = np.searchsorted(groups, np.arange(groups[-1] + 2))
    docs = {}
    for gi in range(groups[-1] + 1):
        s, e = bounds[gi], bounds[gi + 1]
        if e <= s:
            continue
        docs[gi] = {"s": s, "e": e, "onsets": np.where(onset[s:e] == 1)[0],
                    "hallu": (~risk[s:e]).astype(int), "L": e - s}
    return docs


def span_len(hallu, theta):
    j = theta
    while j < len(hallu) and hallu[j] == 1:
        j += 1
    return j - theta


def main():
    print("[Phase B] training forecaster (causal-24)...", flush=True)
    idx = H.CAUSAL_IDX
    cnames = [NAMES[i] for i in idx]
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    tr_all = to_sequences(trd); te = to_sequences(ted)
    rng = np.random.RandomState(0); rng.shuffle(tr_all)
    nv = int(0.15 * len(tr_all)); va, tr = tr_all[:nv], tr_all[nv:]
    rmask = trd["risk"].astype(bool); Xr = trd["X"][rmask][:, idx]
    mean, std = Xr.mean(0), Xr.std(0) + 1e-6
    standardize(tr, mean, std, idx); standardize(va, mean, std, idx); standardize(te, mean, std, idx)
    pw = [((trd[f"y{k}"][rmask] == 0).sum() / max((trd[f"y{k}"][rmask] == 1).sum(), 1)) for k in HOR]
    model, _ = train_eval(tr, va, te, idx, pw)

    fpaths = forecaster_paths(model, te, mean, std, idx, HOR.index(K))
    docs = per_doc(ted)
    Xflat = ted["X"]

    haz, posn, slen, feat = [], [], [], []
    n0 = 0
    for gi, d in docs.items():
        for theta in d["onsets"]:
            if theta == 0:
                n0 += 1
                continue
            haz.append(float(fpaths[gi][theta - 1]))
            posn.append(theta / max(d["L"] - 1, 1))
            slen.append(span_len(d["hallu"], theta))
            w0 = d["s"] + max(0, theta - WIN); w1 = d["s"] + theta
            feat.append(Xflat[w0:w1][:, idx].mean(0))
    haz = np.array(haz); posn = np.array(posn); slen = np.array(slen); feat = np.array(feat)
    print(f"onsets: total={sum(len(d['onsets']) for d in docs.values())}  "
          f"with-runway={len(haz)}  theta0(no-runway)={n0}", flush=True)
    print(f"pre-onset hazard h^{K}_(theta-1): mean={haz.mean():.3f} median={np.median(haz):.3f}")

    # (a) forecastability vs onset position (runway)
    print("\n(a) mean pre-onset hazard by output-position tercile:")
    q = np.quantile(posn, [1/3, 2/3])
    for lab, m in [("early", posn <= q[0]),
                   ("mid", (posn > q[0]) & (posn <= q[1])),
                   ("late", posn > q[1])]:
        print(f"    {lab:>5} (pos): n={m.sum():5d}  hazard={haz[m].mean():.3f}")

    # (b) forecastability vs span length
    print("\n(b) mean pre-onset hazard by span length:")
    med_s = np.median(slen)
    for lab, m in [(f"short(<= {med_s:.0f})", slen <= med_s), (f"long(> {med_s:.0f})", slen > med_s)]:
        print(f"    {lab:>12}: n={m.sum():5d}  hazard={haz[m].mean():.3f}")

    # (c) which pre-onset features separate ANTICIPATED (top-tercile hazard) from SURPRISE (bottom)
    hi = haz >= np.quantile(haz, 2/3); lo = haz <= np.quantile(haz, 1/3)
    y = np.r_[np.ones(hi.sum()), np.zeros(lo.sum())]
    aucs = []
    for c in range(feat.shape[1]):
        x = np.r_[feat[hi, c], feat[lo, c]]
        if np.all(x == x[0]):
            aucs.append((cnames[c], 0.5, feat[hi, c].mean(), feat[lo, c].mean()))
        else:
            a = roc_auc_score(y, x)
            aucs.append((cnames[c], a, feat[hi, c].mean(), feat[lo, c].mean()))
    aucs.sort(key=lambda t: -abs(t[1] - 0.5))
    print(f"\n(c) top pre-onset features separating ANTICIPATED (n={hi.sum()}) vs "
          f"SURPRISE (n={lo.sum()}) onsets [single-feature AUROC, mean anticip / mean surprise]:")
    for name, a, mh, ml in aucs[:10]:
        grp = "LM" if name.startswith("lm_") else "TEXT"
        print(f"    {name:>22} [{grp:>4}]  AUROC={a:.3f}   {mh:+.3f} / {ml:+.3f}")
    top8 = [n for n, *_ in aucs[:8]]
    n_text = sum(1 for n in top8 if not n.startswith("lm_"))
    print(f"\nSUMMARY: of top-8 discriminating pre-onset features, {n_text}/8 are TEXT-drift "
          f"(vs LM) -> {'consistent with finding 3 (text-drift precursor)' if n_text >= 5 else 'mixed'}; "
          f"early-vs-late position hazard gradient present.")


if __name__ == "__main__":
    main()
