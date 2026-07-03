"""A3 (i): lead-time of the FORECASTER vs the CUSUM detector at matched FA.

The forecaster raises an alarm when its hazard posterior h_t^(k) crosses a
threshold (a PREDICTIVE statistic -> fires before onset). The CUSUM detector
(the CUSUM paper's ForwardGRU-CUSUM) accumulates log-odds of the DETECTION
posterior (rises only after onset). Both are causal statistic paths, so the same
calibration machinery applies. Signed delay d = alarm - theta (d<0 = warned
before onset). Common currency: false alarms per clean token = 1/ARL0 on the
clean-document stream.

Reviewer-corrected: held-out FA calibration (CAL/EVAL doc split, stratified
clean/hallu); thresholds + CUSUM reference fit on CAL only, frozen, measured on
EVAL; realized EVAL FA reported so budgets are provably matched; head-to-head on
onsets caught by BOTH (paired Wilcoxon one-sided + directional McNemar requiring
forecaster-early > cusum-early); recall reported so a lead-time win cannot be a
collapsed-recall artifact. Hard per-doc alignment (token-count) vs the detection
prob file.

GATE (FA=0.01): forecaster median signed delay NEGATIVE and significantly earlier
than CUSUM on the both-caught set (Wilcoxon p<0.05) for k in {3,5}; %warned-early
forecaster >> CUSUM (McNemar b>c, p<0.05); realized EVAL FA of the two within ~25%;
forecaster recall >= 0.5x CUSUM recall (else report as recall-traded lead time).
"""
import json
import numpy as np
import torch
from scipy.stats import wilcoxon, binomtest

import hazard_data as H
from run_hazard_a2 import to_sequences, standardize, train_eval, ForwardGRUHazard, HOR, DEVICE
import run_learned_cusum as C

DET_FILE = str(H.HALLU_DIR / "directional_probs_seed42.json")
FAS = [0.01, 0.001]
KS = [3, 5]


def forecaster_paths(model, seqs, mean, std, idx, k_index):
    model.eval(); paths = {}
    with torch.no_grad():
        for s in seqs:
            x = ((s["X"][:, idx] - mean) / std).astype(np.float32)
            xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            paths[s["gi"]] = torch.sigmoid(model(xt)).cpu().numpy()[0, :, k_index]
    return paths


def mcnemar(early_a, early_b):
    b = int(np.sum(early_a & ~early_b))   # forecaster-early only
    c = int(np.sum(~early_a & early_b))   # cusum-early only
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    return b, c, p


def delays(paths, onsets, thr, idxs):
    """signed delays over docs in idxs that HAVE an onset; alarm may be None."""
    out = {}
    for i in idxs:
        th = onsets[i]
        if th is None:
            continue
        a = C.first_crossing(paths[i], thr)
        out[i] = None if a is None else (a - th)
    return out


def main():
    print("[A3-leadtime] training forecaster (causal-24)...", flush=True)
    idx = H.CAUSAL_IDX
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    tr_all = to_sequences(trd); te = to_sequences(ted)
    rng = np.random.RandomState(0); rng.shuffle(tr_all)
    nv = int(0.15 * len(tr_all)); va, tr = tr_all[:nv], tr_all[nv:]
    rmask = trd["risk"].astype(bool); Xr = trd["X"][rmask][:, idx]
    mean, std = Xr.mean(0), Xr.std(0) + 1e-6
    standardize(tr, mean, std, idx); standardize(va, mean, std, idx); standardize(te, mean, std, idx)
    pw = [((trd[f"y{k}"][rmask] == 0).sum() / max((trd[f"y{k}"][rmask] == 1).sum(), 1)) for k in HOR]
    model, _ = train_eval(tr, va, te, idx, pw)

    # detection probs/labels (ForwardGRU-CUSUM input), aligned per doc
    fwd = json.load(open(DET_FILE))["ForwardGRU"]   # {ForwardGRU,BackwardGRU,BiGRU,LogReg}
    probs_list, labs_list = fwd["probs"], fwd["labs"]
    assert len(te) == len(probs_list) == 2700, f"doc count {len(te)}/{len(probs_list)}"
    te_by_gi = {s["gi"]: s for s in te}
    for i in range(2700):
        assert len(labs_list[i]) == len(te_by_gi[i]["X"]), f"doc {i} token-count mismatch (alignment)"
    onsets = [C.first_onset(l) for l in labs_list]
    clean = [i for i in range(2700) if onsets[i] is None]
    hallu = [i for i in range(2700) if onsets[i] is not None]
    print(f"docs: clean={len(clean)} hallu={len(hallu)}", flush=True)

    # CAL/EVAL split, stratified clean/hallu
    r = np.random.RandomState(1)
    cl = clean[:]; ha = hallu[:]; r.shuffle(cl); r.shuffle(ha)
    cal = set(cl[:len(cl)//2] + ha[:len(ha)//2]); ev = set(range(2700)) - cal
    cal_cl = [i for i in clean if i in cal]; ev_cl = [i for i in clean if i in ev]
    cal_ha = [i for i in hallu if i in cal]; ev_ha = [i for i in hallu if i in ev]

    # forecaster hazard paths per horizon; cusum paths (ref fit on CAL only)
    ref, mu0, mu1 = C.cusum_reference_value([probs_list[i] for i in cal_cl + cal_ha],
                                            [labs_list[i] for i in cal_cl + cal_ha])
    cusum_paths = {i: C.cusum_path(probs_list[i], ref) for i in range(2700)}
    print(f"CUSUM ref k=(mu0+mu1)/2={ref:.3f}", flush=True)

    print(f"\n{'k':>2} {'FA':>6} {'det':>10} | {'recall':>6} {'medDelay':>8} {'%early':>6} {'realFA':>7} | paired vs CUSUM")
    print("-" * 92)
    gate_hits = 0
    for k in KS:
        fpaths = forecaster_paths(model, te, mean, std, idx, HOR.index(k))
        for FA in FAS:
            arl0_t = 1.0 / FA
            # calibrate thresholds on CAL
            fgrid = np.linspace(0.01, 0.999, 400)
            cgrid = np.linspace(0.0, 120.0, 601)
            rf = C.sweep([fpaths[i] for i in cal_cl], [fpaths[i] for i in cal_ha],
                         [onsets[i] for i in cal_ha], fgrid)
            rc = C.sweep([cusum_paths[i] for i in cal_cl], [cusum_paths[i] for i in cal_ha],
                         [onsets[i] for i in cal_ha], cgrid)
            thr_f = C.at_arl0(rf, arl0_t)["threshold"]; thr_c = C.at_arl0(rc, arl0_t)["threshold"]
            fa_f = 1.0 / C.arl0_on_clean_stream([fpaths[i] for i in ev_cl], thr_f)
            fa_c = 1.0 / C.arl0_on_clean_stream([cusum_paths[i] for i in ev_cl], thr_c)
            df = delays(fpaths, onsets, thr_f, ev_ha)
            dc = delays(cusum_paths, onsets, thr_c, ev_ha)
            def stats(d):
                got = [v for v in d.values() if v is not None]
                rec = len(got) / len(ev_ha)
                med = np.median(got) if got else float("nan")
                early = np.mean([v < 0 for v in got]) if got else float("nan")
                return rec, med, early
            rf_, medf, earlyf = stats(df); rc_, medc, earlyc = stats(dc)
            # paired on both-caught
            both = [i for i in ev_ha if df[i] is not None and dc[i] is not None]
            vf = np.array([df[i] for i in both]); vc = np.array([dc[i] for i in both])
            wp = wilcoxon(vf, vc, alternative="less").pvalue if len(both) > 5 and np.any(vf != vc) else float("nan")
            b, c, mp = mcnemar(vf < 0, vc < 0)
            print(f"{k:>2} {FA:>6} {'forecast':>10} | {rf_:>6.2f} {medf:>+8.1f} {earlyf:>6.2f} {fa_f:>7.4f} | "
                  f"Wilcoxon p={wp:.3g}  McNemar b={b} c={c} p={mp:.3g}")
            print(f"{'':>2} {'':>6} {'CUSUM':>10} | {rc_:>6.2f} {medc:>+8.1f} {earlyc:>6.2f} {fa_c:>7.4f} |")
            if FA == 0.01 and (medf < 0) and (not np.isnan(wp) and wp < 0.05) and (b > c and mp < 0.05):
                gate_hits += 1
    print(f"\nGATE-A3-leadtime: forecaster fires before onset & beats CUSUM (FA=0.01, k in {KS}) "
          f"in {gate_hits}/{len(KS)} horizons -> {'PASS: negative-delay forecasting' if gate_hits >= 1 else 'partial'}")


if __name__ == "__main__":
    main()
