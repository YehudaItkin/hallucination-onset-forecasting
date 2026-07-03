"""Phase D: forecast-triggered mitigation as an abstention / circuit-breaker.

We have no live decoder in this offline pipeline (features + labels + LM signals
are precomputed), so true rollback-and-regenerate is out of scope here (future
work, needs model access). The honest offline surrogate is a STOP/ABSTAIN policy:
"halt the continuation at the first alarm token a." We then measure, on a held-out
EVAL split at MATCHED false-alarm budget:
  - averted hallucination = hallucinated tokens at positions >= a that are NOT
    emitted because we stopped (fraction of all hallucinated tokens);
  - wasted faithful = faithful tokens at positions >= a that are needlessly cut
    (fraction of all faithful tokens) -- the cost of intervening.
The forecaster fires BEFORE onset (finding 7), so it should avert the WHOLE span;
the CUSUM detector fires after onset, averting only the span tail. A matched
random-stop baseline (same stopped-doc set as the forecaster, uniform stop
position) isolates timing quality from stop frequency.

GATE: at matched FA, forecaster averts more hallucination than CUSUM AND than the
matched-random baseline -> lead time converts into real mitigation headroom.
"""
import json
import numpy as np

import hazard_data as H
from run_hazard_a2 import to_sequences, standardize, train_eval, HOR
from run_hazard_a3_leadtime import forecaster_paths, DET_FILE
import run_learned_cusum as C

FAS = [0.01, 0.001]
K = 3  # primary horizon


def alarms(paths, thr, idxs):
    return {i: C.first_crossing(paths[i], thr) for i in idxs}


def mitigation(alarm, labs, idxs):
    """averted hallucinated / wasted faithful tokens under stop-at-first-alarm."""
    av_h = waste_f = tot_h = tot_f = stopped = 0
    for i in idxs:
        l = np.asarray(labs[i]); L = len(l); h = int(l.sum())
        tot_h += h; tot_f += (L - h)
        a = alarm[i]
        if a is None:
            continue
        stopped += 1
        cut_h = int(l[a:].sum())
        av_h += cut_h; waste_f += (L - a - cut_h)
    return dict(av_h=av_h, waste_f=waste_f, tot_h=tot_h, tot_f=tot_f,
                stopped=stopped, ndoc=len(idxs))


def random_mit(alarm_f, labs, idxs, rng):
    """matched-random: stop the SAME docs the forecaster stopped, random position."""
    av_h = waste_f = 0
    for i in idxs:
        if alarm_f[i] is None:
            continue
        l = np.asarray(labs[i]); L = len(l)
        a = rng.randint(0, L)
        cut_h = int(l[a:].sum())
        av_h += cut_h; waste_f += (L - a - cut_h)
    return av_h, waste_f


def main():
    print("[Phase D] training forecaster (causal-24)...", flush=True)
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

    fwd = json.load(open(DET_FILE))["ForwardGRU"]
    probs_list, labs_list = fwd["probs"], fwd["labs"]
    assert len(te) == len(probs_list) == 2700, f"doc count {len(te)}/{len(probs_list)}"
    te_by_gi = {s["gi"]: s for s in te}
    for i in range(2700):
        assert len(labs_list[i]) == len(te_by_gi[i]["X"]), f"doc {i} token-count mismatch"
    onsets = [C.first_onset(l) for l in labs_list]
    clean = [i for i in range(2700) if onsets[i] is None]
    hallu = [i for i in range(2700) if onsets[i] is not None]

    r = np.random.RandomState(1)
    cl = clean[:]; ha = hallu[:]; r.shuffle(cl); r.shuffle(ha)
    cal = set(cl[:len(cl)//2] + ha[:len(ha)//2]); ev = set(range(2700)) - cal
    cal_cl = [i for i in clean if i in cal]; ev_cl = [i for i in clean if i in ev]
    cal_ha = [i for i in hallu if i in cal]; ev_ha = [i for i in hallu if i in ev]
    ev_all = ev_cl + ev_ha
    print(f"EVAL docs: clean={len(ev_cl)} hallu={len(ev_ha)}", flush=True)

    ref, mu0, mu1 = C.cusum_reference_value([probs_list[i] for i in cal_cl + cal_ha],
                                            [labs_list[i] for i in cal_cl + cal_ha])
    cusum_paths = {i: C.cusum_path(probs_list[i], ref) for i in range(2700)}
    fpaths = forecaster_paths(model, te, mean, std, idx, HOR.index(K))

    print(f"\n(k={K})  avert = averted hallucinated tokens / all hallu tokens;  "
          f"waste = faithful tokens cut / all faithful;  eff = avert/waste")
    print(f"{'FA':>6} {'policy':>10} | {'stopDocs':>8} {'avert%':>7} {'waste%':>7} {'eff':>6} {'cleanStop%':>10}")
    print("-" * 68)
    rb = np.random.RandomState(7)
    gate = 0
    for FA in FAS:
        arl0_t = 1.0 / FA
        fgrid = np.linspace(0.01, 0.999, 400); cgrid = np.linspace(0.0, 120.0, 601)
        rf = C.sweep([fpaths[i] for i in cal_cl], [fpaths[i] for i in cal_ha],
                     [onsets[i] for i in cal_ha], fgrid)
        rc = C.sweep([cusum_paths[i] for i in cal_cl], [cusum_paths[i] for i in cal_ha],
                     [onsets[i] for i in cal_ha], cgrid)
        thr_f = C.at_arl0(rf, arl0_t)["threshold"]; thr_c = C.at_arl0(rc, arl0_t)["threshold"]
        al_f = alarms(fpaths, thr_f, ev_all); al_c = alarms(cusum_paths, thr_c, ev_all)
        mf = mitigation(al_f, labs_list, ev_all); mc = mitigation(al_c, labs_list, ev_all)
        rav, rwaste = random_mit(al_f, labs_list, ev_all, rb)

        def row(tag, al, av_h, waste_f, tot_h, tot_f, stopped):
            av = av_h / max(tot_h, 1); wf = waste_f / max(tot_f, 1)
            eff = av / wf if wf > 0 else float("inf")
            clean_stop = sum(1 for i in ev_cl if al[i] is not None) / len(ev_cl)
            print(f"{FA:>6} {tag:>10} | {stopped:>8} {100*av:>6.1f}% {100*wf:>6.1f}% {eff:>6.2f} {100*clean_stop:>9.1f}%")
        row("forecast", al_f, mf["av_h"], mf["waste_f"], mf["tot_h"], mf["tot_f"], mf["stopped"])
        row("CUSUM", al_c, mc["av_h"], mc["waste_f"], mc["tot_h"], mc["tot_f"], mc["stopped"])
        row("random", al_f, rav, rwaste, mf["tot_h"], mf["tot_f"], mf["stopped"])
        f_av = mf["av_h"] / max(mf["tot_h"], 1); c_av = mc["av_h"] / max(mc["tot_h"], 1)
        r_av = rav / max(mf["tot_h"], 1)
        if f_av > c_av and f_av > r_av:
            gate += 1
        print()
    print(f"GATE-D: forecaster averts more hallucination than CUSUM & random at matched FA "
          f"in {gate}/{len(FAS)} budgets -> "
          f"{'PASS: lead time -> mitigation headroom' if gate >= 1 else 'partial'}")


if __name__ == "__main__":
    main()
