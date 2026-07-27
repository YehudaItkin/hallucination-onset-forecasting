"""Theory check: the pre-onset runway E[theta] and the Proposition-1 early-warning ceiling.

Proposition 1 (uninformative-prefix ceiling). Let T be an alarm rule (a stopping
time on the prefix filtration) calibrated to a per-token false-alarm budget alpha
on the clean stream, P_inf(T <= m) <= alpha*m. If the still-faithful prefix is
uninformative about the onset, i.e. P(A | theta = m) = P_inf(A) for every event A
measurable w.r.t. the first m-1 tokens, then

    P(T < theta)  <=  alpha * E[theta - 1],

where theta - 1 is the pre-onset runway (the number of faithful tokens the
document emits before its first hallucinated one).

This script measures E[theta - 1] on exactly the EVAL half used by
run_hazard_a3_leadtime.py (same first_onset definition, same stratified CAL/EVAL
split with RandomState(1)), and prints the resulting ceiling next to the measured
pre-onset alarm rates of the forecaster and the CUSUM detector.

Read-only: the only input is the per-document token-label array inside the
ForwardGRU detection-probability file. Nothing is written outside this directory.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

try:                                   # same directory on the training host
    from refcheck import check_reference
except ImportError:                    # repo layout, then the released-archive layout
    _here = Path(__file__).resolve()
    for _p in (_here.parents[1] / "paper" / "forecasting" / "diagnostics",
               _here.parent / "diagnostics"):
        if (_p / "refcheck.py").exists():
            sys.path.insert(0, str(_p))
            break
    from refcheck import check_reference

# Same file run_hazard_a3_leadtime.py reads: it lives in HALLU_DIR on the training
# host. Set DET_PROBS to point at a copy held anywhere else.
CANDIDATES = [
    p for p in (
        Path(os.environ["DET_PROBS"]) if os.environ.get("DET_PROBS") else None,
        Path(os.environ.get("HALLU_DIR", str(Path.home() / "hallucination_exp")))
        / "directional_probs_seed42.json",
    ) if p is not None
]

# Measured at the alpha = 1e-3 budget, k = 3, EVAL half (run_hazard_a3_leadtime.py):
# (recall, %early, realized clean-stream FA rate). recall x %early is the
# unconditional rate of a pre-onset alarm; the realized FA rate is the alpha that
# actually held on the EVAL clean documents, which is what the ceiling must use.
MEASURED = {
    "forecaster": (0.46, 0.78, 0.0009),
    "detector (ForwardGRU-CUSUM)": (0.47, 0.29, 0.0011),
}
ALPHA = 1e-3


def first_onset(labs):
    """Index of the first 0->1 transition (changepoint theta, 0-based). None if clean."""
    for t in range(len(labs)):
        if labs[t] > 0.5:
            return t
    return None


def main():
    det_file = next((p for p in CANDIDATES if p.exists()), None)
    if det_file is None:
        raise SystemExit("detection-probability file not found; tried:\n  " +
                         "\n  ".join(str(p) for p in CANDIDATES))
    print(f"[theory-runway] labels from {det_file.name}", flush=True)
    labs_list = json.load(open(det_file))["ForwardGRU"]["labs"]
    n_docs = len(labs_list)

    onsets = [first_onset(l) for l in labs_list]
    clean = [i for i in range(n_docs) if onsets[i] is None]
    hallu = [i for i in range(n_docs) if onsets[i] is not None]
    print(f"docs: clean={len(clean)} hallu={len(hallu)}")

    # CAL/EVAL split, stratified clean/hallu -- identical to run_hazard_a3_leadtime.py
    r = np.random.RandomState(1)
    cl, ha = clean[:], hallu[:]
    r.shuffle(cl)
    r.shuffle(ha)
    cal = set(cl[:len(cl) // 2] + ha[:len(ha) // 2])
    ev = set(range(n_docs)) - cal
    ev_ha = [i for i in hallu if i in ev]

    # theta is 0-based, so onsets[i] IS the runway theta-1 (faithful tokens before onset).
    runway = np.array([onsets[i] for i in ev_ha], dtype=float)
    doc_len = np.array([len(labs_list[i]) for i in ev_ha], dtype=float)
    print(f"\nEVAL hallucinating documents: n={len(ev_ha)}")
    print(f"pre-onset runway theta-1 : mean={runway.mean():.1f}  median={np.median(runway):.1f}  "
          f"p90={np.percentile(runway, 90):.1f}  max={runway.max():.0f}")
    print(f"document length          : mean={doc_len.mean():.1f}  median={np.median(doc_len):.1f}")

    print(f"\nProposition 1 ceiling, budget alpha={ALPHA:g}:  P(T < theta) <= alpha_realized * E[theta-1]")
    excess = {}
    for name, (recall, pct_early, fa) in MEASURED.items():
        rate = recall * pct_early
        ceiling = fa * runway.mean()
        excess[name] = rate / ceiling
        print(f"  {name:<28} realized alpha={fa:.4f}  ceiling={ceiling:.4f}   "
              f"measured P(T<theta)={recall:.2f}x{pct_early:.2f}={rate:.3f}   {rate / ceiling:.1f}x the ceiling")
        print(f"  {'':<28} (the ceiling would hold only if E[theta-1] >= {rate / fa:.0f} tokens)")

    # Executable contracts for the numbers Section 5.4 and Appendix B quote. A rebuilt
    # cache or a changed split that stops reproducing them raises here rather than
    # silently contradicting the manuscript.
    print()
    check_reference("pre-onset runway E[theta-1] (paper 62.4 tokens)", runway.mean(), 62.4, 0.5)
    check_reference("runway median (paper 51 tokens)", float(np.median(runway)), 51.0, 1.0)
    check_reference("forecaster ceiling excess (paper 6.4x)", excess["forecaster"], 6.4, 0.3)
    check_reference("detector ceiling excess (paper 2.0x)",
                    excess["detector (ForwardGRU-CUSUM)"], 2.0, 0.2)
    check_reference("the ceiling is violated, not merely approached",
                    excess["forecaster"], None, None, lower=1.0)


if __name__ == "__main__":
    main()
