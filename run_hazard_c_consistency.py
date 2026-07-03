"""Phase C: does a black-box self-consistency measurement help the FORECASTER?

The consistency block (SelfCheckGPT n-gram + NLI over K=5 resamples) was a clean
null for the CUSUM DETECTOR (H2). Here we ask the different, forecasting question:
does it raise the hazard forecaster's AUROC@k? Two honest caveats up front:
  1. Consistency features resample the WHOLE output, so they are NOT streaming --
     this is an OFFLINE augmentation / upper bound, not the streaming forecaster.
  2. Only the llama-2-7b-chat RAGTruth-train subset has consistency features, so we
     do an internal paired 70/15/15 split (base vs base+consistency share the split
     and the model seed) and report over 5 seeds (H2 lesson: >=5 before believing).

We reuse the exact H2 alignment (build_subset: consistency aligned to the 33-d base
by example id + token count) and only swap the CUSUM model for the paper's causal
ForwardGRU-hazard head. base = causal-24; aug = causal-24 + 2 consistency dims.

GATE: aug AUROC@3 exceeds base by more than 1 sigma over 5 seeds -> consistency
carries independent anticipatory signal; else null (consistency doesn't forecast).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# server_consistency_h2 (+ run_extended, run_learned_cusum) live in HALLU_DIR
sys.path.insert(0, os.environ.get("HALLU_DIR", os.path.expanduser("~/hallucination_exp")))

from server_consistency_h2 import build_subset, split_indices  # noqa: E402
import hazard_data as H  # noqa: E402
from run_hazard_a2 import ForwardGRUHazard, batches, evaluate, standardize, HOR  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAUSAL = H.CAUSAL_IDX  # 24 streaming-safe indices into the 33-d base


def make_seqs(base, aug, ex, use_consistency):
    """per-doc sequences with hazard multi-horizon labels; feature = causal-24
    (+ the 2 consistency dims if use_consistency)."""
    seqs = []
    for b, a, e in zip(base, aug, ex):
        risk, y, _ = H._hazard_labels(e["token_labels"])
        n = len(b)
        if n == 0 or len(risk) != n:
            continue
        feat = np.hstack([b[:, CAUSAL], a[:, b.shape[1]:]]) if use_consistency else b[:, CAUSAL]
        seqs.append({"X": feat.astype(np.float32),
                     "risk": risk.astype(bool),
                     "Y": np.stack([y[k] for k in HOR], axis=1).astype(np.float32)})
    return seqs


def train_eval_local(tr, va, te, dim, pos_weight, epochs=15, bs=32, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ForwardGRUHazard(dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(reduction="none",
                                 pos_weight=torch.tensor(pos_weight, device=DEVICE))
    best, best_state, patience = -1, None, 0
    for _ in range(epochs):
        model.train()
        for X, Y, M in batches(tr, bs, True):
            opt.zero_grad()
            m = M.unsqueeze(-1).float()
            loss = (lossf(model(X), Y) * m).sum() / m.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        va_auc = evaluate(model, va)[f"auc@{HOR[1]}"]
        if va_auc > best:
            best, best_state, patience = va_auc, {k: v.cpu().clone()
                                                  for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return evaluate(model, te)


def prep(seqs, tr_i, va_i, te_i):
    tr = [seqs[i] for i in tr_i]; va = [seqs[i] for i in va_i]; te = [seqs[i] for i in te_i]
    Xr = np.concatenate([s["X"][s["risk"]] for s in tr])
    mean, std = Xr.mean(0), Xr.std(0) + 1e-6
    idx = list(range(seqs[0]["X"].shape[1]))
    standardize(tr, mean, std, idx); standardize(va, mean, std, idx); standardize(te, mean, std, idx)
    pw = []
    for k in HOR:
        yk = np.concatenate([s["Y"][:, HOR.index(k)][s["risk"]] for s in tr])
        pw.append((yk == 0).sum() / max((yk == 1).sum(), 1))
    return tr, va, te, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consistency", default="consistency_feats_train_llama7b.json")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    print(f"[Phase C] consistency -> forecaster  device={DEVICE}", flush=True)
    base, aug, labs, ex = build_subset(a.consistency)
    ncons = aug[0].shape[1] - base[0].shape[1]
    print(f"consistency dims added: {ncons}", flush=True)

    res = {"base": {k: [] for k in HOR}, "aug": {k: [] for k in HOR}}
    for name, uc in [("base", False), ("aug", True)]:
        seqs = make_seqs(base, aug, ex, uc)
        tr_i, va_i, te_i = split_indices(len(seqs))   # fixed split, paired base/aug
        tr, va, te, pw = prep(seqs, tr_i, va_i, te_i)
        for sd in a.seeds:
            m = train_eval_local(tr, va, te, seqs[0]["X"].shape[1], pw, seed=sd)
            for k in HOR:
                res[name][k].append(m[f"auc@{k}"])
            print(f"  {name:>4} seed {sd}: " +
                  " ".join(f"@{k}={m[f'auc@{k}']:.3f}" for k in HOR), flush=True)

    print(f"\n  {'k':>3} | {'base AUROC':>16} | {'aug (+consistency)':>18} | {'Δ (aug-base)':>12}")
    print("-" * 60)
    hit = 0
    for k in HOR:
        bm, bs_ = np.mean(res["base"][k]), np.std(res["base"][k])
        am, as_ = np.mean(res["aug"][k]), np.std(res["aug"][k])
        d = am - bm
        print(f"  {k:>3} | {bm:.3f} ± {bs_:.3f}    | {am:.3f} ± {as_:.3f}     | {d:+.3f}")
        if k == 3 and d > max(bs_, as_):
            hit = 1
    print(f"\nGATE-C: does consistency lift AUROC@3 beyond 1σ? "
          f"-> {'YES (consistency forecasts)' if hit else 'NO -- null, consistency adds no anticipatory signal'}")


if __name__ == "__main__":
    main()
