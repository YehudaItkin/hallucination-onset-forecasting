"""A1: build the hazard-forecasting dataset from the paper's EXACT 33-dim features.

We reuse the original feature pipeline verbatim (no rewriting):
  - token_features.extract_token_features / extract_token_labels  (worktree)
  - run_extended.enrich_text_features / enrich_nli_features /
    enrich_lm_features / compute_lm_train_medians / assemble_features  (main)
  - onset_metrics.find_span_onsets  (worktree)

Data (prepared_data, nli_features, lm_features) is READ-ONLY from the
original feature-extraction pipeline. Nothing is written back to it.

Hazard label (survival-style): for each faithful token t, over horizon k,
    y_t^(k) = 1[ some hallucination-span onset o with t < o <= t+k ].
Risk set = faithful tokens (label==0). Tokens inside a span are already in the
event state and are excluded from the risk set.

Feature settings:
  - "full"  : all 33 features (includes future-leaking ones) -> optimistic ceiling
  - "causal": 24 streaming-safe features. Dropped, with reason:
      #4  position     = i/(output_len-1)   -> normalized by TOTAL length (leak)
      #19 sent_pos     = si/(n_sents-1)     -> normalized by TOTAL #sentences (leak)
      #20-26 NLI (7)   = sentence-level, projected to every token of the sentence
                         including tokens before the sentence is complete (leak)
"""
import os
import sys
from pathlib import Path

import numpy as np

# Single consistent source of code + data. On the GPU server this is
# ~/hallucination_exp (the originals that produced the paper's numbers, with
# matching train+test prepared_data / nli_features / lm_features).
HALLU_DIR = Path(os.environ.get("HALLU_DIR", str(Path.home() / "hallucination_exp")))
CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

sys.path.insert(0, str(HALLU_DIR))   # token_features, run_extended, onset_metrics

import run_extended as R              # noqa: E402
R.DATA_DIR = HALLU_DIR                # original loaders read from here
from onset_metrics import find_span_onsets  # noqa: E402

HORIZONS = [1, 3, 5, 10]

# 33-dim order = text[0:20] + nli[20:27] + lm[27:33]
NONCAUSAL_IDX = [4, 19] + list(range(20, 27))
CAUSAL_IDX = [i for i in range(33) if i not in NONCAUSAL_IDX]


def _assemble_split(split, lm_medians=None):
    """Return (features_list[33-dim per example], examples), plus lm_medians (train)."""
    base, _labs, ex = R.load_base_features(split)
    base = R.enrich_text_features(base, ex)
    nli = R.enrich_nli_features(R.load_extra_features(split, "nli"))
    lm_raw = R.load_extra_features(split, "lm")
    if lm_medians is None:
        lm_medians = R.compute_lm_train_medians(lm_raw)
    lm = R.enrich_lm_features(lm_raw, lm_medians)
    feats = R.assemble_features(base, nli, lm, use_nli=True, use_lm=True)
    return feats, ex, lm_medians


def _hazard_labels(token_labels):
    """Per-token: risk mask (faithful) and y^(k) for each horizon."""
    n = len(token_labels)
    is_hallu = np.array([int(t["is_hallucination"]) for t in token_labels])
    onsets = np.array(find_span_onsets(token_labels), dtype=int)
    risk = (is_hallu == 0)  # eligible tokens: currently faithful
    y = {}
    for k in HORIZONS:
        yk = np.zeros(n, dtype=np.int8)
        if len(onsets):
            for t in range(n):
                # onset strictly after t, within horizon k
                if np.any((onsets > t) & (onsets <= t + k)):
                    yk[t] = 1
        y[k] = yk
    return risk, y, onsets


def build(split, lm_medians=None, limit=None, cache_tag=""):
    """Assemble X (token x 33), per-horizon y, risk mask, example-id groups.

    cache_tag namespaces the cache so the SAME split built with different
    lm_medians (e.g. PsiloQA under RAGTruth-train vs its own medians) does not
    collide (reviewer-flagged divergent-normalization risk).
    """
    tag = f"{split}" + (f"_lim{limit}" if limit else "") + cache_tag
    npz = CACHE / f"hazard_{tag}_v2.npz"   # v2: adds onset(int8) + pos(int32 within-group)
    if npz.exists():
        # cache is self-generated (plain numeric arrays only) -> pickle not needed
        d = np.load(npz)
        return {k: d[k] for k in d.files}, lm_medians

    feats, ex, lm_medians = _assemble_split(split, lm_medians)
    if limit:
        feats, ex = feats[:limit], ex[:limit]

    X, groups, pos_all, onset_all = [], [], [], []
    risk_all = []
    y_all = {k: [] for k in HORIZONS}
    onset_count = 0
    for gi, (f, e) in enumerate(zip(feats, ex)):
        if len(f) == 0 or f.shape[1] != 33:
            continue
        risk, y, onsets = _hazard_labels(e["token_labels"])
        onset_count += len(onsets)
        n = len(f)
        onset_ind = np.zeros(n, dtype=np.int8)
        if len(onsets):
            onset_ind[onsets] = 1
        X.append(f.astype(np.float32))
        groups.append(np.full(n, gi, dtype=np.int32))
        pos_all.append(np.arange(n, dtype=np.int32))          # within-example index
        onset_all.append(onset_ind)                            # 1 at span starts
        risk_all.append(risk)
        for k in HORIZONS:
            y_all[k].append(y[k])

    out = {
        "X": np.concatenate(X),
        "groups": np.concatenate(groups),
        "pos": np.concatenate(pos_all),
        "onset": np.concatenate(onset_all),
        "risk": np.concatenate(risk_all),
    }
    for k in HORIZONS:
        out[f"y{k}"] = np.concatenate(y_all[k])
    out["n_examples"] = np.array([len(X)])
    out["n_onsets"] = np.array([onset_count])
    # sanity: onset indicators must match find_span_onsets count
    assert int(out["onset"].sum()) == onset_count, "onset indicator/count mismatch"
    np.savez_compressed(npz, **out)
    return out, lm_medians


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    tr, med = build("train", limit=a.limit)
    te, _ = build("test", lm_medians=med, limit=a.limit)
    for name, d in [("train", tr), ("test", te)]:
        risk = d["risk"].astype(bool)
        print(f"\n[{name}] examples={int(d['n_examples'][0])} tokens={len(d['X'])} "
              f"risk-set={risk.sum()} onsets={int(d['n_onsets'][0])}")
        print(f"  causal dims={len(CAUSAL_IDX)} / 33")
        for k in HORIZONS:
            yk = d[f"y{k}"][risk]
            print(f"  horizon k={k:2d}: risk-set positive rate = {yk.mean():.4f} "
                  f"({int(yk.sum())} / {len(yk)})")
