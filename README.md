# Forecasting the Onset of Hallucination

Analysis code for the paper **"Forecasting the Onset of Hallucination: Causal,
Token-Level, Black-Box Survival Analysis During Generation"** (Igor Itkin, 2026).

The third paper in a series. Detection asks *whether* a token is hallucinated; the
[quickest-change companion](https://github.com/YehudaItkin/quickest-hallucination-onset)
(arXiv:2606.12476) asks *how fast* one can react after an onset, and proves a delay
bound of about 1.3 tokens **after** onset. This paper asks whether the onset can be
**forecast** — predicted from causal, black-box, token-level features **before** it
begins.

We cast it as discrete-time survival analysis: a per-token hazard
`h_t^(k) = P(onset in (t, t+k] | causal features ≤ t)` over the risk set of
still-faithful tokens, with a streaming-safe feature audit (24 of 33 features) and a
forward-only recurrent hazard head.

## Headline findings

- **Onset is forecastable.** Causal ForwardGRU hazard reaches **0.777 ± 0.002 AUROC**
  at k=3, three tokens before onset (5 seeds), ~12 points over a pointwise logistic
  floor.
- **The precursor is text-novelty drift, not language-model surprisal** — text
  features alone match the full causal set; the LM block adds nothing at the margin.
- **Not autocorrelation, not self-excitation** — beats a matched-capacity
  label-history null by +0.13 AUROC; a Hawkes term dies under a within-document
  permutation null (p=1.0).
- **Within-document timing is real** — restricted to a single document (where any
  prompt-level predictor is uninformative by construction) the forecaster still ranks
  imminent-onset tokens at **0.687 AUROC**.
- **Negative delay** — at a matched false-alarm budget the forecaster warns a median
  of **11 tokens before** onset where the companion detector fires **15 after**
  (Wilcoxon p = 2.6e-19).
- **Domain-specific** — forecastable in a second corpus in-domain (0.74) but does not
  transfer zero-shot; mitigation via a pure abort does not need lead time;
  self-consistency does not help.

## Requirements

```
pip install -r requirements.txt   # numpy, scipy, scikit-learn, torch
```

The scripts consume the **33-dimensional RAGTruth feature pipeline** from the prior
temporal-detection work (`run_extended.py`, `token_features.py`, `onset_metrics.py`
plus the prepared `prepared_data/`, `nli_features/`, `lm_features/`). Point
`HALLU_DIR` at that directory:

```
export HALLU_DIR=/path/to/hallucination_exp
python hazard_data.py            # builds + caches the hazard dataset (v2)
```

The feature pipeline itself is released with the
[Temporal paper](https://github.com/YehudaItkin/temporal-hallucination-detection)
(Zenodo DOI 10.5281/zenodo.20977858). `run_learned_cusum.py` is vendored from the
onset-detection companion so the lead-time comparison is self-contained.

## Reproducing each result

| Script | Result |
| --- | --- |
| `run_hazard_a1.py` | logistic hazard floor + feature-set comparison |
| `run_hazard_a2.py` | ForwardGRU hazard (single feature set) |
| `run_hazard_a2_ablation.py` | 5-seed CIs, text-vs-LM dissociation, static baselines |
| `run_hazard_withindoc.py` | within-document (timing-only) AUROC |
| `run_theory_markov_horizon.py` | matched-capacity autocorrelation null |
| `run_hawkes.py` | Hawkes self-excitation null (permutation-controlled) |
| `run_hazard_a3_leadtime.py` | negative-delay comparison vs the CUSUM detector |
| `run_hazard_a3_psiloqa.py` | zero-shot cross-domain transfer |
| `run_hazard_a3_psiloqa_refit.py` | in-domain PsiloQA refit (disambiguates transfer) |
| `run_hazard_b_taxonomy.py` | predictability taxonomy of onsets |
| `run_hazard_d_mitigation.py` | abort-policy mitigation simulation |
| `run_hazard_c_consistency.py` | self-consistency null (needs consistency features) |

`run_hazard_c_consistency.py` additionally requires the black-box self-consistency
features (SelfCheckGPT resamples) from the companion work; see the paper for details.
The `results/` directory holds the log outputs the paper reports.

## Citation

See `CITATION.cff`. Companion: arXiv:2606.12476 (onset detection); prior work:
Zenodo 10.5281/zenodo.20977858 (feature pipeline).

## License

MIT — see `LICENSE`.
