"""Turn the saved stdout of run_m1_sign.py into the Appendix D tabular.

This script computes nothing. run_m1_sign.py is the authority for the numbers; this
only reads its log and re-prints the flipped text features as LaTeX, so the appendix
table cannot drift from what the diagnostic actually produced.

    cd ~/paper/forecasting/diagnostics && tmux new -s m1sign
    python run_m1_sign.py 2>&1 | tee m1_sign.log
    python format_m1_sign.py m1_sign.log

run_m1_sign.py prints one row per causal feature as

      idx  grp |  corr_RAG  corr_PsiloQA  flip?
       11  text |  +0.048    -0.151      FLIP

and a summary line "text flips: N/18   lm flips: M/6". Both are parsed here.
"""
import re
import sys

# 33-dim feature order, identical to run_hazard_b_taxonomy.NAMES (text 0..19,
# nli 20..26, lm 27..32). Repeated rather than imported so the formatter runs
# anywhere, including off the training host.
NAMES = ["in_context", "is_number", "is_capitalized", "word_length", "position",
         "running_ctx_ratio", "running_novel_ratio", "is_novel", "num_in_context",
         "novel_number", "window_5_novelty", "window_10_novelty", "window_20_novelty",
         "delta_novel_ratio", "delta2_novel_ratio", "bigram", "trigram", "entity",
         "consec", "sent_pos"] + [f"nli_{i}" for i in range(7)] + [
         "lm_logprob", "lm_entropy", "lm_log1p_c2", "lm_log1p_c3",
         "lm_has_signal", "lm_logprob_x_signal"]

PRETTY = {
    "in_context": "in-context token", "is_number": "numeric token",
    "is_capitalized": "capitalized", "word_length": "word length",
    "running_ctx_ratio": "running in-context ratio",
    "running_novel_ratio": "running novel ratio", "is_novel": "novel token",
    "num_in_context": "in-context numeral", "novel_number": "novel numeral",
    "window_5_novelty": "width-5 novelty window",
    "window_10_novelty": "width-10 novelty window",
    "window_20_novelty": "width-20 novelty window",
    "delta_novel_ratio": "novelty first difference",
    "delta2_novel_ratio": "novelty second difference",
    "bigram": "bigram overlap", "trigram": "trigram overlap",
    "entity": "entity flag", "consec": "consecutive-novel counter",
    "lm_logprob": "token log-probability", "lm_entropy": "entropy",
    "lm_log1p_c2": "log1p top-2 gap", "lm_log1p_c3": "log1p top-3 gap",
    "lm_has_signal": "signal-availability flag",
    "lm_logprob_x_signal": "log-probability $\\times$ flag",
}

ROW = re.compile(r"^\s*(\d+)\s+(text|lm)\s*\|\s*([+-][\d.]+)\s+([+-][\d.]+)\s*(FLIP)?\s*$")
SUMMARY = re.compile(r"text flips:\s*(\d+)/(\d+)\s+lm flips:\s*(\d+)/(\d+)")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <m1_sign.log>")
    text = open(sys.argv[1]).read()

    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            idx, grp, r_rag, r_pq, flip = m.groups()
            rows.append((int(idx), grp, float(r_rag), float(r_pq), flip is not None))
    if not rows:
        raise SystemExit("no feature rows parsed; is this run_m1_sign.py output?")

    s = SUMMARY.search(text)
    if s:
        tf, tn, lf, ln = (int(x) for x in s.groups())
        print(f"% parsed {len(rows)} causal features; text flips {tf}/{tn}, lm flips {lf}/{ln}")
        counted = sum(1 for _i, g, _a, _b, f in rows if g == "text" and f)
        if counted != tf:
            raise SystemExit(f"parse mismatch: {counted} flipped text rows vs summary {tf}")
    else:
        print(f"% parsed {len(rows)} causal features (no summary line found)")

    flipped = [r for r in rows if r[1] == "text" and r[4]]
    flipped.sort(key=lambda r: -abs(r[2]))

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\footnotesize")
    print("\\setlength{\\tabcolsep}{4pt}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("causal text feature & $r$ RAGTruth & $r$ PsiloQA \\\\")
    print("\\midrule")
    for idx, _grp, r_rag, r_pq, _f in flipped:
        print(f"{PRETTY.get(NAMES[idx], NAMES[idx])} & ${r_rag:+.3f}$ & ${r_pq:+.3f}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{The causal text features whose correlation with imminent onset "
          "($y^{(3)}$ over the risk set) reverses between the corpora, largest "
          "$|r|$ in RAGTruth first. Each corpus is standardized with its own "
          "language-model medians, so the reversal is not a normalization artifact. "
          "A feature counts as flipped when the two correlations have opposite signs "
          "and $|r|>0.02$ in both.}")
    print("\\label{tab:signflip}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
