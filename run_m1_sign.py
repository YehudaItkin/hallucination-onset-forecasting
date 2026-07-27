"""M1 mechanism: does the feature->onset relationship flip sign across domains?
Point-biserial corr of each causal feature with y3 (onset within 3) over the risk
set, RAGTruth-test vs PsiloQA-test (each under its own lm_medians). Text vs LM split.
"""
import numpy as np
import hazard_data as H
from refcheck import check_reference
IDX = H.CAUSAL_IDX
TEXT = [i for i in IDX if i < 20]   # 18 text
LM   = [i for i in IDX if i >= 27]  # 6 lm

def corrs(build_args):
    d,_ = H.build(*build_args[0], **build_args[1])
    r = d["risk"].astype(bool)
    X = d["X"][r]; y = d["y3"][r].astype(float)
    out={}
    for i in IDX:
        x=X[:,i]
        if x.std()<1e-9: out[i]=0.0
        else: out[i]=float(np.corrcoef(x,y)[0,1])
    return out

rag = corrs((("test",), {"lm_medians": H.build("train")[1]}))
pq  = corrs((("psiloqa_test",), {"lm_medians": H.build("psiloqa_train", cache_tag="_pq")[1], "cache_tag":"_pq"}))

flips=[i for i in IDX if rag[i]*pq[i] < 0 and abs(rag[i])>0.02 and abs(pq[i])>0.02]
print(f"\n  idx  grp |  corr_RAG  corr_PsiloQA  flip?")
print("-"*44)
for i in IDX:
    grp = "text" if i in TEXT else ("lm" if i in LM else "?")
    fl = "  FLIP" if (rag[i]*pq[i]<0 and abs(rag[i])>0.02 and abs(pq[i])>0.02) else ""
    print(f"  {i:>3}  {grp:>4} | {rag[i]:>+7.3f}   {pq[i]:>+7.3f}   {fl}")
print(f"\nsign flips (|r|>0.02 both): {len(flips)} of {len(IDX)} causal features")
tf=[i for i in flips if i in TEXT]; lf=[i for i in flips if i in LM]
print(f"  text flips: {len(tf)}/{len(TEXT)}   lm flips: {len(lf)}/{len(LM)}")
print(f"  mean corr_RAG  text={np.mean([rag[i] for i in TEXT]):+.3f} lm={np.mean([rag[i] for i in LM]):+.3f}")
print(f"  mean corr_PsiloQA text={np.mean([pq[i] for i in TEXT]):+.3f} lm={np.mean([pq[i] for i in LM]):+.3f}")

# Regression guard: the mechanism claim (paper: 10 of 18 text features flip sign).
check_reference("text-feature sign flips (paper 10/18)", len(tf), None, None, lower=7)
