"""M1 diagnostic: is the RAGTruth->PsiloQA zero-shot inversion (<0.5) a
cross-domain STANDARDIZATION artifact, or a genuine reversal of the mapping?

Same RAGTruth-trained forecaster in all variants (NO weight refit):
  Z  true zero-shot : PsiloQA features under RAGTruth-train lm_medians + RAGTruth mean/std
  N  re-normalized  : PsiloQA features under PsiloQA-train lm_medians + PsiloQA-train mean/std
If N >> Z and N > 0.5 => inversion was normalization; signal is present, needs only
per-domain input calibration (no retraining). If N still < 0.5 => genuine reversal.
"""
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import hazard_data as H
from run_hazard_a2 import to_sequences, standardize, train_eval, batches, HOR
from refcheck import check_reference
IDX = H.CAUSAL_IDX

def auroc(model, seqs):
    model.eval(); ys={k:[] for k in HOR}; ps={k:[] for k in HOR}
    with torch.no_grad():
        for X,Y,M in batches(seqs,64,False):
            p=torch.sigmoid(model(X)).cpu().numpy()
            Yc,Mc=Y.cpu().numpy(),M.cpu().numpy().astype(bool)
            for hi,k in enumerate(HOR):
                ys[k].append(Yc[...,hi][Mc]); ps[k].append(p[...,hi][Mc])
    return {k:(float('nan') if (y:=np.concatenate(ys[k])).min()==y.max()
               else roc_auc_score(y, np.concatenate(ps[k]))) for k in HOR}

def main():
    print("[M1] training RAGTruth forecaster (causal-24)...", flush=True)
    trd, med = H.build("train"); ted,_=H.build("test", lm_medians=med)
    tr_all=to_sequences(trd); te=to_sequences(ted)
    rng=np.random.RandomState(0); rng.shuffle(tr_all)
    nv=int(0.15*len(tr_all)); va,tr=tr_all[:nv],tr_all[nv:]
    rmask=trd["risk"].astype(bool); Xr=trd["X"][rmask][:,IDX]
    mean_r,std_r=Xr.mean(0),Xr.std(0)+1e-6
    standardize(tr,mean_r,std_r,IDX); standardize(va,mean_r,std_r,IDX); standardize(te,mean_r,std_r,IDX)
    pw=[((trd[f"y{k}"][rmask]==0).sum()/max((trd[f"y{k}"][rmask]==1).sum(),1)) for k in HOR]
    model, rag = train_eval(tr,va,te,IDX,pw)

    # Variant Z: true zero-shot
    print("[M1] Variant Z: RAGTruth med + RAGTruth mean/std ...", flush=True)
    pqz,_=H.build("psiloqa_test", lm_medians=med)
    sz=to_sequences(pqz); standardize(sz,mean_r,std_r,IDX)
    Z=auroc(model,sz)

    # Variant N: re-normalized (PsiloQA med + PsiloQA-train mean/std), SAME model
    print("[M1] Variant N: PsiloQA med + PsiloQA-train mean/std (no refit) ...", flush=True)
    pqtr,pqmed=H.build("psiloqa_train", cache_tag="_pq")
    pqte,_=H.build("psiloqa_test", lm_medians=pqmed, cache_tag="_pq")
    prmask=pqtr["risk"].astype(bool); Xp=pqtr["X"][prmask][:,IDX]
    mean_p,std_p=Xp.mean(0),Xp.std(0)+1e-6
    sn=to_sequences(pqte); standardize(sn,mean_p,std_p,IDX)
    N=auroc(model,sn)

    print("\n  k  | RAGTruth in-dom | Z zero-shot | N re-normalized (same weights)")
    print("-"*66)
    for k in HOR:
        print(f"  {k:>2} |     {rag[f'auc@{k}']:.3f}       |   {Z[k]:.3f}     |   {N[k]:.3f}")
    print(f"\nverdict@3: Z={Z[3]:.3f}  N={N[3]:.3f}  "
          f"-> {'NORMALIZATION artifact (N recovers >0.5)' if N[3]>0.5 else 'genuine reversal (N still <0.5)'}")

    # Regression guards: the paper's M1 claims, as executable contracts.
    check_reference("zero-shot AUROC@3 (paper 0.383)", Z[3], 0.383, 0.03)
    check_reference("re-normalized AUROC@3 stays below chance", N[3], None, None, upper=0.5)

if __name__=="__main__":
    main()
