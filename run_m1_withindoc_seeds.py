"""Within-document AUROC over 5 seeds -> mean+-std, to match Table 1 rigor.
The paper reports within-doc 0.687 (single seed, bootstrap-CI over docs); this adds
across-seed variance for the key timing number.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import hazard_data as H
from run_hazard_a2 import to_sequences, standardize, train_eval, HOR
from run_hazard_a3_leadtime import forecaster_paths
from refcheck import check_reference
KS=[1,3,5,10]; IDX=H.CAUSAL_IDX

def wd_auc(fsc,lab):
    a=[roc_auc_score(lab[gi],fsc[gi]) for gi in fsc if lab[gi].min()!=lab[gi].max()]
    return np.array(a)

def main():
    trd,med=H.build("train"); ted,_=H.build("test",lm_medians=med)
    te=to_sequences(ted)
    rmask=trd["risk"].astype(bool); Xr=trd["X"][rmask][:,IDX]
    mean,std=Xr.mean(0),Xr.std(0)+1e-6
    standardize(te,mean,std,IDX)
    te_by_gi={s["gi"]:s for s in te}
    pw=[((trd[f"y{k}"][rmask]==0).sum()/max((trd[f"y{k}"][rmask]==1).sum(),1)) for k in HOR]
    per={k:[] for k in KS}
    for seed in range(5):
        tr_all=to_sequences(trd)
        rng=np.random.RandomState(seed); rng.shuffle(tr_all)
        nv=int(0.15*len(tr_all)); va,tr=tr_all[:nv],tr_all[nv:]
        standardize(tr,mean,std,IDX); standardize(va,mean,std,IDX)
        model,_=train_eval(tr,va,te,IDX,pw,seed=seed)
        line=[]
        for k in KS:
            fp=forecaster_paths(model,te,mean,std,IDX,HOR.index(k))
            fsc,lab={},{}
            for gi,s in te_by_gi.items():
                r=s["risk"].astype(bool)
                if r.sum()<2: continue
                fsc[gi]=fp[gi][r]; lab[gi]=s["Y"][:,HOR.index(k)][r]
            m=wd_auc(fsc,lab).mean(); per[k].append(m); line.append(f"k{k}={m:.3f}")
        print(f"seed {seed}: "+" ".join(line),flush=True)
    print("\n  k | within-doc AUROC  mean +- std (5 seeds)")
    print("  "+"-"*40)
    for k in KS:
        a=np.array(per[k]); print(f"  {k:>2} | {a.mean():.3f} +- {a.std():.3f}")

    # Regression guard: the paper's within-doc timing number (0.687 +- 0.003).
    check_reference("within-doc AUROC@3 (paper 0.687)", float(np.array(per[3]).mean()), 0.687, 0.01)

if __name__=="__main__":
    main()
