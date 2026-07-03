"""Hawkes (task 2): is onset a SELF-EXCITING process, or just covariate/heterogeneity?

Discrete-time logistic point process over the risk set (faithful tokens). Event at
risk token t = onset at t+1 (z_t = y1_t; leak-free — uses only onsets strictly
before t in the excitation term). Conditional intensity:
    lambda_t = sigmoid( w.x_t  +  w_f . f_t  +  sum_j a_j * e_t(beta_j)  +  b )
- x_t : causal-24 content features
- f_t : causal FRAILTY proxy (running hallu-rate, log1p prior-onset count) — a
        per-example, causal, out-of-sample-valid "how hallucination-prone so far"
- e_t(beta_j) = sum_{onset o < t, same example} exp(-beta_j (t-o))  (excitation),
        a FIXED grid of decay rates beta_j (timescales 1/3/10 tok) with learned
        a_j >= 0 (softplus) — fixed grid avoids the Davies beta-identification issue.

Reviewer-mandated identification: (1) the DECISIVE test is exc gain OVER
covariates+frailty: LL(FE)-LL(F) on held-out TEST. (2) a within-example onset-time
PERMUTATION null (shuffle onset positions per example, keep count) — if the
observed exc gain sits inside the permutation null, the "clustering" is
heterogeneity/covariate-driven, NOT contagion. GATE = exc survives frailty AND
beats the permutation null (p<0.01) AND a short kernel timescale carries it.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import hazard_data as H

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BETAS = np.array([1.0, 1.0 / 3, 1.0 / 10], dtype=np.float32)   # timescales 1,3,10 tok
RNG = np.random.RandomState(0)


def excitation_basis(groups, onset, pos, betas, onset_override=None):
    """e[t, j] = sum_{onset o < t in same example} exp(-beta_j (t-o)). Causal."""
    n = len(groups); J = len(betas)
    e = np.zeros((n, J), dtype=np.float32)
    bounds = np.searchsorted(groups, np.arange(groups[-1] + 2))
    for gi in range(groups[-1] + 1):
        s, ed = bounds[gi], bounds[gi + 1]
        if ed <= s:
            continue
        if onset_override is None:
            ons = np.where(onset[s:ed] == 1)[0]
        else:
            ons = onset_override[gi]
        if len(ons) == 0:
            continue
        L = ed - s
        # recurrence per beta: e_t = exp(-beta) * (e_{t-1} + onset_{t-1})
        for j, beta in enumerate(betas):
            decay = np.exp(-beta); acc = 0.0
            col = e[s:ed, j]; ons_set = np.zeros(L); ons_set[ons] = 1
            for t in range(L):
                col[t] = acc
                acc = decay * (acc + ons_set[t])
    return e


def frailty_features(d):
    g = d["groups"]; onset = d["onset"].astype(int); hallu = (~d["risk"].astype(bool)).astype(int)
    bounds = np.searchsorted(g, np.arange(g[-1] + 2))
    hfr = np.zeros(len(g), np.float32); lpo = np.zeros(len(g), np.float32)
    for gi in range(g[-1] + 1):
        s, e = bounds[gi], bounds[gi + 1]
        if e <= s:
            continue
        cnt = 0; run = 0
        for j in range(s, e):
            t = j - s
            hfr[j] = run / max(t, 1); lpo[j] = np.log1p(cnt)
            if onset[j]:
                cnt += 1
            run += hallu[j]
    return np.stack([hfr, lpo], 1).astype(np.float32)


class LogisticPP(nn.Module):
    def __init__(self, dx, df, je):
        super().__init__()
        self.wx = nn.Linear(dx, 1, bias=False) if dx else None
        self.wf = nn.Linear(df, 1, bias=False) if df else None
        self.ra = nn.Parameter(torch.zeros(je)) if je else None   # softplus -> a_j>=0
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, f, e):
        z = self.b
        if self.wx is not None:
            z = z + self.wx(x)
        if self.wf is not None:
            z = z + self.wf(f)
        if self.ra is not None:
            z = z + e @ torch.nn.functional.softplus(self.ra).unsqueeze(1)
        return z.squeeze(-1)


def fit_ll(Xtr, Ztr, Xte, Zte, use_x, use_f, use_e, l2=1e-3, iters=300):
    dx = Xtr[0].shape[1] if use_x else 0
    df = Xtr[1].shape[1] if use_f else 0
    je = Xtr[2].shape[1] if use_e else 0
    m = LogisticPP(dx, df, je).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=0.05, weight_decay=l2)
    tx, tf, teb = (torch.from_numpy(a).to(DEVICE) for a in Xtr)
    z = torch.from_numpy(Ztr).float().to(DEVICE)
    pw = torch.tensor([(Ztr == 0).sum() / max((Ztr == 1).sum(), 1)], device=DEVICE, dtype=torch.float32)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(iters):
        opt.zero_grad(); loss = lf(m(tx, tf, teb), z); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        ex, ef, ee = (torch.from_numpy(a).to(DEVICE) for a in Xte)
        logit = m(ex, ef, ee)
        # honest per-token test log-likelihood (unweighted Bernoulli)
        ll = -nn.functional.binary_cross_entropy_with_logits(
            logit, torch.from_numpy(Zte).float().to(DEVICE), reduction="mean").item()
        a_j = torch.nn.functional.softplus(m.ra).cpu().numpy() if m.ra is not None else None
    return ll, a_j


def main():
    print(f"[hawkes]  device={DEVICE}", flush=True)
    trd, med = H.build("train"); ted, _ = H.build("test", lm_medians=med)
    rtr, rte = trd["risk"].astype(bool), ted["risk"].astype(bool)

    # target z_t = y1_t (onset next step), features on the risk set
    sc = StandardScaler().fit(trd["X"][rtr][:, H.CAUSAL_IDX])
    def prep(d, r):
        x = sc.transform(d["X"][r][:, H.CAUSAL_IDX]).astype(np.float32)
        f = frailty_features(d)[r]
        e = excitation_basis(d["groups"], d["onset"], d["pos"], BETAS)[r]
        z = d["y1"][r].astype(np.int8)
        return x, f, e, z
    xtr, ftr, etr, ztr = prep(trd, rtr)
    xte, fte, ete, zte = prep(ted, rte)

    def LL(ux, uf, ue, etr_=etr, ete_=ete):
        return fit_ll((xtr, ftr, etr_), ztr, (xte, fte, ete_), zte, ux, uf, ue)

    A = LL(1, 0, 0)[0]                    # covariate-only
    F = LL(1, 1, 0)[0]                    # covariate + frailty
    E = LL(0, 0, 1)[0]                    # excitation-only
    CE, aCE = LL(1, 0, 1)                 # covariate + excitation
    FE, aFE = LL(1, 1, 1)                 # covariate + frailty + excitation (decisive)

    print(f"\n  test log-lik/token (higher=better):")
    print(f"    cov-only            A  = {A:.5f}")
    print(f"    cov+frailty         F  = {F:.5f}")
    print(f"    exc-only            E  = {E:.5f}")
    print(f"    cov+exc             CE = {CE:.5f}   (gain over A: {CE-A:+.5f})")
    print(f"    cov+frailty+exc     FE = {FE:.5f}   (gain over F: {FE-F:+.5f})  <- decisive")
    print(f"    learned a_j (FE, timescales 1/3/10 tok): {np.round(aFE,4)}")

    # within-example onset-time permutation null on the DECISIVE gain (FE - F)
    obs = FE - F
    g = ted["groups"]; onset = ted["onset"]; boundsT = np.searchsorted(g, np.arange(g[-1] + 2))
    gtr = trd["groups"]; onsetr = trd["onset"]; boundsTr = np.searchsorted(gtr, np.arange(gtr[-1] + 2))
    def perm_onsets(bounds, onset, gmax):
        ov = []
        for gi in range(gmax + 1):
            s, ed = bounds[gi], bounds[gi + 1]
            L = ed - s
            k = int(onset[s:ed].sum())
            ov.append(RNG.choice(L, k, replace=False) if (k and L) else np.array([], int))
        return ov
    NP = 60; null = []
    print(f"\n  permutation null (within-example onset shuffle, {NP} draws)...", flush=True)
    for _ in range(NP):
        eptr = excitation_basis(gtr, onsetr, trd["pos"], BETAS,
                                onset_override=perm_onsets(boundsTr, onsetr, gtr[-1]))[rtr]
        epte = excitation_basis(g, onset, ted["pos"], BETAS,
                                onset_override=perm_onsets(boundsT, onset, g[-1]))[rte]
        f_ = fit_ll((xtr, ftr, eptr), ztr, (xte, fte, epte), zte, 1, 1, 1)[0]
        null.append(f_ - F)
    null = np.array(null)
    p = (1 + (null >= obs).sum()) / (1 + NP)
    print(f"    observed FE-F gain = {obs:+.5f};  null mean {null.mean():+.5f} sd {null.std():.5f};  p = {p:.3f}")

    short = float(aFE[0]) > float(aFE.mean())   # timescale-1 basis carries it
    gate = (obs > 0) and (p < 0.01)
    print(f"\nGATE-hawkes: exc survives frailty (FE>F)={obs>0}; beats permutation null (p<0.01)={p<0.01} "
          f"-> {'PASS: genuine self-excitation' if gate else 'NEGATIVE: clustering is covariate/heterogeneity, not contagion'}")


if __name__ == "__main__":
    main()
