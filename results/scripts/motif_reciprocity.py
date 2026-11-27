"""Pilot (2026-11-27): binary reciprocity E<->E, I<->I, E<->I on the front networks (4 points x 5 seeds) vs a
senders-shuffled-per-target null. Connected = |W|/c > 0.01 (as like-to-like). Prints only (reports/IDEAS.md B3).

    python results/scripts/motif_reciprocity.py      (from results/scripts, ~45 s)
"""
import contextlib, io, time
import numpy as np, torch
from cmc.train_cog import load_checkpoint
from fig_tv_ei_fanin import RUNS
torch.set_num_threads(1)
POINTS, THRESH, NPERM = ["control", "firing", "wiring", "both"], 0.01, 200

def perm_cols(A, rng, square):                     # target null: each post keeps its inputs, senders shuffled
    n = A.shape[0]
    if not square:
        return np.take_along_axis(A, rng.random(A.shape).argsort(0), 0)
    off = ~np.eye(n, dtype=bool)
    v = A.T[off.T].reshape(n, n - 1).T
    v = np.take_along_axis(v, rng.random(v.shape).argsort(0), 0)
    out = np.zeros_like(A); out.T[off.T] = v.T.ravel(); return out

def recip(Cab, Cba):                               # P(a->b and b->a) / (P(a->b) P(b->a))
    return (Cab & Cba.T).mean() / max(Cab.mean() * Cba.mean(), 1e-12)

t0 = time.time()
rows = []
for p in POINTS:
    for s in range(5):
        with contextlib.redirect_stdout(io.StringIO()):
            model, cfg, ck = load_checkpoint(RUNS["front"] / p / f"seed_{s:02d}.pt")
        ne = model.n_e
        W = np.abs(model.effective_w_rec().detach().numpy()) / model.ei_row_scale().detach().numpy()[:, None]
        np.fill_diagonal(W, 0)
        C = W > THRESH
        EE, II, EI, IE = C[:ne, :ne], C[ne:, ne:], C[:ne, ne:], C[ne:, :ne]
        offE, offI = ~np.eye(ne, dtype=bool), ~np.eye(len(II), dtype=bool)
        rE = (EE & EE.T)[offE].mean() / EE[offE].mean() ** 2
        rI = (II & II.T)[offI].mean() / II[offI].mean() ** 2
        rEI = recip(EI, IE)
        rng = np.random.default_rng(s)
        nE, nI, nEI = [], [], []
        for _ in range(NPERM):
            P = perm_cols(EE, rng, True); nE.append((P & P.T)[offE].mean() / P[offE].mean() ** 2)
            P = perm_cols(II, rng, True); nI.append((P & P.T)[offI].mean() / P[offI].mean() ** 2)
            nEI.append(recip(perm_cols(EI, rng, False), IE))
        # weighted: corr of the two directions over pairs with at least one synapse
        iu = np.triu_indices(ne, 1); a, b = W[:ne, :ne][iu], W[:ne, :ne].T[iu]; k = (a > THRESH) | (b > THRESH)
        wE = np.corrcoef(a[k], b[k])[0, 1] if k.sum() > 3 else np.nan
        rows.append((p, s, EE[offE].mean(), II[offI].mean(), EI.mean(), IE.mean(), rE, np.mean(nE), rI, np.mean(nI), rEI, np.mean(nEI), wE))
print(f"time {time.time() - t0:.1f} s for 20 networks, {NPERM} shuffles each\n")
print(f"{'':8s} {'p EE':>5s} {'p II':>5s} {'p EI':>5s} {'p IE':>5s} | {'E<->E obs/null':>15s} | {'I<->I obs/null':>15s} | {'E<->I obs/null':>15s} | corr wEE")
for p in POINTS:
    R = np.array([r[2:] for r in rows if r[0] == p], float)
    m = R.mean(0)
    print(f"{p:8s} {m[0]:5.2f} {m[1]:5.2f} {m[2]:5.2f} {m[3]:5.2f} | {m[4]:6.2f} / {m[5]:5.2f}  | {m[6]:6.2f} / {m[7]:5.2f}  | {m[8]:6.2f} / {m[9]:5.2f}  | {m[10]:+.2f}")
    print(f"{'  seeds':8s} E<->E obs: {np.round(R[:, 4], 2).tolist()}  I<->I: {np.round(R[:, 6], 2).tolist()}  E<->I: {np.round(R[:, 8], 2).tolist()}")
