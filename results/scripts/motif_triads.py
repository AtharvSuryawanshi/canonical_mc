"""Pilot (2026-11-27): triad census of the E->E graph (front, 4 points x 5 seeds) vs Song et al. 2005's pair
null. Prints only (reports/IDEAS.md B3).

    python results/scripts/motif_triads.py           (from results/scripts, ~90 s)
"""
import contextlib, io, itertools, time
from math import comb
import networkx as nx, numpy as np, torch
from cmc.train_cog import load_checkpoint
from fig_tv_ei_fanin import RUNS
POINTS, THRESH = ["control", "firing", "wiring", "both"], 0.01
NAMES = nx.algorithms.triads.TRIAD_NAMES
SHOW = ["021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]

def null_probs(pn, pu, pb):                           # P(triad class) when the 3 pairs are independent
    out = dict.fromkeys(NAMES, 0.0)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for states in itertools.product(range(4), repeat=3):   # 0 none, 1 a->b, 2 b->a, 3 both
        G = nx.DiGraph(); G.add_nodes_from(range(3)); pr = 1.0
        for (a, b), st in zip(pairs, states):
            pr *= (pn, pu / 2, pu / 2, pb)[st]
            if st in (1, 3): G.add_edge(a, b)
            if st in (2, 3): G.add_edge(b, a)
        c = nx.triadic_census(G)
        out[next(k for k, v in c.items() if v)] += pr
    return out

t0 = time.time(); res = {}
for p in POINTS:
    for s in range(5):
        with contextlib.redirect_stdout(io.StringIO()):
            model, _, _ = load_checkpoint(RUNS["front"] / p / f"seed_{s:02d}.pt")
        ne = model.n_e
        W = np.abs(model.effective_w_rec().detach().numpy()) / model.ei_row_scale().detach().numpy()[:, None]
        C = W[:ne, :ne] > THRESH; np.fill_diagonal(C, False)
        iu = np.triu_indices(ne, 1); ab, ba = C[iu], C.T[iu]
        pb = (ab & ba).mean(); pu = (ab ^ ba).mean(); pn = 1 - pb - pu
        obs = nx.triadic_census(nx.from_numpy_array(C.astype(int), create_using=nx.DiGraph))
        exp = {k: v * comb(ne, 3) for k, v in null_probs(pn, pu, pb).items()}
        res[p, s] = {k: (obs[k], exp[k]) for k in NAMES}
    print(f"{p} done {time.time() - t0:.0f} s", flush=True)

print("\nobserved / expected (Song pair null), mean over 5 seeds [min-max]; n = mean observed count")
for p in POINTS:
    print(f"\n{p}")
    for k in SHOW:
        r = np.array([res[p, s][k][0] / max(res[p, s][k][1], 1e-12) for s in range(5)])
        n = np.mean([res[p, s][k][0] for s in range(5)])
        e = np.mean([res[p, s][k][1] for s in range(5)])
        print(f"  {k:5s} {r.mean():7.2f}  [{r.min():6.2f}-{r.max():6.2f}]   n = {n:9.0f}  expected {e:9.1f}")
