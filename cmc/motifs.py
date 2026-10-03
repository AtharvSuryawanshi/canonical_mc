"""Weighted E/I connectivity motifs of a DaleRNN, with shuffle nulls.

All motifs are directed paths, ``W[pre, post]`` (the ``h @ W`` convention), on
synaptic magnitudes ``A = |W|``. A motif's intensity is the geometric mean of
its weights (Onnela et al. 2005, Phys Rev E 71:065103), so a two-synapse motif
``a -> b -> c`` has intensity ``sqrt(A[a,b] * A[b,c])``. Pass a 0/1 matrix
instead of magnitudes to get plain motif counts.

    back inhibition     E1 -> I -> E1        (= a reciprocal E<->I pair)
    lateral inhibition  E1 -> I -> E2, E1 != E2
    disinhibition       I1 -> I2 -> E,  I1 != I2
    I<->I reciprocity   I1 -> I2 -> I1

Raw means are not comparable across networks: a wiring budget shrinks all
weights. Each statistic is therefore compared with a null that keeps the
weights and destroys only the structure in question (see ``null_stats``):

* ``hub`` null -- permute the identity of the middle neuron between its
  incoming and outgoing synapses. Every weight and every block stays the same;
  only the pairing "how much input this I neuron gets" <-> "how much output it
  sends" is broken. Under any null that keeps each I neuron's own input and
  output weights, the *overall* chain strength is fixed (it is
  sum_i in_i * out_i), so over-representation of a chain type as a whole means
  exactly this pairing: hub interneurons.
* ``target`` null -- for each postsynaptic neuron, permute which presynaptic
  neurons its inputs come from (within the block). Every neuron keeps its input
  weights, hence its in-strength; *which* neuron sends what is random. Tests
  specificity: back vs lateral, same- vs different-cluster, reciprocity, and
  whether a few I neurons do most of the I -> I inhibition.
"""

import numpy as np


def blocks(W, n_e):
    """E/I blocks of |W|: EE, EI (E pre -> I post), IE, II."""
    A = np.abs(np.asarray(W, dtype=np.float64))
    return dict(EE=A[:n_e, :n_e], EI=A[:n_e, n_e:], IE=A[n_e:, :n_e], II=A[n_e:, n_e:])


def _offdiag_mean(M):
    n = M.shape[0]
    return (M.sum() - np.trace(M)) / (n * (n - 1))


def top_share(x, frac=0.2):
    """Fraction of sum(x) held by the top ``frac`` of entries."""
    x = np.sort(np.asarray(x))[::-1]
    k = max(1, int(round(frac * len(x))))
    return x[:k].sum() / max(x.sum(), 1e-30)


def motif_stats(EI, IE, II, e_labels=None):
    """Motif statistics from E->I, I->E, I->I magnitudes (or 0/1 masks).

    ``e_labels``: optional functional cluster per E neuron (-1 = unassigned);
    adds lateral inhibition between neurons of the same vs different clusters.
    """
    sEI, sIE, sII = np.sqrt(EI), np.sqrt(IE), np.sqrt(II)
    n_e, n_i = EI.shape
    M = sEI @ sIE                                  # (E1, E2): summed over the middle I
    out = {
        "back": float(np.mean(sEI * sIE.T)),       # E1 -> I -> E1
        "lateral": float(_offdiag_mean(M) / n_i),  # E1 -> I -> E2
        "disinhibition": float((sII @ sIE).sum() / (n_i * (n_i - 1) * n_e)),
        "recip_II": float(_offdiag_mean(sII * sII.T)),
        # Concentration of I -> I output on a few I neurons (top 20 %).
        "II_top20_share": float(top_share(II.sum(axis=1))),
    }
    out["back_over_lateral"] = out["back"] / max(out["lateral"], 1e-30)
    if e_labels is not None:
        lab = np.asarray(e_labels)
        ok = lab >= 0
        same = (lab[:, None] == lab[None, :]) & ok[:, None] & ok[None, :]
        np.fill_diagonal(same, False)
        diff = (lab[:, None] != lab[None, :]) & ok[:, None] & ok[None, :]
        if same.any() and diff.any():
            out["lateral_same"] = float(M[same].mean() / n_i)
            out["lateral_diff"] = float(M[diff].mean() / n_i)
            out["lateral_diff_over_same"] = out["lateral_diff"] / max(out["lateral_same"], 1e-30)
    return out


def _permute_columns(A, rng):
    """Independently permute the entries of each column (each target's inputs)."""
    idx = rng.random(A.shape).argsort(axis=0)
    return np.take_along_axis(A, idx, axis=0)


def _permute_columns_offdiag(A, rng):
    """Same for a square block with a structurally empty diagonal."""
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    vals = A.T[off.T].reshape(n, n - 1).T             # (n-1, n): column j without A[j, j]
    vals = _permute_columns(vals, rng)
    out = np.zeros_like(A)
    out.T[off.T] = vals.T.ravel()
    return out


HUB_KEYS = ("lateral_all", "disinhibition")
TARGET_KEYS = ("back", "back_over_lateral", "lateral_diff_over_same", "recip_II", "II_top20_share")


def null_stats(EI, IE, II, e_labels=None, n_perm=200, seed=0):
    """Observed statistics with their null mean / std, z-score and observed/null ratio.

    Returns {stat: dict(obs, null_mean, null_std, z, ratio)}. ``lateral_all``
    is E -> I -> E including back inhibition (the hub null fixes the split, so
    only the total is tested there).
    """
    rng = np.random.default_rng(seed)
    obs = motif_stats(EI, IE, II, e_labels)
    sEI, sIE, sII = np.sqrt(EI), np.sqrt(IE), np.sqrt(II)
    n_e, n_i = EI.shape
    obs["lateral_all"] = float((sEI @ sIE).mean() / n_i)

    samples = {k: [] for k in HUB_KEYS + TARGET_KEYS}
    for _ in range(n_perm):
        # hub null: re-pair each I neuron's inputs with another I neuron's outputs
        p = rng.permutation(n_i)
        samples["lateral_all"].append((sEI[:, p] @ sIE).mean() / n_i)
        p = rng.permutation(n_i)
        sII_p = sII[:, p]                          # I2 identity on the incoming side
        samples["disinhibition"].append((sII_p @ sIE).sum() / (n_i * (n_i - 1) * n_e))
        # target null: each target keeps its inputs, senders are shuffled
        st = motif_stats(_permute_columns(EI, rng), IE, _permute_columns_offdiag(II, rng), e_labels)
        for k in TARGET_KEYS:
            if k in st:
                samples[k].append(st[k])

    res = {}
    for k, vals in samples.items():
        if not vals or k not in obs:
            continue
        vals = np.asarray(vals)
        mu, sd = float(vals.mean()), float(vals.std())
        res[k] = dict(obs=obs[k], null_mean=mu, null_std=sd,
                      z=(obs[k] - mu) / sd if sd > 0 else np.nan,
                      ratio=obs[k] / mu if mu > 0 else np.nan)
    for k in ("lateral", "lateral_same", "lateral_diff"):   # descriptive, no null
        if k in obs:
            res[k] = dict(obs=obs[k])
    return res


# ---------------------------------------------------------------------------
# Group-to-group (e.g. task-cluster-to-task-cluster) connectivity and motifs

def _onehot(groups, n_groups):
    g = np.asarray(groups)
    G = np.zeros((len(g), n_groups))
    ok = g >= 0
    G[np.flatnonzero(ok), g[ok]] = 1.0
    return G


def _group_mean(M, Ga, Gb, exclude_diag):
    """Mean of M[a, b] over a in group i (rows), b in group j (cols); NaN if empty."""
    tot = Ga.T @ M @ Gb
    cnt = np.outer(Ga.sum(0), Gb.sum(0))
    if exclude_diag:                      # same neurons on both axes: drop a == b
        tot = tot - Ga.T @ (np.diag(np.diag(M)) @ Gb)
        cnt = cnt - Ga.T @ Gb
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


GROUP_STATS = ("E→E", "E→I", "I→E", "I→I", "lateral E→I→E", "disinhibition I→I→E")


def group_stats(A, n_e, groups, n_groups):
    """Group x group mean strength of direct connections and two-synapse motifs.

    ``A``: |W| (or a 0/1 mask), ``groups``: group index per neuron (-1 = none).
    Rows of every matrix are the *presynaptic* group (motifs: the first neuron),
    columns the postsynaptic (last) group. Motifs as in ``motif_stats``
    (geometric-mean intensity, averaged over the middle neuron).
    """
    b = blocks(A, n_e)
    G = _onehot(groups, n_groups)
    GE, GI = G[:n_e], G[n_e:]
    sEI, sIE, sII = np.sqrt(b["EI"]), np.sqrt(b["IE"]), np.sqrt(b["II"])
    n_i = b["II"].shape[0]
    return {
        "E→E": _group_mean(b["EE"], GE, GE, True),
        "E→I": _group_mean(b["EI"], GE, GI, False),
        "I→E": _group_mean(b["IE"], GI, GE, False),
        "I→I": _group_mean(b["II"], GI, GI, True),
        "lateral E→I→E": _group_mean(sEI @ sIE / n_i, GE, GE, True),
        "disinhibition I→I→E": _group_mean(sII @ sIE / (n_i - 1), GI, GE, False),
    }


def group_null_stats(A, n_e, groups, n_groups, n_perm=200, seed=0):
    """``group_stats`` against the target null (each neuron keeps its input
    weights, the senders are shuffled within each block).

    Returns {stat: dict(obs, ratio, z)} with (n_groups, n_groups) arrays.
    """
    rng = np.random.default_rng(seed)
    A = np.abs(np.asarray(A, dtype=np.float64))
    obs = group_stats(A, n_e, groups, n_groups)
    samples = {k: [] for k in obs}
    for _ in range(n_perm):
        P = np.zeros_like(A)
        P[:n_e, :n_e] = _permute_columns_offdiag(A[:n_e, :n_e], rng)
        P[:n_e, n_e:] = _permute_columns(A[:n_e, n_e:], rng)
        P[n_e:, :n_e] = _permute_columns(A[n_e:, :n_e], rng)
        P[n_e:, n_e:] = _permute_columns_offdiag(A[n_e:, n_e:], rng)
        for k, v in group_stats(P, n_e, groups, n_groups).items():
            samples[k].append(v)
    res = {}
    for k, v in obs.items():
        s = np.asarray(samples[k])
        mu, sd = s.mean(0), s.std(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[k] = dict(obs=v, ratio=v / mu, z=(v - mu) / sd)
    return res


# ---------------------------------------------------------------------------
# What learning changed: motifs in dW

def _project_l1_ball_np(z, radius):
    """Euclidean projection of nonnegative z onto {sum <= radius} (soft threshold)."""
    if z.sum() <= radius:
        return z
    u = np.sort(z.ravel())[::-1]
    css = np.cumsum(u) - radius
    j = np.arange(1, u.size + 1)
    rho = np.flatnonzero(u - css / j > 0).max() + 1
    return np.clip(z - css[rho - 1] / rho, 0.0, None)


def learned_change(W, W0, row_scale):
    """Per-synapse magnitude change that is *not* explained by an overall shrink.

    Both matrices are taken as E/I-normalised magnitudes (|W| / row_scale, the
    units of the wiring cost). The init is first brought to the trained
    network's total wiring cost the way the budget does it: soft-thresholded
    (L1 projection) if it has more, scaled up if it has less. Then
    ``D = |W| - matched(|W0|)``: D > 0 means learning strengthened the synapse
    beyond the uniform shrink, D < 0 weakened or pruned it. Diagonal is 0.
    """
    rs = np.asarray(row_scale, dtype=np.float64)[:, None]
    M = np.abs(np.asarray(W, dtype=np.float64)) / rs
    M0 = np.abs(np.asarray(W0, dtype=np.float64)) / rs
    np.fill_diagonal(M, 0.0)
    np.fill_diagonal(M0, 0.0)
    target = M.sum()
    P = _project_l1_ball_np(M0, target) if M0.sum() > target else M0 * (target / M0.sum())
    return M - P


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _offdiag_pairs(X):
    iu = np.triu_indices(X.shape[0], 1)
    return X[iu], X.T[iu]


def _same_task_pref(X, ga, gb, square):
    """(mean X within group - mean X between groups) / std X, over assigned neurons."""
    ok = (ga[:, None] >= 0) & (gb[None, :] >= 0)
    same = (ga[:, None] == gb[None, :]) & ok
    diff = (ga[:, None] != gb[None, :]) & ok
    if square:
        np.fill_diagonal(same, False)
        np.fill_diagonal(diff, False)
    if not same.any() or not diff.any():
        return np.nan
    return float((X[same].mean() - X[diff].mean()) / X[same | diff].std())


DELTA_STATS = (
    "recip E↔E", "recip E↔I (back inhibition)", "recip I↔I",
    "hub lateral (I in-change vs out-to-E change)", "hub disinhibition (I in-from-I vs out-to-E)",
    "same-task E→E", "same-task E→I", "same-task I→E", "same-task I→I",
)


def delta_stats(D, n_e, groups=None):
    """Correlation-type motif statistics of a signed change matrix D[pre, post].

    * reciprocity: corr(D[a,b], D[b,a]) -- do the two directions of a pair
      change together? (E<->I reciprocity is back inhibition E1 -> I -> E1.)
    * hub: corr over interneurons of the total change of their inputs and of
      their outputs to E -- interneurons that gain input also gain output?
    * same-task: within-task minus between-task mean change, in std units
      (needs ``groups``, e.g. the preferred task of each neuron's cluster).
    """
    EE, EI, IE, II = D[:n_e, :n_e], D[:n_e, n_e:], D[n_e:, :n_e], D[n_e:, n_e:]
    out = {
        DELTA_STATS[0]: _corr(*_offdiag_pairs(EE)),
        DELTA_STATS[1]: _corr(EI.ravel(), IE.T.ravel()),
        DELTA_STATS[2]: _corr(*_offdiag_pairs(II)),
        DELTA_STATS[3]: _corr(EI.sum(0), IE.sum(1)),
        DELTA_STATS[4]: _corr(II.sum(0), IE.sum(1)),
    }
    if groups is not None:
        g = np.asarray(groups)
        gE, gI = g[:n_e], g[n_e:]
        out[DELTA_STATS[5]] = _same_task_pref(EE, gE, gE, True)
        out[DELTA_STATS[6]] = _same_task_pref(EI, gE, gI, False)
        out[DELTA_STATS[7]] = _same_task_pref(IE, gI, gE, False)
        out[DELTA_STATS[8]] = _same_task_pref(II, gI, gI, True)
    return out


def delta_null_stats(D, n_e, groups=None, n_perm=200, seed=0):
    """``delta_stats`` against the target null: every neuron keeps the changes of
    its inputs, which presynaptic neuron they come from is shuffled within each
    E/I block. Returns {stat: dict(obs, null_mean, null_std, z)}."""
    rng = np.random.default_rng(seed)
    obs = delta_stats(D, n_e, groups)
    samples = {k: [] for k in obs}
    for _ in range(n_perm):
        P = np.empty_like(D)
        P[:n_e, :n_e] = _permute_columns_offdiag(D[:n_e, :n_e], rng)
        P[:n_e, n_e:] = _permute_columns(D[:n_e, n_e:], rng)
        P[n_e:, :n_e] = _permute_columns(D[n_e:, :n_e], rng)
        P[n_e:, n_e:] = _permute_columns_offdiag(D[n_e:, n_e:], rng)
        for k, v in delta_stats(P, n_e, groups).items():
            samples[k].append(v)
    res = {}
    for k, v in obs.items():
        s = np.asarray(samples[k], dtype=float)
        mu, sd = np.nanmean(s), np.nanstd(s)
        res[k] = dict(obs=v, null_mean=float(mu), null_std=float(sd),
                      z=(v - mu) / sd if sd > 0 else np.nan)
    return res
