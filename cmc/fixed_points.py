"""Fixed points of a trained DaleRNN under a constant input (Sussillo & Barak 2013; as in Driscoll et al. 2024).

Update: h' = (1 - a) h + a ReLU(g),  g = h @ W_rec + u @ W_in + b   (W_rec[pre, post], as in DaleRNN.simulate).
A fixed point satisfies h = ReLU(g); the leak a drops out. Because ReLU is piecewise linear, once the set S of
active units is known the fixed point is the solution of a linear system,

    h_S = W_rec[S, S]^T h_S + c_S,   h_rest = 0,   c = u @ W_in + b,

and it is exact if it is self-consistent (g > 0 on S, g <= 0 elsewhere). So: (1) minimise the speed
q(h) = 0.5 |F(h) - h|^2 from many starting states (Adam), (2) solve the linear system for the active set of each
candidate and keep it only if self-consistent, (3) remove duplicates. The Jacobian of the update at a fixed point is
exact: J = (1 - a) I + a diag(S) W_rec^T; the fixed point is stable if max |eig(J)| < 1.
"""
import numpy as np
import torch


def parts(model):
    """(W_rec[pre, post], W_in, b, alpha) as float64 numpy."""
    with torch.no_grad():
        W = model.effective_w_rec().double().cpu().numpy()
        W_in = model.w_in.double().cpu().numpy()
        b = model.bias.double().cpu().numpy()
    return W, W_in, b, float(model.alpha)


def speed_minimise(model, u, h0, n_iter=3000, lr=1e-2):
    """Minimise q(h) = 0.5 |F(h) - h|^2 from starting states h0 (n, N). Returns (h, q) as numpy."""
    W, W_in, b, a = parts(model)
    W = torch.as_tensor(W)
    c = torch.as_tensor(np.asarray(u, float) @ W_in + b)
    h = torch.tensor(np.asarray(h0, float), requires_grad=True)
    opt = torch.optim.Adam([h], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        q = 0.5 * (a * (torch.relu(h @ W + c) - h)).pow(2).sum(1)
        q.sum().backward()
        opt.step()
    with torch.no_grad():
        q = 0.5 * (a * (torch.relu(h @ W + c) - h)).pow(2).sum(1)
    return h.detach().numpy(), q.numpy()


def exact(model, u, h, max_iter=20):
    """Exact fixed point from a candidate h by solving the linear system of its active set. Returns (h*, ok)."""
    W, W_in, b, _ = parts(model)
    c = np.asarray(u, float) @ W_in + b
    S = (h @ W + c) > 0
    for _ in range(max_iter):
        x = np.zeros_like(c)
        if S.any():
            A = np.eye(S.sum()) - W[np.ix_(S, S)].T
            x[S] = np.linalg.lstsq(A, c[S], rcond=None)[0]
        g = x @ W + c
        S_new = g > 0
        if np.array_equal(S_new, S):
            ok = np.all(x[S] > 0) and np.allclose(np.maximum(g, 0), x, atol=1e-9)
            return x, bool(ok)
        S = S_new
    return x, False


def jacobian(model, u, h):
    W, W_in, b, a = parts(model)
    m = ((h @ W + np.asarray(u, float) @ W_in + b) > 0).astype(float)
    return (1 - a) * np.eye(len(h)) + a * m[:, None] * W.T


def max_eig(model, u, h):
    return float(np.abs(np.linalg.eigvals(jacobian(model, u, h))).max())


def relax(model, u, h0, n_iter=3000, lr=1e-2, speed_tol=1e-3):
    """Fixed and slow points reached from each start, one per start (no deduplication), as Driscoll's finder reports.

    Each start is pushed down its speed |F(h) - h| (per update step); the result is replaced by the exact fixed point
    of its active set when that is self-consistent and nearby. Returns dict per start: h, speed, exact (bool),
    slow (speed < speed_tol or exact), max_eig.
    """
    h, _ = speed_minimise(model, u, h0, n_iter, lr)
    h = np.maximum(h, 0)
    W, W_in, b, a = parts(model)
    c = np.asarray(u, float) @ W_in + b
    out = dict(h=[], speed=[], exact=[], slow=[], max_eig=[])
    for cand in h:
        x, ok = exact(model, u, cand)
        ok = ok and np.linalg.norm(x - cand) < 0.05 * max(1.0, np.linalg.norm(cand))
        p = x if ok else cand
        s = float(np.linalg.norm(a * (np.maximum(p @ W + c, 0) - p)))
        out["h"].append(p); out["speed"].append(s); out["exact"].append(bool(ok))
        out["slow"].append(bool(ok or s < speed_tol)); out["max_eig"].append(max_eig(model, u, p))
    return {k: np.array(v) for k, v in out.items()}


def find(model, u, h0, n_iter=3000, lr=1e-2, dedupe_tol=1e-3):
    """Exact, unique fixed points reached from h0. Returns dict(h (k, N), max_eig (k,), eigs list, n_found)."""
    h, _ = speed_minimise(model, u, h0, n_iter, lr)
    pts = []
    for cand in h:
        x, ok = exact(model, u, np.maximum(cand, 0))
        if ok and not any(np.linalg.norm(x - p) < dedupe_tol * max(1.0, np.linalg.norm(p)) for p in pts):
            pts.append(x)
    H = np.array(pts) if pts else np.zeros((0, h.shape[1]))
    eigs = [np.linalg.eigvals(jacobian(model, u, x)) for x in H]
    return dict(h=H, max_eig=np.array([np.abs(e).max() for e in eigs]), eigs=eigs, n_found=len(H))
