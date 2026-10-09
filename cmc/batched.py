"""Train a population of DaleRNNs at once, stacked into the same tensors.

One 256-unit RNN step is a few tiny kernels, so a single network leaves the GPU
almost idle: its speed is set by kernel-launch latency, host syncs and CPU trial
generation, not by compute. ``BatchedDaleRNN`` stacks P networks' weights
(``w_raw`` is ``[P, N, N]``) and runs them with batched matmuls, so P networks
cost about what one costs. Used by ``cmc.moo --batched`` and
``cmc.moo_zoom --batched``; the single-network path (``train_cog.train``) is
untouched.

What is the same as ``train_cog.train`` for every network: the init (each one
is built by ``runner.make_fresh_model`` with its own seed), the model, the loss
(``masked_mse`` + rate / wiring terms, fixed lambdas or budgets), the budget
enforcement (``BudgetConstraint`` math, L1-ball projection of W_rec), Adam and
per-network gradient clipping. Gradients never mix: the loss is a sum of
per-network losses.

What differs: all P networks see the **same trial stream** (one batch per step,
generated once), and training does no per-task logging eval. So a batched
network is not bit-identical to the single-run network with the same seed; it
is an equally valid sample. Compare batched with batched, single with single.
Only DaleRNN, projection-mode wiring budgets and ``conn_target="w_rec"``.

A network whose loss or gradient goes non-finite is frozen at its last finite
weights (the others carry on); ``history["diverged_step"]`` records when, and
its metrics are NaN.

    python -m cmc.batched --bench --task-battery all --pop 1,4,24 --steps 30   # speed per P
"""

import argparse
import copy
import math
import time

import numpy as np
import torch

from cmc.network import DaleRNN, _inv_softplus_torch
from cmc.task import default_config, generate_mixed_trials, generate_trials
from cmc.train_cog import (
    BUDGET_DEFAULTS,
    BudgetConstraint,
    _reg_opts,
    evaluate_pareto_metrics,
    trial_to_tensors,
)

PARAM_NAMES = ("w_in", "w_raw", "bias", "w_out", "b_out")


class BatchedDaleRNN(torch.nn.Module):
    """P DaleRNNs with identical shapes, stacked along a leading dim."""

    def __init__(self, models):
        super().__init__()
        if not models or not all(isinstance(m, DaleRNN) for m in models):
            raise ValueError("BatchedDaleRNN: needs a non-empty list of DaleRNN")
        m0 = models[0]
        for m in models[1:]:
            same = (m.n_input, m.n_rnn, m.n_output, m.alpha, m.n_e, m.activation_name, m.sigma,
                    m.prune_eps) == (m0.n_input, m0.n_rnn, m0.n_output, m0.alpha, m0.n_e,
                                     m0.activation_name, m0.sigma, m0.prune_eps)
            if not same or not torch.equal(m.sign_vector, m0.sign_vector):
                raise ValueError("BatchedDaleRNN: all networks must share architecture and E/I signs")
        if m0.prune_eps > 0.0:
            raise ValueError("BatchedDaleRNN: prune_eps must be 0")
        self.models = models  # plain list: not registered, written back by unstack()
        self.P = len(models)
        self.n_input, self.n_rnn, self.n_output = m0.n_input, m0.n_rnn, m0.n_output
        self.n_e, self.n_i = m0.n_e, m0.n_i
        self.alpha, self.sigma = m0.alpha, m0.sigma
        self._act = m0._act
        for name in PARAM_NAMES:
            stacked = torch.stack([getattr(m, name).detach() for m in models]).clone()
            setattr(self, name, torch.nn.Parameter(stacked))
        self.register_buffer("sign_vector", m0.sign_vector.clone())
        self.register_buffer("no_autapse", m0.no_autapse.clone())

    def ei_row_scale(self):
        scale = torch.ones(self.n_rnn, device=self.w_raw.device, dtype=self.w_raw.dtype)
        if self.n_i > 0:
            scale[self.n_e:] = self.n_e / self.n_i
        return scale

    def effective_w_rec(self):
        """[P, N, N]: softplus magnitudes, presynaptic sign on rows, no autapses."""
        mag = torch.nn.functional.softplus(self.w_raw) * self.no_autapse
        return mag * self.sign_vector[:, None]

    def simulate(self, x, noise_level=0.0):
        """x: (B, n_input, T), shared by all networks.

        Returns r_hist (P, B, T, N) and y_hat (P, B, T, n_output). Same update as
        DaleRNN.simulate; W_rec and the input drive are computed once per call.
        """
        x = x.to(device=self.w_raw.device, dtype=self.w_raw.dtype)
        B, _, T = x.shape
        w_rec = self.effective_w_rec()
        drive = torch.einsum("bit,pin->tpbn", x, self.w_in) + self.bias[None, :, None, :]
        h = x.new_zeros(self.P, B, self.n_rnn)
        hs = []
        for t in range(T):
            gate = drive[t] + torch.bmm(h, w_rec)
            if noise_level > 0.0:
                gate = gate + torch.randn_like(h) * (noise_level * self.sigma)
            h = (1.0 - self.alpha) * h + self.alpha * self._act(gate)
            hs.append(h)
        r_hist = torch.stack(hs, dim=2)                               # (P, B, T, N)
        y_hat = torch.sigmoid(
            torch.einsum("pbte,peo->pbto", r_hist[..., : self.n_e], self.w_out)
            + self.b_out[:, None, None, :]
        )
        return r_hist, y_hat

    @torch.no_grad()
    def unstack(self):
        """Copy the trained weights back into the P DaleRNNs; returns them."""
        for p, m in enumerate(self.models):
            for name in PARAM_NAMES:
                getattr(m, name).copy_(getattr(self, name)[p])
        return self.models


# --------------------------------------------------------------------------- costs, [P] each

def masked_mse_batched(output, y, c_mask, per_trial=True):
    """train_cog.masked_mse per network. output (P,B,T,O); y, c_mask (B,T,O)."""
    err = c_mask * (output - y).square()
    if not per_trial:
        return err.flatten(1).mean(dim=1)
    num = err.flatten(2).sum(dim=2)
    den = c_mask.flatten(1).sum(dim=1).clamp_min(1e-8)
    return (num / den).mean(dim=1)


def rate_reg_batched(r_hist, bmodel, kind="l2", inh_scale=1.0):
    """train_cog.rate_reg (population-weighted) per network. r_hist (P,B,T,N)."""
    if kind == "l1":
        per_unit = r_hist.abs()
    elif kind == "l2":
        per_unit = r_hist.square()
    else:
        raise ValueError(f"rate_reg: kind must be 'l1' or 'l2', got {kind!r}")
    n_e, n_rnn = bmodel.n_e, bmodel.n_rnn
    if bmodel.n_i == 0:
        return per_unit.flatten(1).mean(dim=1)
    cost_e = per_unit[..., :n_e].flatten(1).mean(dim=1)
    cost_i = per_unit[..., n_e:].flatten(1).mean(dim=1)
    return (n_e / n_rnn) * cost_e + inh_scale * (bmodel.n_i / n_rnn) * cost_i


def _w_rec_magnitude_batched(bmodel, inh_scale=1.0):
    """train_cog._w_rec_magnitude per network: (P,N,N) magnitudes and the (N,N) mask."""
    mag = bmodel.effective_w_rec().abs() / bmodel.ei_row_scale()[None, :, None]
    mask = bmodel.no_autapse
    if inh_scale != 1.0:
        row_w = torch.ones(bmodel.n_rnn, device=mag.device, dtype=mag.dtype)
        row_w[bmodel.n_e:] = inh_scale
        mask = mask * row_w[:, None]
    return mag, mask


def connectivity_reg_batched(bmodel, kind="l1", inh_scale=1.0):
    """train_cog.connectivity_reg (target w_rec) per network."""
    mag, mask = _w_rec_magnitude_batched(bmodel, inh_scale)
    term = mag if kind == "l1" else mag.square()
    return (term * mask).flatten(1).sum(dim=1) / mask.sum().clamp_min(1.0)


def project_l1_ball_batched(z, radius):
    """train_cog._project_l1_ball for each row of z (P, M); radius (P,), inf = no budget.

    Returns (z_new, projected): rows already inside the ball are returned as is.
    """
    total = z.sum(dim=1)
    projected = total > radius * (1.0 + 1e-5)
    r = torch.where(projected, radius, total)  # keeps inf out of the arithmetic
    u, _ = torch.sort(z, dim=1, descending=True)
    css = torch.cumsum(u, dim=1) - r[:, None]
    j = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype)
    cond = (u - css / j) > 0
    rho = (cond * j).amax(dim=1).clamp_min(1).long()                  # largest j where cond holds
    tau = css.gather(1, (rho - 1)[:, None]).squeeze(1) / rho.to(z.dtype)
    z_new = torch.clamp(z - tau[:, None], min=0.0)
    return torch.where(projected[:, None], z_new, z), projected


@torch.no_grad()
def project_w_rec_to_budget_batched(bmodel, budget, inh_scale=1.0):
    """train_cog.project_w_rec_to_budget per network; budget (P,), inf = no wiring budget."""
    mag, mask = _w_rec_magnitude_batched(bmodel, inh_scale)
    P = bmodel.P
    z = (mag * mask).reshape(P, -1)
    z_new, projected = project_l1_ball_batched(z, budget * mask.sum().clamp_min(1.0))
    z_new = z_new.reshape(mag.shape)
    counted = mask > 0
    new_abs = torch.where(counted, z_new / mask.clamp_min(1e-12), mag) * bmodel.ei_row_scale()[None, :, None]
    new_raw = _inv_softplus_torch(new_abs).to(bmodel.w_raw.dtype)
    bmodel.w_raw.copy_(torch.where(projected[:, None, None], new_raw, bmodel.w_raw))


class BatchedBudget:
    """``train_cog.BudgetConstraint`` for P networks at once, as tensors (no host syncs).

    ``budget`` (P,) with NaN = no budget for that network (its penalty is 0).
    """

    def __init__(self, budget, n_steps, device, lr, kp, ramp, rho, lambda_init, ema, dtype=torch.float64):
        b = torch.as_tensor(budget, dtype=dtype, device=device)
        self.active = torch.isfinite(b)
        if bool((b[self.active] <= 0).any()):
            raise ValueError(f"BatchedBudget: budgets must be > 0, got {budget}")
        self.budget = torch.where(self.active, b, torch.ones_like(b))
        self.lr, self.kp, self.rho, self.ema = float(lr), float(kp), float(rho), float(ema)
        self.ramp_steps = int(round(float(ramp) * n_steps)) if ramp and n_steps else 0
        lam0 = float(np.log(np.clip(lambda_init, BudgetConstraint.LAM_MIN, BudgetConstraint.LAM_MAX)))
        self.integral = torch.full_like(self.budget, lam0)
        self.log_lam = self.integral.clone()
        self.v_ema = None
        self.start_cost = None
        self.step = 0
        self.last_cost = torch.full_like(self.budget, float("nan"))

    @property
    def lam(self):
        return torch.exp(self.log_lam)

    def current_budget(self):
        if self.step >= self.ramp_steps or self.start_cost is None:
            return self.budget
        frac = self.step / self.ramp_steps
        ramped = self.start_cost ** (1.0 - frac) * self.budget ** frac
        return torch.where(self.start_cost <= self.budget, self.budget, ramped)

    def set_start(self, cost):
        if self.start_cost is None:
            self.start_cost = cost.detach().to(self.budget.dtype)

    def penalty(self, cost):
        """(P,) loss terms; ``cost`` (P,) differentiable. 0 where there is no budget."""
        self.last_cost = cost.detach().to(self.budget.dtype)
        self.set_start(cost)
        budget_t = self.current_budget().to(cost.dtype)
        v = cost / budget_t - 1.0
        term = self.lam.to(cost.dtype) * cost + 0.5 * self.rho * budget_t * torch.relu(v).square()
        return torch.where(self.active, term, torch.zeros_like(term))

    def update(self):
        v = self.last_cost / self.current_budget() - 1.0
        self.v_ema = v if self.v_ema is None else self.ema * self.v_ema + (1 - self.ema) * v
        e = self.v_ema.clamp(-1.0, 1.0)
        lo, hi = math.log(BudgetConstraint.LAM_MIN), math.log(BudgetConstraint.LAM_MAX)
        self.integral = (self.integral + self.lr * e).clamp(lo, hi)
        self.log_lam = (self.integral + self.kp * e).clamp(lo, hi)
        self.step += 1


# --------------------------------------------------------------------------- training

def _spec_value(spec, key):
    v = spec.get(key)
    return float("nan") if v is None else float(v)


def _clip_and_mask_grads(bmodel, grad_clip, alive):
    """Per-network clip_grad_norm_ (same formula), and zero the grads of dead networks."""
    params = [getattr(bmodel, n) for n in PARAM_NAMES]
    sq = sum(p.grad.flatten(1).square().sum(dim=1) for p in params)
    norm = sq.sqrt()
    alive = alive & torch.isfinite(norm)
    coef = torch.ones_like(norm)
    if grad_clip is not None and grad_clip > 0:
        coef = (grad_clip / (norm + 1e-6)).clamp(max=1.0)
    for p in params:
        shape = (-1,) + (1,) * (p.grad.ndim - 1)
        p.grad.copy_(torch.where(alive.view(shape), p.grad * coef.view(shape), torch.zeros_like(p.grad)))
    return alive


def train_batched(
    bmodel,
    config,
    active_tasks,
    specs,
    n_steps=200,
    batch_size=32,
    lr=1e-3,
    noise_level=0.0,
    grad_clip=1.0,
    log_every=20,
    mixed_batch=True,
    loss_per_trial=True,
    budget_lr=BUDGET_DEFAULTS["budget_lr"],
    budget_kp=BUDGET_DEFAULTS["budget_kp"],
    budget_ramp=BUDGET_DEFAULTS["budget_ramp"],
    budget_rho=BUDGET_DEFAULTS["budget_rho"],
    budget_lambda_init=BUDGET_DEFAULTS["budget_lambda_init"],
    budget_ema=BUDGET_DEFAULTS["budget_ema"],
    conn_budget_mode=BUDGET_DEFAULTS["conn_budget_mode"],
    callback_steps=(),
    step_callback=None,
    show_progress=False,
    trial_source=None,
    **reg_overrides,
):
    """Train the P networks of ``bmodel`` together; returns one history per network.

    ``specs[p]`` holds network p's ``lambda_rate``, ``lambda_connectivity``,
    ``rate_budget``, ``conn_budget`` (None = absent). ``config["rng"]`` drives the
    one shared trial stream. ``step_callback(step, histories)`` runs at the steps
    in ``callback_steps``, after the weights are written back into
    ``bmodel.models`` (e.g. to save intermediate networks). ``trial_source(step)``
    replaces trial generation (tests). The budget flags are shared by all networks.
    """
    reg = _reg_opts(**reg_overrides)
    P = bmodel.P
    if len(specs) != P:
        raise ValueError(f"train_batched: {len(specs)} specs for {P} networks")
    device, dtype = bmodel.w_raw.device, bmodel.w_raw.dtype
    lam_rate = torch.tensor([float(s.get("lambda_rate") or 0.0) for s in specs], device=device, dtype=dtype)
    lam_conn = torch.tensor([float(s.get("lambda_connectivity") or 0.0) for s in specs], device=device, dtype=dtype)
    rate_b = [_spec_value(s, "rate_budget") for s in specs]
    conn_b = [_spec_value(s, "conn_budget") for s in specs]
    has_rate = [math.isfinite(v) for v in rate_b]
    has_conn = [math.isfinite(v) for v in conn_b]
    for s, hr, hc in zip(specs, has_rate, has_conn):
        if (hr and s.get("lambda_rate")) or (hc and s.get("lambda_connectivity")):
            raise ValueError("train_batched: give a lambda or a budget per cost, not both")
    if (any(has_conn) or bool(lam_conn.any())) and reg["conn_target"] != "w_rec":
        raise ValueError("train_batched: only conn_target='w_rec'")
    if any(has_conn):
        if conn_budget_mode != "projection":
            raise ValueError("train_batched: wiring budgets only in projection mode")
        if reg["conn_kind"] != "l1":
            raise ValueError("train_batched: projection needs conn_kind='l1'")
    budget_kw = dict(lr=budget_lr, kp=budget_kp, ramp=budget_ramp, rho=budget_rho,
                     lambda_init=budget_lambda_init, ema=budget_ema)
    rate_con = BatchedBudget(rate_b, n_steps, device, **budget_kw) if any(has_rate) else None
    conn_proj = BatchedBudget(conn_b, n_steps, device, **budget_kw) if any(has_conn) else None
    use_lam_rate, use_lam_conn = bool(lam_rate.any()), bool(lam_conn.any())

    params = [getattr(bmodel, n) for n in PARAM_NAMES]
    optimizer = torch.optim.Adam(params, lr=lr)
    alive = torch.ones(P, dtype=torch.bool, device=device)
    diverged_step = torch.zeros(P, dtype=torch.long, device=device)

    # Per-step logs stay on the device and are copied to the host in blocks.
    log = {"loss": [], "rate_cost": [], "rate_lambda": [], "rate_budget_t": [],
           "conn_cost": [], "conn_budget_t": []}
    host = {k: [] for k in log}

    def flush():
        for k, buf in log.items():
            if buf:
                host[k].append(torch.stack(buf).double().cpu().numpy())
                buf.clear()

    nan_p = torch.full((P,), float("nan"), device=device, dtype=torch.float64)
    callback_steps = set(callback_steps)
    t_start = time.perf_counter()
    for step in range(1, n_steps + 1):
        if trial_source is not None:
            x, y, c_mask = trial_source(step)
        else:
            if mixed_batch and len(active_tasks) > 1:
                trial = generate_mixed_trials(active_tasks, config, batch_size, noise_on=True)
            else:
                rule = str(config["rng"].choice(active_tasks))
                trial = generate_trials(rule, config, batch_size, noise_on=True)
            x, y, c_mask, _ = trial_to_tensors(trial, device)
        y, c_mask = y.to(dtype), c_mask.to(dtype)

        optimizer.zero_grad()
        r_hist, y_hat = bmodel.simulate(x, noise_level=noise_level)
        loss = masked_mse_batched(y_hat, y, c_mask, per_trial=loss_per_trial)
        if use_lam_rate or rate_con is not None:
            rcost = rate_reg_batched(r_hist, bmodel, kind=reg["rate_kind"], inh_scale=reg["rate_inh_scale"])
            if use_lam_rate:
                loss = loss + lam_rate * rcost
            if rate_con is not None:
                loss = loss + rate_con.penalty(rcost)
        if use_lam_conn:
            loss = loss + lam_conn * connectivity_reg_batched(bmodel, kind=reg["conn_kind"],
                                                              inh_scale=reg["conn_inh_scale"])
        loss_alive = torch.where(alive, loss, torch.zeros_like(loss))
        loss_alive.sum().backward()
        prev = [p.detach().clone() for p in params]
        new_alive = _clip_and_mask_grads(bmodel, grad_clip, alive & torch.isfinite(loss))
        optimizer.step()
        with torch.no_grad():
            if conn_proj is not None:
                conn_proj.set_start(connectivity_reg_batched(bmodel, inh_scale=reg["conn_inh_scale"]))
                budget_t = conn_proj.current_budget()
                b = torch.where(conn_proj.active, budget_t, torch.full_like(budget_t, float("inf")))
                project_w_rec_to_budget_batched(bmodel, b.to(dtype), inh_scale=reg["conn_inh_scale"])
                ccost = connectivity_reg_batched(bmodel, inh_scale=reg["conn_inh_scale"])
                conn_proj.step += 1
                log["conn_budget_t"].append(budget_t)
                log["conn_cost"].append(ccost)
            # Freeze networks that went non-finite at their last finite weights.
            for p, old in zip(params, prev):
                shape = (-1,) + (1,) * (p.ndim - 1)
                p.copy_(torch.where(new_alive.view(shape), p, old))
            diverged_step = torch.where(alive & ~new_alive, torch.full_like(diverged_step, step), diverged_step)
            alive = new_alive
        log["loss"].append(loss.detach())
        if rate_con is not None:
            log["rate_budget_t"].append(rate_con.current_budget())
            rate_con.update()
            log["rate_lambda"].append(rate_con.lam)
            log["rate_cost"].append(rate_con.last_cost)

        if step % log_every == 0 or step == n_steps or step in callback_steps:
            flush()
            if show_progress:
                print(f"step {step}/{n_steps}  {time.perf_counter() - t_start:.0f}s  "
                      f"alive {int(alive.sum())}/{P}", flush=True)
        if step_callback is not None and step in callback_steps:
            bmodel.unstack()
            step_callback(step, _histories(host, specs, has_rate, has_conn, diverged_step, conn_budget_mode,
                                           budget_kw, reg, loss_per_trial))
    flush()
    bmodel.unstack()
    return _histories(host, specs, has_rate, has_conn, diverged_step, conn_budget_mode, budget_kw, reg,
                      loss_per_trial)


def _histories(host, specs, has_rate, has_conn, diverged_step, conn_budget_mode, budget_kw, reg, loss_per_trial):
    """Per-network history dicts with the keys train_cog.train produces."""
    cat = {k: (np.concatenate(v) if v else None) for k, v in host.items()}
    n = 0 if cat["loss"] is None else len(cat["loss"])
    div = diverged_step.cpu().numpy()
    out = []
    for p, spec in enumerate(specs):
        h = {"loss": [] if n == 0 else cat["loss"][:, p].tolist(), "step": list(range(1, n + 1)),
             "eval_step": [], "frac_silent": [], "frac_saturated": [], "per_task": {},
             "reg": dict(reg), "loss_per_trial": bool(loss_per_trial), "batched": True,
             "diverged_step": int(div[p]) or None}
        if has_rate[p] or has_conn[p]:
            b = {"rate_budget": spec.get("rate_budget"), "conn_budget": spec.get("conn_budget"),
                 "conn_budget_mode": conn_budget_mode, **budget_kw,
                 "rate": {"lambda": [], "cost": [], "budget_t": []},
                 "conn": {"lambda": [], "cost": [], "budget_t": []}}
            if has_rate[p] and cat["rate_cost"] is not None:
                b["rate"] = {"lambda": cat["rate_lambda"][:, p].tolist(),
                             "cost": cat["rate_cost"][:, p].tolist(),
                             "budget_t": cat["rate_budget_t"][:, p].tolist()}
            if has_conn[p] and cat["conn_cost"] is not None:
                b["conn"] = {"lambda": [float("nan")] * n,
                             "cost": cat["conn_cost"][:, p].tolist(),
                             "budget_t": cat["conn_budget_t"][:, p].tolist()}
            h["budget"] = b
        out.append(h)
    return out


def train_and_evaluate_batched(args, specs, active_tasks, device, noise_level, eval_seeds, reg_opts,
                               data_seed=None, callback_steps=(), step_callback=None):
    """Batched ``runner.train_and_evaluate``: one result per spec, same shape.

    ``specs[p]``: ``seed`` plus ``lambda_rate`` / ``lambda_connectivity`` /
    ``rate_budget`` / ``conn_budget`` (None = absent). Budget settings come from
    ``args`` (``--budget-*``). Each network's config is the one a single run with
    its seed would use (so eval and checkpoints are identical in form); the shared
    trial stream is seeded with ``data_seed`` (default: the first spec's seed).
    ``step_callback(step, models, configs, histories)`` runs at ``callback_steps``.

    Returns a list of (model, config, history, metrics).
    """
    from cmc.runner import make_fresh_model

    if getattr(args, "freeze", ""):
        raise ValueError("--freeze is not implemented for --batched; run without --batched")
    configs =[default_config(n_eachring=args.n_eachring, seed=int(s["seed"]), easy_task=True) for s in specs]
    models = [make_fresh_model(args, cfg, device, seed=int(s["seed"])) for s, cfg in zip(specs, configs)]
    bmodel = BatchedDaleRNN(models).to(device)
    data_seed = int(specs[0]["seed"] if data_seed is None else data_seed)
    train_config = default_config(n_eachring=args.n_eachring, seed=data_seed, easy_task=True)
    torch.manual_seed(data_seed)  # recurrent noise
    budget_kw = {k: getattr(args, k) for k in ("budget_lr", "budget_kp", "budget_ramp", "budget_rho",
                                                "budget_lambda_init", "budget_ema", "conn_budget_mode")
                 if hasattr(args, k)}
    cb = None
    if step_callback is not None:
        def cb(step, histories):
            step_callback(step, models, configs, histories)
    histories = train_batched(
        bmodel, train_config, active_tasks, specs,
        n_steps=args.steps, batch_size=args.batch_size, lr=args.lr, noise_level=noise_level,
        log_every=args.log_every, loss_per_trial=args.loss_per_trial,
        callback_steps=callback_steps, step_callback=cb, **budget_kw, **reg_opts,
    )
    results = []
    for model, config, history in zip(models, configs, histories):
        model.eval()
        metrics = evaluate_pareto_metrics(
            model, config, active_tasks, device, eval_seeds=eval_seeds,
            batch_size=args.eval_batch_size, noise_level=args.eval_noise_level,
            input_noise=args.eval_input_noise, easy_task=True, **reg_opts,
        )
        if history["diverged_step"] is not None:
            metrics = {k: float("nan") for k in metrics}
        results.append((model, config, history, metrics))
    return results


def add_batched_args(parser):
    """Opt-in batched training flags (cmc.moo, cmc.moo_zoom). Off by default."""
    parser.add_argument(
        "--batched", action="store_true",
        help="Train networks together in one process, stacked into the same tensors "
        "(cmc.batched): much faster on a GPU. All networks of a batch share one trial "
        "stream, so they are not bit-identical to the default one-network-at-a-time runs.",
    )
    parser.add_argument("--batch-pop", type=int, default=24,
                        help="With --batched: max networks per batch.")
    parser.add_argument(
        "--devices", type=str, default=None,
        help='With --batched and --workers > 1: comma-separated devices assigned to the '
        'worker batches round-robin, e.g. "cuda:0,cuda:1". Default: --device.',
    )
    return parser


def chunked(items, size):
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def args_on_device(args, device):
    """Copy of ``args`` with ``--device`` replaced (for one worker per GPU)."""
    out = copy.copy(args)
    out.device = device
    return out


def parse_devices(text):
    return [d.strip() for d in str(text).split(",") if d.strip()] if text else []


# --------------------------------------------------------------------------- benchmark

def bench(argv=None):
    from cmc.runner import add_common_args, resolve_run_settings

    parser = argparse.ArgumentParser(description="Time one training step for P networks at once.")
    add_common_args(parser, n_seeds=False)
    parser.add_argument("--pop", type=str, default="1,4,24")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args(argv)
    if args.steps == 1000:
        args.steps = 20
    active_tasks, battery, device, noise_level, _, reg_opts = resolve_run_settings(args)
    print(f"device={device}  tasks={battery} ({len(active_tasks)})  batch={args.batch_size}  "
          f"steps timed={args.steps}")

    # Trial generation alone (numpy, CPU), the part every P shares.
    cfg = default_config(n_eachring=args.n_eachring, seed=0, easy_task=True)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        trial = generate_mixed_trials(active_tasks, cfg, args.batch_size, noise_on=True)
        trial_to_tensors(trial, device)
    t_gen = (time.perf_counter() - t0) / args.steps
    print(f"trial generation + transfer: {t_gen * 1e3:.1f} ms/step")

    for P in [int(v) for v in args.pop.split(",") if v.strip()]:
        specs = [{"seed": k, "rate_budget": 0.02, "conn_budget": 0.005} for k in range(P)]
        configs = [default_config(n_eachring=args.n_eachring, seed=k, easy_task=True) for k in range(P)]
        from cmc.runner import make_fresh_model
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            models = [make_fresh_model(args, c, device, seed=k) for k, c in enumerate(configs)]
        bmodel = BatchedDaleRNN(models).to(device)
        kw = dict(batch_size=args.batch_size, lr=args.lr, noise_level=noise_level,
                  log_every=10 ** 9, loss_per_trial=args.loss_per_trial, **reg_opts)
        train_batched(bmodel, configs[0], active_tasks, specs, n_steps=args.warmup, **kw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_batched(bmodel, configs[0], active_tasks, specs, n_steps=args.steps, **kw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.steps
        mem = f"  peak mem {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GB" if device.type == "cuda" else ""
        print(f"P={P:3d}: {dt:.3f} s/step  = {dt / P * 1e3:.1f} ms per network-step{mem}", flush=True)


if __name__ == "__main__":
    bench()
