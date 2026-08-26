"""PyTorch port of Yang et al. LeakyRNN (gyyang/multitask network.py)."""

import math

import numpy as np
import torch
import torch.nn as nn


def gen_ortho_matrix(dim, rng=None):
    """Random orthogonal matrix (Householder construction from Yang tools.py)."""
    if rng is None:
        rng = np.random
    h = np.eye(dim)
    for n in range(1, dim):
        x = rng.normal(size=(dim - n + 1,))
        d = np.sign(x[0])
        x[0] += d * np.sqrt((x * x).sum())
        hx = -d * (np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1 :, n - 1 :] = hx
        h = np.dot(h, mat)
    return h


def popvec(y):
    """Population vector readout. y: (Batch, Units) or (..., Units)."""
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / y.shape[-1])
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y * np.cos(pref), axis=-1) / temp_sum
    temp_sin = np.sum(y * np.sin(pref), axis=-1) / temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2 * np.pi)


def get_perf(y_hat, y_loc):
    """Trial performance from last time step. y_hat: (Time, Batch, Unit), y_loc: (Time, Batch)."""
    if len(y_hat.shape) != 3:
        raise ValueError("y_hat must have shape (Time, Batch, Unit)")
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])
    fixating = y_hat_fix > 0.5
    original_dist = y_loc - y_hat_loc
    dist = np.minimum(np.abs(original_dist), 2 * np.pi - np.abs(original_dist))
    corr_loc = dist < 0.2 * np.pi
    should_fix = y_loc < 0
    perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
    return perf


def _activation_fn(name):
    if name == "softplus":
        return nn.Softplus()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "power":
        return lambda x: torch.square(torch.relu(x))
    if name == "retanh":
        return lambda x: torch.tanh(torch.relu(x))
    raise ValueError(f"Unknown activation: {name}")


def _init_fused_kernel(n_input, n_rnn, activation, w_rec_init, rng):
    """Match Yang LeakyRNNCell weight initialization."""
    if activation == "softplus":
        w_in_start, w_rec_start = 1.0, 0.5
    elif activation == "tanh":
        w_in_start, w_rec_start = 1.0, 1.0
    elif activation in ("relu", "power", "retanh"):
        w_in_start, w_rec_start = 1.0, 0.5
    else:
        w_in_start, w_rec_start = 1.0, 0.5

    w_in0 = rng.randn(n_input, n_rnn) / np.sqrt(n_input) * w_in_start
    if w_rec_init == "diag":
        w_rec0 = w_rec_start * np.eye(n_rnn)
    elif w_rec_init == "randortho":
        w_rec0 = w_rec_start * gen_ortho_matrix(n_rnn, rng=rng)
    elif w_rec_init == "randgauss":
        w_rec0 = w_rec_start * rng.randn(n_rnn, n_rnn) / np.sqrt(n_rnn)
    else:
        raise ValueError(f"Unknown w_rec_init: {w_rec_init}")
    return np.concatenate((w_in0, w_rec0), axis=0).astype(np.float32)


class LeakyRNN(nn.Module):
    """
    Yang-style leaky RNN: h' = (1-α)h + α·act(W·[x,h]+b+noise), sigmoid readout.

    Input layout matches task.py: (batch, n_input, T).
    """

    def __init__(
        self,
        n_input,
        n_rnn,
        n_output,
        alpha=0.1,
        activation="relu",
        w_rec_init="diag",
        sigma_rec=0.05,
        seed=0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_rnn = n_rnn
        self.n_output = n_output
        self.alpha = alpha
        self.activation_name = activation
        self.sigma = math.sqrt(2.0 / alpha) * sigma_rec

        rng = np.random.RandomState(seed)
        kernel0 = _init_fused_kernel(n_input, n_rnn, activation, w_rec_init, rng)
        self.kernel = nn.Parameter(torch.from_numpy(kernel0))
        self.bias = nn.Parameter(torch.zeros(n_rnn))
        self.w_out = nn.Parameter(torch.randn(n_rnn, n_output) * 0.01)
        self.b_out = nn.Parameter(torch.zeros(n_output))

        self._act = _activation_fn(activation)

    @property
    def w_in(self):
        return self.kernel[: self.n_input, :]

    @property
    def w_rec(self):
        return self.kernel[self.n_input :, :]

    def _normalize_input(self, input_matrix):
        if not torch.is_tensor(input_matrix):
            input_matrix = torch.as_tensor(input_matrix, dtype=torch.float32)
        if input_matrix.ndim == 1:
            input_matrix = input_matrix.unsqueeze(0).unsqueeze(0)
        elif input_matrix.ndim == 2:
            input_matrix = input_matrix.unsqueeze(0)
        elif input_matrix.ndim != 3:
            raise ValueError("input_matrix must be (batch, n_input, T) or compatible")
        param = next(self.parameters(), None)
        if param is not None:
            input_matrix = input_matrix.to(device=param.device, dtype=param.dtype)
        return input_matrix

    def _cell_act(self, gate_inputs):
        if callable(self._act) and not isinstance(self._act, nn.Module):
            return self._act(gate_inputs)
        return self._act(gate_inputs)

    def simulate(self, input_matrix, x_init=None, noise_level=0.0):
        """
        Returns
        -------
        h_hist, h_hist, y_hat : (batch, T, n_rnn), (batch, T, n_rnn), (batch, T, n_output)
        """
        input_matrix = self._normalize_input(input_matrix)
        batch_size, _, time_steps = input_matrix.shape
        device = input_matrix.device
        dtype = input_matrix.dtype

        if x_init is None:
            h = torch.zeros(batch_size, self.n_rnn, device=device, dtype=dtype)
        elif x_init.ndim == 1:
            h = x_init.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            h = x_init.clone()

        h_hist = []
        y_hist = []
        alpha = self.alpha

        for t in range(time_steps):
            x_t = input_matrix[:, :, t]
            gate = torch.cat([x_t, h], dim=1) @ self.kernel + self.bias
            if noise_level > 0.0:
                gate = gate + torch.randn_like(h) * (noise_level * self.sigma)
            h_new = self._cell_act(gate)
            h = (1.0 - alpha) * h + alpha * h_new
            h_hist.append(h)
            y_hist.append(torch.sigmoid(h @ self.w_out + self.b_out))

        h_hist = torch.stack(h_hist, dim=1)
        y_hat = torch.stack(y_hist, dim=1)
        return h_hist, h_hist, y_hat

    def activity_stats(self, h_hist, x_hist=None):
        h = h_hist.detach()
        stats = {
            "rate_mean": h.mean().item(),
            "rate_std": h.std().item(),
            "frac_silent": (h.abs() < 0.05).float().mean().item(),
            "frac_saturated": (h > 10.0).float().mean().item(),
        }
        if x_hist is not None:
            stats["x_mean"] = x_hist.detach().mean().item()
            stats["x_std"] = x_hist.detach().std().item()
        return stats

    def train_step(self, optimizer, input_matrix, loss_fn, noise_level=0.0, grad_clip=1.0):
        optimizer.zero_grad()
        h_hist, x_hist, output_matrix = self.simulate(input_matrix, noise_level=noise_level)
        loss = loss_fn(h_hist, x_hist, output_matrix)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        return loss.detach()
