import numpy as np
import matplotlib.pyplot as plt
import torch

from model_np import RateRNN

CMC_LABELS = ("E", "PV", "SST", "VIP")

CMC_BLOCK_STRENGTH = {
    f"{pre}→{post}": 0.4
    for pre in CMC_LABELS
    for post in CMC_LABELS
}

CMC_BLOCK_PROB = {
    f"{pre}→{post}": 0.8
    for pre in CMC_LABELS[1:]
    for post in CMC_LABELS[1:]
}

CMC_BLOCK_PROB_E = {
    f"{pre}→{post}": 0.2
    for pre in CMC_LABELS[0]
    for post in CMC_LABELS[0]
}

def _build_cmc_w_raw(n_neurons, bounds):
    """Build recurrent weights from 4x4 block strength / probability tables."""
    w_raw = torch.zeros(n_neurons, n_neurons)
    scale = 1.0 / np.sqrt(n_neurons)
    for pre_idx, pre_label in enumerate(CMC_LABELS):
        c0, c1 = bounds[pre_idx], bounds[pre_idx + 1]
        if c1 <= c0:
            continue
        for post_idx, post_label in enumerate(CMC_LABELS):
            r0, r1 = bounds[post_idx], bounds[post_idx + 1]
            if r1 <= r0:
                continue
            key = f"{pre_label}→{post_label}"
            strength = CMC_BLOCK_STRENGTH[key]
            prob = CMC_BLOCK_PROB.get(key, CMC_BLOCK_PROB_E.get(key, 1.0))
            block = torch.abs(torch.randn(r1 - r0, c1 - c0)) * strength * scale
            if prob < 1.0:
                block = block * (torch.rand_like(block) < prob).float()
            w_raw[r0:r1, c0:c1] = block
    w_raw.fill_diagonal_(0.0)
    return w_raw

class RateRNN(torch.nn.Module):
    """
    Differentiable rate RNN with Dale's law, soft firing-rate saturation,
    and tanh readout. Task-agnostic: use with external task modules for
    stimulus generation, loss, and training loops.
    """

    def __init__(
        self,
        n_neurons=10,
        frac_e=0.8,
        tau=10.0,
        dt=1.0,
        g=0.4,
        rate_max=30.0,
        act_gain=0.15,
        input_dim=3,
        output_dim=1,
        circuit_type = 'basic_dale'
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rate_max = rate_max
        self.act_gain = act_gain
        self.circuit_type = circuit_type
        self.n_e = int(n_neurons * frac_e)
        if self.circuit_type == 'basic_dale':
            neuron_types = torch.ones(n_neurons)
            self.n_i = n_neurons - self.n_e
            neuron_types[self.n_e:] = -1.0
        elif self.circuit_type == 'cmc':
            percentage_inhibitory = 0.33
            neuron_types = torch.ones(n_neurons)
            pop = torch.zeros(n_neurons, dtype=torch.long)
            self.n_pv = int(n_neurons * percentage_inhibitory * (1 - frac_e))
            self.n_sst = int(n_neurons * percentage_inhibitory * (1 - frac_e))
            self.n_vip = n_neurons - self.n_e - self.n_pv - self.n_sst
            self.n_i = self.n_pv + self.n_sst + self.n_vip
            neuron_types[self.n_e : self.n_e + self.n_pv] = -1.0
            neuron_types[self.n_e + self.n_pv : self.n_e + self.n_pv + self.n_sst] = -2.0
            neuron_types[self.n_e + self.n_pv + self.n_sst :] = -3.0
            pop[self.n_e : self.n_e + self.n_pv] = 1
            pop[self.n_e + self.n_pv : self.n_e + self.n_pv + self.n_sst] = 2
            pop[self.n_e + self.n_pv + self.n_sst :] = 3
            self.register_buffer("pop", pop)
        else:
            raise ValueError(f"Invalid circuit type: {circuit_type}")
        self.register_buffer("neuron_types", neuron_types)
        # Weights 
        if self.circuit_type == 'basic_dale':
            init_std = g / np.sqrt(n_neurons)
            w_raw = torch.abs(torch.randn(n_neurons, n_neurons) * init_std)
            w_raw.fill_diagonal_(0.0)
            self.W_raw = torch.nn.Parameter(w_raw)
        elif self.circuit_type == 'cmc':
            bounds = [
                0,
                self.n_e,
                self.n_e + self.n_pv,
                self.n_e + self.n_pv + self.n_sst,
                n_neurons,
            ]
            w_raw = _build_cmc_w_raw(n_neurons, bounds)
            self.W_raw = torch.nn.Parameter(w_raw)
        self.W_in = torch.nn.Parameter(torch.randn(n_neurons, input_dim) * 0.05 + 0.05)
        self.W_out = torch.nn.Parameter(torch.randn(output_dim, n_neurons) * 0.05)
        with torch.no_grad():
            self.W_out[:, self.n_e:] = 0.0

    @property
    def W(self):
        if self.circuit_type == 'basic_dale':
            # Sig of neuron type
            return self.W_raw * torch.sign(self.neuron_types.unsqueeze(0))
        elif self.circuit_type == 'cmc':
            signs = torch.tensor([1.0, -1.0, -1.0, -1.0], device=self.W_raw.device, dtype=self.W_raw.dtype)
            return self.W_raw * signs[self.pop].unsqueeze(0)
        else:
            raise ValueError(f"Invalid circuit type: {self.circuit_type}")


    def firing_rate(self, x):
        """
        Smooth supralinear rate with a soft ceiling.

        softplus avoids a hard ReLU dead zone; tanh caps rates differentiably.
        """
        raw = self.act_gain * torch.nn.functional.softplus(x).square()
        return self.rate_max * torch.tanh(raw / self.rate_max)

    @staticmethod
    def activation(x, alpha=1.0):
        """Legacy helper kept for compatibility with the NumPy API."""
        return alpha * torch.relu(x).square()

    def activity_stats(self, r_hist, x_hist=None):
        """Return fractions of saturated / near-silent units for diagnostics."""
        r = r_hist.detach()
        stats = {
            "rate_mean": r.mean().item(),
            "rate_std": r.std().item(),
            "frac_saturated": (r > 0.95 * self.rate_max).float().mean().item(),
            "frac_silent": (r < 0.05 * self.rate_max).float().mean().item(),
        }
        if x_hist is not None:
            stats["x_mean"] = x_hist.detach().mean().item()
            stats["x_std"] = x_hist.detach().std().item()
        return stats

    def _normalize_input(self, input_matrix):
        """Return (batch, input_dim, time_steps) tensor on model device."""
        if not torch.is_tensor(input_matrix):
            input_matrix = torch.as_tensor(input_matrix, dtype=torch.float32)
        if input_matrix.ndim == 1:
            input_matrix = input_matrix.unsqueeze(0).unsqueeze(0)
        elif input_matrix.ndim == 2:
            input_matrix = input_matrix.unsqueeze(0)
        elif input_matrix.ndim != 3:
            raise ValueError("input_matrix must have shape (input_dim, T), (batch, input_dim, T), or (T,)")
        param = next(self.parameters(), None)
        if param is not None:
            input_matrix = input_matrix.to(device=param.device, dtype=param.dtype)
        return input_matrix

    def simulate(self, input_matrix, x_init=None, noise_level=0.0):
        """
        Differentiable forward pass (BPTT-ready).

        Parameters
        ----------
        input_matrix : (input_dim, T) or (batch, input_dim, T)
        x_init : optional initial state, shape (n_neurons,) or (batch, n_neurons)
        noise_level : std of Gaussian state noise per step

        Returns
        -------
        r_hist : (batch, T, n_neurons)
        x_hist : (batch, T, n_neurons)
        output_matrix : (batch, T, output_dim)
        """
        input_matrix = self._normalize_input(input_matrix)
        batch_size, _, time_steps = input_matrix.shape
        device = input_matrix.device
        dtype = input_matrix.dtype

        if x_init is None:
            x = torch.rand(batch_size, self.n_neurons, device=device, dtype=dtype) * 0.01
        elif x_init.ndim == 1:
            x = x_init.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            x = x_init.clone()

        i_ext = torch.einsum("ni,biT->bnT", self.W_in, input_matrix)
        w_rec = self.W

        x_hist = []
        r_hist = []
        output_hist = []

        for t in range(time_steps):
            r = self.firing_rate(x)
            i_rec = torch.matmul(r, w_rec.t()) / np.sqrt(self.n_neurons)
            noise = 0.0
            if noise_level > 0.0:
                noise = torch.randn_like(x) * noise_level * np.sqrt(self.dt)
            dx = (-x + i_rec + i_ext[:, :, t]) * self.alpha + noise
            x = x + dx

            x_hist.append(x)
            r_hist.append(r)
            output_hist.append(torch.tanh(torch.matmul(r, self.W_out.t())))

        x_hist = torch.stack(x_hist, dim=1)
        r_hist = torch.stack(r_hist, dim=1)
        output_matrix = torch.stack(output_hist, dim=1)

        return r_hist, x_hist, output_matrix

    def train_step(self, optimizer, input_matrix, loss_fn, noise_level=0.0, grad_clip=1.0):
        """
        Single BPTT update.

        Parameters
        ----------
        loss_fn : callable(r_hist, x_hist, output_matrix) -> scalar loss tensor
        """
        optimizer.zero_grad()
        r_hist, x_hist, output_matrix = self.simulate(input_matrix, noise_level=noise_level)
        loss = loss_fn(r_hist, x_hist, output_matrix)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        return loss.detach()
