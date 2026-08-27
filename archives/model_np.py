import numpy as np
import matplotlib.pyplot as plt

class RateRNN:
    def __init__(self, n_neurons=10, frac_e=0.8, tau=10.0, dt=1.0, g=1.5, rate_max = 100, input_dim=1, output_dim=1):
        """
        n_neurons: Total number of neurons
        frac_e: Fraction of excitatory neurons (e.g. 0.8 = 80% E, 20% I)
        tau: Decay time constant (ms)
        dt: Simulation time step (ms)
        g: Synaptic gain factor (scales initial weight variance)
        """
        self.n_neurons = n_neurons
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau  # Decay rate multiplier per step
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Cell type assignment
        self.n_e = int(n_neurons * frac_e)
        self.n_i = n_neurons - self.n_e
        self.neuron_types = np.ones(n_neurons)
        self.neuron_types[self.n_e:] = -1.0  # Inhibitory mask
        self.rate_max = rate_max

        # 2. Dale's Law compliance: raw weights are non-negative
        self.W_raw = np.abs(np.random.normal(0, g / np.sqrt(n_neurons), (n_neurons, n_neurons))) # chaotic regime of g/sqrt(N)
        self.W_in = np.ones((n_neurons, input_dim)) # weighting of external input
        self.W_out = np.ones((output_dim, n_neurons)) # weighting of output
        # zero output for inhibitory neurons
        self.W_out[self.n_e:, :] = 0.0
        np.fill_diagonal(self.W_raw, 0.0)  # Remove self-loops

    @property
    def W(self):
        """
        Enforce Dale's Law:
        Columns 0..N_e-1 are Excitatory (>= 0)
        Columns N_e..N-1 are Inhibitory (<= 0)
        """
        return self.W_raw * self.neuron_types[None, :]
    

    @staticmethod
    def activation(x, alpha=1.0):
        """Supralinear activation: (ReLU(x))^2"""
        return alpha * np.maximum(0.0, x) ** 2

    def simulate(self, input_matrix, x_init=None, time_steps=500, noise_level=0.0):
        """
        input_matrix: Shape (n_channels, time_steps) # Same for all neurons, weighting done in W_in
        Returns states x and rates r over time.
        """
        if input_matrix.ndim == 1:
            I_ext = np.tile(self.W_in * input_matrix, (time_steps, 1))
        else:
            I_ext = np.dot(self.W_in, input_matrix)
            print(I_ext.shape)

        if x_init is None:
            x = np.random.uniform(0, 0.1, self.n_neurons)
        else:
            x = x_init.copy()

        x_hist = np.zeros((time_steps, self.n_neurons))
        r_hist = np.zeros((time_steps, self.n_neurons))
        output_matrix = np.zeros((time_steps, self.output_dim))

        W = np.abs(self.W)  # Precompute effective weight matrix
        
        for t in range(time_steps):
            r = self.activation(x, alpha=1)
            r = np.clip(r, 0, self.rate_max)
            I_rec = W @ r
            # Update
            dx = (-x + I_rec + I_ext[:,t]) * self.alpha + np.random.normal(0, noise_level, self.n_neurons) * np.sqrt(self.dt)
            x += dx
            x_hist[t] = x
            r_hist[t] = r
            output_matrix[t] = self.W_out @ r
        return r_hist, x_hist, output_matrix