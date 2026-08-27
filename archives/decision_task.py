import torch


class GoCueDecisionTask:
    """
    Go-cue-gated decision task with time-varying evidence.

    Input channel layout (default):
        0: go cue (0 during delay, 1 after go onset)
        1, 2: time-varying decision evidence (Gaussian noise + coherence bias)

    Target is the sign of the cumulative evidence difference over the
    integration window (default: delay period before go onset).
    """

    def __init__(
        self,
        go_onset=200,
        go_cue_channel=0,
        decision_channels=(1, 2),
        evidence_onset=0,
        evidence_end=None,
    ):
        self.go_onset = go_onset
        self.go_cue_channel = go_cue_channel
        self.decision_channels = decision_channels
        self.evidence_onset = evidence_onset
        self.evidence_end = evidence_end

    @property
    def input_dim(self):
        return max(self.decision_channels) + 1

    @staticmethod
    def target_from_evidence(evidence_a, evidence_b, integration_mask=None):
        """Target in {-1, +1}: +1 if channel A integrates higher than B, else -1."""
        if integration_mask is not None:
            evidence_a = evidence_a[..., integration_mask]
            evidence_b = evidence_b[..., integration_mask]
        cumulative = evidence_a.sum(dim=-1) - evidence_b.sum(dim=-1)
        target = torch.sign(cumulative)
        target = torch.where(target == 0, torch.ones_like(target), target)
        return target

    def sample_trial(self, time_steps, coherence=0.1, device=None, dtype=torch.float32):
        """
        Build one trial.

        Returns
        -------
        input_matrix : (input_dim, time_steps)
        target : scalar tensor in {-1, +1}
        """
        evidence_end = self.evidence_end if self.evidence_end is not None else self.go_onset

        input_matrix = torch.zeros(self.input_dim, time_steps, device=device, dtype=dtype)
        input_matrix[self.go_cue_channel, self.go_onset:] = 1.0

        ch_a, ch_b = self.decision_channels
        noise_a = torch.randn(time_steps, device=device, dtype=dtype)
        noise_b = torch.randn(time_steps, device=device, dtype=dtype)

        favor_a = torch.rand((), device=device) > 0.5
        if favor_a:
            input_matrix[ch_a, :] = noise_a + coherence
            input_matrix[ch_b, :] = noise_b - coherence
        else:
            input_matrix[ch_a, :] = noise_a - coherence
            input_matrix[ch_b, :] = noise_b + coherence

        integration_mask = torch.zeros(time_steps, dtype=torch.bool, device=device)
        integration_mask[self.evidence_onset:evidence_end] = True
        target = self.target_from_evidence(
            input_matrix[ch_a],
            input_matrix[ch_b],
            integration_mask=integration_mask,
        )
        return input_matrix, target

    def sample_batch(self, batch_size, time_steps, coherence=0.1, device=None, dtype=torch.float32):
        inputs = []
        targets = []
        for _ in range(batch_size):
            inp, tgt = self.sample_trial(time_steps, coherence=coherence, device=device, dtype=dtype)
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)

    def go_mask(self, inputs):
        """Boolean mask (batch, time_steps) where go cue is active."""
        if inputs.ndim == 2:
            return inputs[self.go_cue_channel, :] > 0.5
        return inputs[:, self.go_cue_channel, :] > 0.5

    def loss(self, outputs, targets, go_mask):
        """Go-cue-gated MSE on tanh readout."""
        batch_size, time_steps, _ = outputs.shape
        device = outputs.device
        dtype = outputs.dtype

        targets = torch.as_tensor(targets, device=device, dtype=dtype)
        if targets.ndim == 0:
            targets = targets.unsqueeze(0).expand(batch_size)
        elif targets.ndim == 1 and targets.shape[0] == 1 and batch_size > 1:
            targets = targets.expand(batch_size)

        target_expanded = targets.unsqueeze(-1).expand(-1, time_steps)
        sq_err = (outputs.squeeze(-1) - target_expanded).square()
        masked_err = sq_err * go_mask
        denom = go_mask.sum().clamp_min(1.0)
        return masked_err.sum() / denom

    @staticmethod
    def trial_predictions(output_matrix, go_mask):
        """Trial-level decision from mean readout during the go epoch."""
        go_outputs = output_matrix.squeeze(-1).masked_fill(~go_mask, float("nan"))
        trial_mean = torch.nanmean(go_outputs, dim=1)
        preds = torch.sign(trial_mean)
        preds = torch.where(preds == 0, torch.ones_like(preds), preds)
        return preds, trial_mean

    def accuracy(self, outputs, targets, go_mask):
        preds, _ = self.trial_predictions(outputs, go_mask)
        targets = torch.as_tensor(targets, device=outputs.device)
        return (preds == targets).float().mean().item()

    def train_step(self, model, optimizer, inputs, targets, noise_level=0.0, grad_clip=1.0):
        """One BPTT update for this task."""
        go_mask = self.go_mask(inputs)

        def loss_fn(r_hist, x_hist, output_matrix):
            return self.loss(output_matrix, targets, go_mask)

        return model.train_step(
            optimizer,
            inputs,
            loss_fn,
            noise_level=noise_level,
            grad_clip=grad_clip,
        )

    def train(
        self,
        model,
        n_epochs=100,
        batch_size=16,
        time_steps=1000,
        coherence=0.1,
        lr=1e-3,
        noise_level=0.02,
        log_every=10,
        grad_clip=1.0,
        coherence_curriculum=True,
    ):
        """
        Train a RateRNN_torch on this task via BPTT.

        Returns
        -------
        history : dict with ``loss``, ``accuracy``, and ``activity`` lists
        """
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"loss": [], "accuracy": [], "activity": []}

        for epoch in range(1, n_epochs + 1):
            train_coh = coherence
            if coherence_curriculum and n_epochs > 1:
                warmup = max(coherence, 0.25)
                progress = (epoch - 1) / (n_epochs - 1)
                train_coh = warmup + (coherence - warmup) * progress

            inputs, targets = self.sample_batch(
                batch_size,
                time_steps,
                coherence=train_coh,
                device=device,
            )
            loss = self.train_step(
                model,
                optimizer,
                inputs,
                targets,
                noise_level=noise_level,
                grad_clip=grad_clip,
            )

            with torch.no_grad():
                r_hist, x_hist, outputs = model.simulate(inputs, noise_level=0.0)
                go_mask = self.go_mask(inputs)
                accuracy = self.accuracy(outputs, targets, go_mask)
                activity = model.activity_stats(r_hist, x_hist)

            history["loss"].append(loss.item())
            history["accuracy"].append(accuracy)
            history["activity"].append(activity)
            if epoch == 1 or epoch % log_every == 0 or epoch == n_epochs:
                print(
                    f"epoch {epoch:4d} | loss {loss.item():.4f} | accuracy {accuracy:.3f} "
                    f"| coh {train_coh:.3f} | sat {activity['frac_saturated']:.2f} "
                    f"| silent {activity['frac_silent']:.2f}"
                )

        return history

    def evaluate_coherence_sweep(
        self,
        model,
        coherences,
        n_trials=50,
        time_steps=1000,
        noise_level=0.0,
    ):
        """Return accuracy at each coherence level."""
        device = next(model.parameters()).device
        accuracies = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for coh in coherences:
                inputs, targets = self.sample_batch(
                    n_trials,
                    time_steps,
                    coherence=coh,
                    device=device,
                )
                _, _, outputs = model.simulate(inputs, noise_level=noise_level)
                go_mask = self.go_mask(inputs)
                accuracies.append(self.accuracy(outputs, targets, go_mask))
        if was_training:
            model.train()
        return accuracies
