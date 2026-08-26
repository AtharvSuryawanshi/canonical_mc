"""Scoped Yang 2019 task battery: fdgo, delaygo (single ring, random mode)."""

import numpy as np

rules_dict = {"all": ["fdgo", "delaygo"]}

rule_index_map = {
    ruleset: {rule: ind for ind, rule in enumerate(rules)}
    for ruleset, rules in rules_dict.items()
}


def default_config(n_eachring=32, seed=0, easy_task=True, **kwargs):
    """Hyperparameters shared by the task generators and the RNN."""
    # Accept legacy kw alias n_neurons_per_ring
    if "n_neurons_per_ring" in kwargs:
        n_eachring = kwargs["n_neurons_per_ring"]
    # Match Yang multitask defaults (train.py get_default_hp)
    dt = 20
    tau = 100.0
    num_ring = 1
    n_rule = len(rules_dict["all"])
    config = {
        "dt": dt,
        "tau": tau,
        "alpha": dt / tau,
        "sigma_x": 0.01,
        "n_eachring": n_eachring,
        "n_neurons_per_ring": n_eachring,
        "num_ring": num_ring,
        "n_rule": n_rule,
        "n_rules": n_rule,
        "ruleset": "all",
        "rule_start": 1 + n_eachring * num_ring,
        "n_input": 1 + n_eachring * num_ring + n_rule,
        "n_output": 1 + n_eachring,
        "loss_type": "lsq",
        "easy_task": easy_task,
        "rng": np.random.RandomState(seed),
    }
    return config


def get_rule_index(rule, config):
    return rule_index_map[config["ruleset"]][rule] + config["rule_start"]


def get_dist(original_dist):
    """Circular distance on [0, 2π)."""
    return np.minimum(np.abs(original_dist), 2 * np.pi - np.abs(original_dist))


class Trial:
    """A batch of trials: x, y, y_loc, c_mask."""

    def __init__(self, config, tdim, batch_size):
        self.float_type = "float32"
        self.config = config
        self.dt = config["dt"]
        self.n_neurons_per_ring = config["n_neurons_per_ring"]
        self.n_eachring = config["n_eachring"]
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.pref = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_neurons_per_ring)

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        if config["loss_type"] == "lsq":
            self.y[:, :, :] = 0.05
        self.y_loc = -np.ones((tdim, batch_size), dtype=self.float_type)
        self.c_mask = None
        self.epochs = {}
        self.rule = None
        self._sigma_x = config["sigma_x"] * np.sqrt(2.0 / config["alpha"])

    def expand(self, var):
        if not hasattr(var, "__iter__"):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1):
        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)

        for i in range(self.batch_size):
            if loc_type == "fix_in":
                self.x[ons[i] : offs[i], i, 0] = 1
            elif loc_type == "stim":
                self.x[ons[i] : offs[i], i, 1 : 1 + self.n_neurons_per_ring] += (
                    self.add_x_loc(locs[i]) * strengths[i]
                )
            elif loc_type == "fix_out":
                if self.config["loss_type"] == "lsq":
                    self.y[ons[i] : offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i] : offs[i], i, 0] = 1.0
            elif loc_type == "out":
                if self.config["loss_type"] == "lsq":
                    self.y[ons[i] : offs[i], i, 1:] += self.add_y_loc(locs[i]) * strengths[i]
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp = y_tmp / np.sum(y_tmp)
                    self.y[ons[i] : offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i] : offs[i], i] = locs[i]
            else:
                raise ValueError(f"Unknown loc_type: {loc_type}")

    def add_x_noise(self):
        self.x += self.config["rng"].randn(*self.x.shape).astype(self.float_type) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons):
        pre_on = int(100 / self.dt)
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        if self.config["loss_type"] == "lsq":
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                c_mask[post_ons[i] :, i, :] = 5.0
                c_mask[pre_on : pre_offs[i], i, :] = 1.0
            c_mask[:, :, 0] *= 2.0
            self.c_mask = c_mask
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                c_mask[post_ons[i] :, i] = 5.0
                c_mask[pre_on : pre_offs[i], i] = 1.0
            self.c_mask = c_mask.reshape((self.tdim * self.batch_size,))
            self.c_mask /= self.c_mask.mean()

    def add_rule(self, rule, on=None, off=None, strength=1.0):
        if isinstance(rule, (int, np.integer)):
            self.x[on:off, :, self.config["rule_start"] + int(rule)] = strength
        else:
            self.x[on:off, :, get_rule_index(rule, self.config)] = strength

    def add_x_loc(self, x_loc):
        dist = get_dist(x_loc - self.pref)
        dist /= np.pi / 8
        return 0.8 * np.exp(-(dist ** 2) / 2)

    def add_y_loc(self, y_loc):
        dist = get_dist(y_loc - self.pref)
        if self.config["loss_type"] == "lsq":
            dist /= np.pi / 8
            return 0.8 * np.exp(-(dist ** 2) / 2)
        y = np.zeros_like(dist)
        y[np.argmin(dist)] = 1.0
        return y


def fdgo(config, batch_size):
    """Fixation-delayed go: stimulus on until go; saccade to its location."""
    dt = config["dt"]
    rng = config["rng"]
    stim_locs = rng.rand(batch_size) * 2 * np.pi
    stim_ons = int(rng.uniform(300, 700) / dt)
    fix_offs = stim_ons + int(rng.uniform(500, 1500) / dt)
    tdim = int(500 / dt) + fix_offs
    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=fix_offs)
    trial.add("stim", stim_locs, ons=stim_ons, offs=fix_offs)
    trial.add("fix_out", offs=fix_offs)
    trial.add("out", stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim_ons),
        "stim1": (stim_ons, fix_offs),
        "go1": (fix_offs, None),
    }
    return trial


def delaygo(config, batch_size):
    """Working-memory go: stimulus removed before go; saccade to remembered location."""
    dt = config["dt"]
    rng = config["rng"]
    stim_locs = rng.rand(batch_size) * 2 * np.pi
    stim_ons = int(rng.choice([300, 500, 700]) / dt)
    stim_offs = stim_ons + int(rng.choice([200, 400, 600]) / dt)
    fix_offs = stim_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    tdim = fix_offs + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=fix_offs)
    trial.add("stim", stim_locs, ons=stim_ons, offs=stim_offs)
    trial.add("fix_out", offs=fix_offs)
    trial.add("out", stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim_ons),
        "stim1": (stim_ons, stim_offs),
        "delay1": (stim_offs, fix_offs),
        "go1": (fix_offs, None),
    }
    return trial


rule_mapping = {
    "fdgo": fdgo,
    "delaygo": delaygo,
}


def generate_trials(rule, config, batch_size, noise_on=True):
    """Generate one homogeneous batch for ``rule`` and attach rule input + noise."""
    if rule not in rule_mapping:
        raise ValueError(f"Unknown rule: {rule}. Choose from {list(rule_mapping)}")
    trial = rule_mapping[rule](config, batch_size)
    trial.add_rule(rule)
    trial.rule = rule
    if noise_on:
        trial.add_x_noise()
    return trial


def pad_trial(trial, tdim):
    """Pad trial arrays along time; padded steps get zero loss weight."""
    if trial.tdim >= tdim:
        return trial
    pad = tdim - trial.tdim
    trial.x = np.pad(trial.x, ((0, pad), (0, 0), (0, 0)))
    trial.y = np.pad(trial.y, ((0, pad), (0, 0), (0, 0)), constant_values=0.05)
    trial.y_loc = np.pad(trial.y_loc, ((0, pad), (0, 0)), constant_values=-1)
    if trial.c_mask is not None:
        trial.c_mask = np.pad(trial.c_mask, ((0, pad), (0, 0), (0, 0)))
    trial.tdim = tdim
    return trial


def concat_trials(trials):
    """Concatenate same-config trials along batch; pad to common ``tdim``."""
    if len(trials) == 1:
        return trials[0]
    tdim = max(t.tdim for t in trials)
    trials = [pad_trial(t, tdim) for t in trials]
    merged = trials[0]
    merged.x = np.concatenate([t.x for t in trials], axis=1)
    merged.y = np.concatenate([t.y for t in trials], axis=1)
    merged.y_loc = np.concatenate([t.y_loc for t in trials], axis=1)
    merged.c_mask = np.concatenate([t.c_mask for t in trials], axis=1)
    merged.batch_size = int(sum(t.batch_size for t in trials))
    merged.rule = "mixed"
    return merged


def generate_mixed_trials(active_tasks, config, batch_size, noise_on=True):
    """One batch with roughly equal counts from each task (mixed multitask batch)."""
    active_tasks = tuple(active_tasks)
    n_tasks = len(active_tasks)
    if n_tasks == 0:
        raise ValueError("active_tasks must be non-empty")
    if n_tasks == 1:
        return generate_trials(active_tasks[0], config, batch_size, noise_on=noise_on)

    base = batch_size // n_tasks
    rem = batch_size % n_tasks
    sizes = [base + (1 if i < rem else 0) for i in range(n_tasks)]
    trials = [
        generate_trials(task, config, sz, noise_on=noise_on)
        for task, sz in zip(active_tasks, sizes)
        if sz > 0
    ]
    return concat_trials(trials)
