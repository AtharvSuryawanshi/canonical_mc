"""Yang 2019 task battery (single ring, random mode)."""

import numpy as np

rules_dict = {
    "all": [
        "fdgo",
        "reactgo",
        "delaygo",
        "fdanti",
        "reactanti",
        "delayanti",
        "dm1",
        "dm2",
        "contextdm1",
        "contextdm2",
        "multidm",
        "delaydm1",
        "delaydm2",
        "contextdelaydm1",
        "contextdelaydm2",
        "multidelaydm",
        "dmsgo",
        "dmsnogo",
        "dmcgo",
        "dmcnogo",
    ],
}

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
    num_ring = 2
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

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=1):
        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == "fix_in":
                self.x[ons[i] : offs[i], i, 0] = 1
            elif loc_type == "stim":
                mod = mods[i]
                start = 1 + (mod - 1) * self.n_neurons_per_ring
                end = 1 + mod * self.n_neurons_per_ring
                self.x[ons[i] : offs[i], i, start:end] += (
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


def _response_locs(stim_locs, anti_response):
    """Map stimulus locations to pro- or anti-saccade targets on the ring."""
    stim_locs = np.asarray(stim_locs)
    if not anti_response:
        return stim_locs
    return (stim_locs + np.pi) % (2 * np.pi)


def _fdgo(config, batch_size, anti_response=False):
    """Fixation-delayed go/anti: stimulus on until go; saccade pro or anti."""
    dt = config["dt"]
    rng = config["rng"]
    stim_locs = rng.rand(batch_size) * 2 * np.pi
    stim_ons = int(rng.uniform(300, 700) / dt)
    fix_offs = stim_ons + int(rng.uniform(500, 1500) / dt)
    tdim = int(500 / dt) + fix_offs
    check_ons = fix_offs + int(100 / dt)
    response_locs = _response_locs(stim_locs, anti_response)

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=fix_offs)
    trial.add("stim", stim_locs, ons=stim_ons, offs=fix_offs)
    trial.add("fix_out", offs=fix_offs)
    trial.add("out", response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim_ons),
        "stim1": (stim_ons, fix_offs),
        "go1": (fix_offs, None),
    }
    return trial


def fdgo(config, batch_size):
    """Fixation-delayed go: stimulus on until go; saccade to its location."""
    return _fdgo(config, batch_size, anti_response=False)


def fdanti(config, batch_size):
    """Fixation-delayed anti: stimulus on until go; saccade to opposite location."""
    return _fdgo(config, batch_size, anti_response=True)


def _reactgo(config, batch_size, anti_response=False):
    """Reaction-time go/anti: hold fixation until stimulus, then saccade immediately."""
    dt = config["dt"]
    rng = config["rng"]
    stim_ons = int(rng.uniform(500, 2500) / dt)
    tdim = int(500 / dt) + stim_ons
    check_ons = stim_ons + int(100 / dt)
    stim_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    response_locs = _response_locs(stim_locs, anti_response)

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=stim_ons)
    trial.add("stim", stim_locs, ons=stim_ons)
    trial.add("fix_out", offs=stim_ons)
    trial.add("out", response_locs, ons=stim_ons)
    trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim_ons),
        "go1": (stim_ons, None),
    }
    return trial


def reactgo(config, batch_size):
    """Reaction-time go: hold fixation until stimulus onset, then saccade immediately."""
    return _reactgo(config, batch_size, anti_response=False)


def reactanti(config, batch_size):
    """Reaction-time anti: hold fixation until stimulus, then saccade to opposite location."""
    return _reactgo(config, batch_size, anti_response=True)


def _delaygo(config, batch_size, anti_response=False):
    """Working-memory go/anti: brief stimulus, delay, then pro or anti saccade."""
    dt = config["dt"]
    rng = config["rng"]
    stim_locs = rng.rand(batch_size) * 2 * np.pi
    stim_ons = int(rng.choice([300, 500, 700]) / dt)
    stim_offs = stim_ons + int(rng.choice([200, 400, 600]) / dt)
    fix_offs = stim_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    tdim = fix_offs + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)
    response_locs = _response_locs(stim_locs, anti_response)

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=fix_offs)
    trial.add("stim", stim_locs, ons=stim_ons, offs=stim_offs)
    trial.add("fix_out", offs=fix_offs)
    trial.add("out", response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim_ons),
        "stim1": (stim_ons, stim_offs),
        "delay1": (stim_offs, fix_offs),
        "go1": (fix_offs, None),
    }
    return trial


def delaygo(config, batch_size):
    """Working-memory go: stimulus removed before go; saccade to remembered location."""
    return _delaygo(config, batch_size, anti_response=False)


def delayanti(config, batch_size):
    """Working-memory anti: stimulus removed before go; saccade to opposite location."""
    return _delaygo(config, batch_size, anti_response=True)


def _dm(config, batch_size, stim_mod=1):
    """Decision making: two simultaneous bumps on one ring; saccade to the stronger."""
    dt = config["dt"]
    rng = config["rng"]

    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice(
        [-1, 1], (batch_size,)
    )
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    stims_mean = rng.uniform(0.8, 1.2, (batch_size,))
    stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08], dtype=np.float32)
    if config.get("easy_task", True):
        stim_coh_range = stim_coh_range * 10

    stims_coh = rng.choice(stim_coh_range, (batch_size,))
    stims_sign = rng.choice([1, -1], (batch_size,))
    stim1_strengths = stims_mean + stims_coh * stims_sign
    stim2_strengths = stims_mean - stims_coh * stims_sign

    stim_on = int(rng.uniform(100, 400) / dt)
    stim_ons = stim_on
    stim_dur = int(rng.choice([400, 800, 1600]) / dt)
    fix_offs = stim_ons + stim_dur
    tdim = stim_on + stim_dur + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=fix_offs)
    trial.add(
        "stim",
        stim1_locs,
        ons=stim_ons,
        offs=fix_offs,
        strengths=stim1_strengths,
        mods=stim_mod,
    )
    trial.add(
        "stim",
        stim2_locs,
        ons=stim_ons,
        offs=fix_offs,
        strengths=stim2_strengths,
        mods=stim_mod,
    )
    trial.add("fix_out", offs=fix_offs)
    stim_locs = np.where(stim1_strengths > stim2_strengths, stim1_locs, stim2_locs)
    trial.add("out", stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim_ons),
        "stim1": (stim_ons, fix_offs),
        "go1": (fix_offs, None),
    }
    return trial


def dm1(config, batch_size):
    """Decision making 1: compare two bumps on ring 1; saccade to the stronger."""
    return _dm(config, batch_size, stim_mod=1)


def dm2(config, batch_size):
    """Decision making 2: same as dm1 but stimuli on ring 2."""
    return _dm(config, batch_size, stim_mod=2)


def _contextdm_genstim(batch_size, rng, stim_coh_range=None):
    stim_mean = rng.uniform(0.8, 1.2, (batch_size,))
    if stim_coh_range is None:
        stim_coh_range = np.array([0.16, 0.32, 0.64])
    stim_coh = rng.choice(stim_coh_range, (batch_size,))
    stim_sign = rng.choice([1, -1], (batch_size,))
    stim1_strengths = stim_mean + stim_coh * stim_sign
    stim2_strengths = stim_mean - stim_coh * stim_sign
    return stim1_strengths, stim2_strengths


def _contextdm(config, batch_size, attend_mod):
    """Context decision making: two bumps per ring; attend one modality or both."""
    dt = config["dt"]
    rng = config["rng"]

    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice(
        [-1, 1], (batch_size,)
    )
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08], dtype=np.float32)
    if config.get("easy_task", True):
        stim_coh_range = stim_coh_range * 10

    if attend_mod in (1, 2):
        stim1_mod1_strengths, stim2_mod1_strengths = _contextdm_genstim(
            batch_size, rng, stim_coh_range
        )
        stim1_mod2_strengths, stim2_mod2_strengths = _contextdm_genstim(
            batch_size, rng, stim_coh_range
        )
        if attend_mod == 1:
            stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
        else:
            stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    else:
        stim1_strengths, stim2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)

        stim1_mod12_diff = (
            stim1_strengths
            * rng.uniform(0.2, 0.8, (batch_size,))
            * rng.choice([1, -1], (batch_size,))
        )
        stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
        stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2

        stim2_mod12_diff = (
            stim2_strengths
            * rng.uniform(0.2, 0.8, (batch_size,))
            * rng.choice([1, -1], (batch_size,))
        )
        stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
        stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

    stim_on = int(rng.uniform(100, 400) / dt)
    stim_ons = stim_on
    stim_dur = int(rng.choice([400, 800, 1600]) / dt)
    stim_offs = stim_ons + stim_dur
    fix_offs = stim_offs
    tdim = stim_on + stim_dur + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == "both":
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=fix_offs)
    trial.add(
        "stim",
        stim1_locs,
        ons=stim_ons,
        offs=stim_offs,
        strengths=stim1_mod1_strengths,
        mods=1,
    )
    trial.add(
        "stim",
        stim2_locs,
        ons=stim_ons,
        offs=stim_offs,
        strengths=stim2_mod1_strengths,
        mods=1,
    )
    trial.add(
        "stim",
        stim1_locs,
        ons=stim_ons,
        offs=stim_offs,
        strengths=stim1_mod2_strengths,
        mods=2,
    )
    trial.add(
        "stim",
        stim2_locs,
        ons=stim_ons,
        offs=stim_offs,
        strengths=stim2_mod2_strengths,
        mods=2,
    )
    trial.add("fix_out", offs=fix_offs)
    stim_locs = np.where(stim1_strengths > stim2_strengths, stim1_locs, stim2_locs)
    trial.add("out", stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim_ons),
        "stim1": (stim_ons, stim_offs),
        "go1": (fix_offs, None),
    }
    return trial


def contextdm1(config, batch_size):
    """Context DM on ring 1: attend modality 1; ignore ring 2."""
    return _contextdm(config, batch_size, attend_mod=1)


def contextdm2(config, batch_size):
    """Context DM on ring 2: attend modality 2; ignore ring 1."""
    return _contextdm(config, batch_size, attend_mod=2)


def multidm(config, batch_size):
    """Multi-modality decision making: integrate both rings; saccade to stronger total."""
    return _contextdm(config, batch_size, attend_mod="both")


def _dms(config, batch_size, matchnogo=False):
    """Delay-match-to-sample: two sequential stimuli; go or nogo on match."""
    dt = config["dt"]
    rng = config["rng"]

    stim1_mod = rng.choice([1, 2])
    stim2_mod = rng.choice([1, 2])
    matches = rng.choice([0, 1], (batch_size,))
    stim_dist = rng.uniform(np.pi / 9, np.pi * 17.0 / 9.0, (batch_size,)) * rng.choice(
        [-1, 1], (batch_size,)
    )
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist * (1 - matches)) % (2 * np.pi)

    stim1_ons = int(rng.choice([200, 400, 600]) / dt)
    stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
    stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    tdim = stim2_ons + int(500 / dt)
    check_ons = stim2_ons + int(100 / dt)

    fix_out_offs = [stim2_ons] * batch_size
    out_offs = [None] * batch_size
    for i in range(batch_size):
        if matches[i] == matchnogo:
            fix_out_offs[i] = None
            out_offs[i] = 0

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in")
    trial.add("stim", stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add("stim", stim2_locs, ons=stim2_ons, mods=stim2_mod)
    trial.add("fix_out", offs=fix_out_offs)
    trial.add("out", stim2_locs, ons=stim2_ons, offs=out_offs)
    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim1_ons),
        "stim1": (stim1_ons, stim1_offs),
        "delay1": (stim1_offs, stim2_ons),
        "go1": (stim2_ons, None),
    }
    return trial


def dmsgo(config, batch_size):
    """Delay-match-to-sample go: saccade on match, hold fixation on non-match."""
    return _dms(config, batch_size, matchnogo=False)


def dmsnogo(config, batch_size):
    """Delay-match-to-sample nogo: saccade on non-match, hold fixation on match."""
    return _dms(config, batch_size, matchnogo=True)


def _delaydm(config, batch_size, stim_mod):
    """Delayed decision making: two sequential bumps; saccade to the stronger."""
    dt = config["dt"]
    rng = config["rng"]

    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice(
        [-1, 1], (batch_size,)
    )
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    stims_mean = rng.uniform(0.8, 1.2, (batch_size,))
    stim_coh_range = np.array([0.08, 0.16, 0.32], dtype=np.float32)
    if config.get("easy_task", True):
        stim_coh_range = stim_coh_range * 2

    stims_coh = rng.choice(stim_coh_range, (batch_size,))
    stims_sign = rng.choice([1, -1], (batch_size,))
    stim1_strengths = stims_mean + stims_coh * stims_sign
    stim2_strengths = stims_mean - stims_coh * stims_sign

    stim1_ons = int(rng.choice([200, 400, 600]) / dt)
    stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
    stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    stim2_offs = stim2_ons + int(rng.choice([200, 400, 600]) / dt)
    fix_offs = stim2_offs + int(rng.uniform(100, 300) / dt)
    tdim = fix_offs + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=fix_offs)
    trial.add(
        "stim",
        stim1_locs,
        ons=stim1_ons,
        offs=stim1_offs,
        strengths=stim1_strengths,
        mods=stim_mod,
    )
    trial.add(
        "stim",
        stim2_locs,
        ons=stim2_ons,
        offs=stim2_offs,
        strengths=stim2_strengths,
        mods=stim_mod,
    )
    trial.add("fix_out", offs=fix_offs)
    stim_locs = np.where(stim1_strengths > stim2_strengths, stim1_locs, stim2_locs)
    trial.add("out", stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim1_ons),
        "stim1": (stim1_ons, stim1_offs),
        "delay1": (stim1_offs, stim2_ons),
        "stim2": (stim2_ons, stim2_offs),
        "delay2": (stim2_offs, fix_offs),
        "go1": (fix_offs, None),
    }
    return trial


def delaydm1(config, batch_size):
    """Delayed DM on ring 1: compare two sequential bumps; saccade to the stronger."""
    return _delaydm(config, batch_size, stim_mod=1)


def delaydm2(config, batch_size):
    """Delayed DM on ring 2: same as delaydm1 but stimuli on ring 2."""
    return _delaydm(config, batch_size, stim_mod=2)


def _contextdelaydm(config, batch_size, attend_mod):
    """Context delayed DM: sequential bumps on both rings; attend one modality or both."""
    dt = config["dt"]
    rng = config["rng"]

    stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice(
        [-1, 1], (batch_size,)
    )
    stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
    stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

    stim_coh_range = np.array([0.08, 0.16, 0.32], dtype=np.float32)
    if config.get("easy_task", True):
        stim_coh_range = stim_coh_range * 2

    if attend_mod in (1, 2):
        stim1_mod1_strengths, stim2_mod1_strengths = _contextdm_genstim(
            batch_size, rng, stim_coh_range
        )
        stim1_mod2_strengths, stim2_mod2_strengths = _contextdm_genstim(
            batch_size, rng, stim_coh_range
        )
        if attend_mod == 1:
            stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
        else:
            stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    else:
        stim1_strengths, stim2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)

        stim1_mod12_diff = (
            stim1_strengths
            * rng.uniform(0.2, 0.8, (batch_size,))
            * rng.choice([1, -1], (batch_size,))
        )
        stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
        stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2

        stim2_mod12_diff = (
            stim2_strengths
            * rng.uniform(0.2, 0.8, (batch_size,))
            * rng.choice([1, -1], (batch_size,))
        )
        stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
        stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

    stim1_ons = int(rng.choice([200, 400, 600]) / dt)
    stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
    stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    stim2_offs = stim2_ons + int(rng.choice([200, 400, 600]) / dt)
    fix_offs = stim2_offs + int(rng.uniform(100, 300) / dt)
    tdim = fix_offs + int(500 / dt)
    check_ons = fix_offs + int(100 / dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == "both":
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in", offs=fix_offs)
    trial.add(
        "stim",
        stim1_locs,
        ons=stim1_ons,
        offs=stim1_offs,
        strengths=stim1_mod1_strengths,
        mods=1,
    )
    trial.add(
        "stim",
        stim2_locs,
        ons=stim2_ons,
        offs=stim2_offs,
        strengths=stim2_mod1_strengths,
        mods=1,
    )
    trial.add(
        "stim",
        stim1_locs,
        ons=stim1_ons,
        offs=stim1_offs,
        strengths=stim1_mod2_strengths,
        mods=2,
    )
    trial.add(
        "stim",
        stim2_locs,
        ons=stim2_ons,
        offs=stim2_offs,
        strengths=stim2_mod2_strengths,
        mods=2,
    )
    trial.add("fix_out", offs=fix_offs)
    stim_locs = np.where(stim1_strengths > stim2_strengths, stim1_locs, stim2_locs)
    trial.add("out", stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim1_ons),
        "stim1": (stim1_ons, stim1_offs),
        "delay1": (stim1_offs, stim2_ons),
        "stim2": (stim2_ons, stim2_offs),
        "delay2": (stim2_offs, fix_offs),
        "go1": (fix_offs, None),
    }
    return trial


def contextdelaydm1(config, batch_size):
    """Context delayed DM on ring 1: attend modality 1 after sequential presentation."""
    return _contextdelaydm(config, batch_size, attend_mod=1)


def contextdelaydm2(config, batch_size):
    """Context delayed DM on ring 2: attend modality 2 after sequential presentation."""
    return _contextdelaydm(config, batch_size, attend_mod=2)


def multidelaydm(config, batch_size):
    """Multi-modality delayed DM: integrate both rings across sequential presentation."""
    return _contextdelaydm(config, batch_size, attend_mod="both")


def _dmc(config, batch_size, matchnogo=False):
    """Delay-match-to-category: go/nogo based on same vs different ring half."""
    dt = config["dt"]
    rng = config["rng"]

    stim1_mod = rng.choice([1, 2])
    stim2_mod = rng.choice([1, 2])
    cat_locs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]) * np.pi
    stim1_locs = rng.choice(cat_locs, size=(batch_size,))
    stim2_locs = rng.choice(cat_locs, size=(batch_size,))

    stim1_ons = int(rng.choice([200, 400, 600]) / dt)
    stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
    stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
    tdim = stim2_ons + int(rng.choice([200, 400, 600]) / dt)
    check_ons = stim2_ons + int(100 / dt)

    matches = (stim1_locs < np.pi) == (stim2_locs < np.pi)

    fix_out_offs = [stim2_ons] * batch_size
    out_offs = [None] * batch_size
    for i in range(batch_size):
        if matches[i] == matchnogo:
            fix_out_offs[i] = None
            out_offs[i] = 0

    trial = Trial(config, tdim, batch_size)
    trial.add("fix_in")
    trial.add("stim", stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add("stim", stim2_locs, ons=stim2_ons, mods=stim2_mod)
    trial.add("fix_out", offs=fix_out_offs)
    trial.add("out", stim2_locs, ons=stim2_ons, offs=out_offs)
    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)
    trial.epochs = {
        "fix1": (None, stim1_ons),
        "stim1": (stim1_ons, stim1_offs),
        "delay1": (stim1_offs, stim2_ons),
        "go1": (stim2_ons, None),
    }
    return trial


def dmcgo(config, batch_size):
    """Delay-match-to-category go: saccade on same category, hold fixation on mismatch."""
    return _dmc(config, batch_size, matchnogo=False)


def dmcnogo(config, batch_size):
    """Delay-match-to-category nogo: saccade on category mismatch, hold fixation on match."""
    return _dmc(config, batch_size, matchnogo=True)


rule_mapping = {
    "fdgo": fdgo,
    "reactgo": reactgo,
    "delaygo": delaygo,
    "fdanti": fdanti,
    "reactanti": reactanti,
    "delayanti": delayanti,
    "dm1": dm1,
    "dm2": dm2,
    "contextdm1": contextdm1,
    "contextdm2": contextdm2,
    "multidm": multidm,
    "delaydm1": delaydm1,
    "delaydm2": delaydm2,
    "contextdelaydm1": contextdelaydm1,
    "contextdelaydm2": contextdelaydm2,
    "multidelaydm": multidelaydm,
    "dmsgo": dmsgo,
    "dmsnogo": dmsnogo,
    "dmcgo": dmcgo,
    "dmcnogo": dmcnogo,
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
