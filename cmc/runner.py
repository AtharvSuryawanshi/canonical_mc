"""Train and evaluate one network: the code path every experiment script shares.

``cmc.lambda_pareto``, ``cmc.lambda_zoom`` and ``cmc.moo`` all train through
``train_and_evaluate`` with the flags from ``add_common_args``, so a network is
the same network whichever script produced it.
"""

import argparse
import csv

import torch

from cmc.front import FEASIBLE_MIN_TASK_ACC
from cmc.task import default_config, rules_dict
from cmc.train_cog import (
    DALE_DEFAULT_NOISE_LEVEL,
    _model_kwargs_from_args,
    evaluate_pareto_metrics,
    make_dale_model,
    make_yang_model,
    resolve_active_tasks,
    train_without_plots,
)


# Fixed eval seeds for comparable runs (see lambda_pareto_analysis.ipynb calibration).
DEFAULT_EVAL_SEEDS = tuple(10000 + i for i in range(10))


def make_fresh_model(args, config, device, seed=None):
    model_kwargs = _model_kwargs_from_args(args)
    if seed is not None:
        model_kwargs["seed"] = int(seed)
    if args.model == "yang":
        return make_yang_model(config, device=device, **model_kwargs)
    return make_dale_model(config, device=device, **model_kwargs)


def append_csv_row(csv_path, row, write_header=False):
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_eval_seeds(text):
    if text is None or str(text).strip() == "":
        return DEFAULT_EVAL_SEEDS
    return tuple(int(s.strip()) for s in str(text).split(",") if s.strip())


def add_common_args(parser, n_seeds=True):
    """Model, training, regularizer and eval flags every experiment script takes.

    Script-specific defaults (e.g. core5, 4000 steps) are set afterwards with
    ``parser.set_defaults``. ``n_seeds=False`` omits --n-seeds (cmc.moo trains
    one seed per genome).
    """
    parser.add_argument("--model", choices=["yang", "dale"], default="dale")
    parser.add_argument(
        "--task-battery",
        choices=["all", "core5", "sanity3"],
        default="all",
        help="Preset task subset (ignored when --tasks is set).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=rules_dict["all"],
        help="Explicit task list (overrides --task-battery).",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-rnn", type=int, default=256)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--n-eachring", type=int, default=16)
    parser.add_argument("--frac-e", type=float, default=0.8)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--sigma-rec", type=float, default=0.05)
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Default: 1.0 for yang, 0.1 for dale.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--feasible-min-task-acc",
        type=float,
        default=FEASIBLE_MIN_TASK_ACC,
        help=f"Worst-task accuracy a network needs to be a front candidate "
        f"(default {FEASIBLE_MIN_TASK_ACC}).",
    )
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default=None,
        help=f"Comma-separated trial RNG seeds for eval (default: {len(DEFAULT_EVAL_SEEDS)} fixed seeds).",
    )
    if n_seeds:
        parser.add_argument(
            "--n-seeds",
            type=int,
            default=1,
            help="Independent training seeds per lambda point. Every grid point used "
            "the same init and the same data stream, so a single unlucky init became "
            "a 'Pareto point' with no error bar. >1 gives mean +- std per lambda; the "
            "front is computed on the per-lambda means.",
        )
    parser.add_argument("--rate-kind", choices=["l1", "l2"], default="l2")
    parser.add_argument("--rate-inh-scale", type=float, default=1.0)
    parser.add_argument("--conn-kind", choices=["l1", "l2"], default="l1")
    parser.add_argument("--conn-target", choices=["w_rec", "w_in"], default="w_rec")
    parser.add_argument("--conn-inh-scale", type=float, default=1.0)
    parser.add_argument("--prune-eps", type=float, default=0.0)
    parser.add_argument(
        "--loss-per-trial", dest="loss_per_trial", action="store_true", default=True
    )
    parser.add_argument(
        "--no-loss-per-trial", dest="loss_per_trial", action="store_false"
    )
    parser.add_argument(
        "--eval-noise-level",
        type=float,
        default=0.0,
        help="Recurrent noise during Pareto eval. 0.0 (default) keeps the clean, "
        "reproducible readout; raise it to make the front reward noise robustness.",
    )
    parser.add_argument(
        "--eval-input-noise",
        action="store_true",
        help="Also apply input noise during Pareto eval.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def resolve_run_settings(args):
    """(active_tasks, battery_label, device, noise_level, eval_seeds, reg_opts)."""
    active_tasks = resolve_active_tasks(args.task_battery, args.tasks)
    battery_label = args.task_battery if args.tasks is None else "custom"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    noise_level = (
        args.noise_level
        if args.noise_level is not None
        else (1.0 if args.model == "yang" else DALE_DEFAULT_NOISE_LEVEL)
    )
    reg_opts = {
        "rate_kind": args.rate_kind,
        "rate_inh_scale": args.rate_inh_scale,
        "conn_kind": args.conn_kind,
        "conn_target": args.conn_target,
        "conn_inh_scale": args.conn_inh_scale,
    }
    return active_tasks, battery_label, device, noise_level, parse_eval_seeds(args.eval_seeds), reg_opts


def train_and_evaluate(args, lambda_rate, lambda_connectivity, seed, active_tasks,
                       device, noise_level, eval_seeds, reg_opts, budget_opts=None):
    """Train one network and evaluate it on the fixed eval seeds.

    Returns (model, config, history, metrics). Used by lambda_pareto, lambda_zoom
    and moo, so every script produces the identical network for a given
    (lambda or budget, seed).
    ``budget_opts`` (``rate_budget``, ``conn_budget``, ``budget_lr``, ...) are
    passed to ``train`` for budget-constrained training; pass zero lambdas then.
    """
    config = default_config(n_eachring=args.n_eachring, seed=seed, easy_task=True)
    model = make_fresh_model(args, config, device, seed=seed)
    history = train_without_plots(
        model,
        config,
        active_tasks=active_tasks,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_rate=float(lambda_rate),
        lambda_connectivity=float(lambda_connectivity),
        noise_level=noise_level,
        log_every=args.log_every,
        show_progress=False,
        loss_per_trial=args.loss_per_trial,
        **(budget_opts or {}),
        **reg_opts,
    )
    model.eval()
    metrics = evaluate_pareto_metrics(
        model,
        config,
        active_tasks,
        device,
        eval_seeds=eval_seeds,
        batch_size=args.eval_batch_size,
        noise_level=args.eval_noise_level,
        input_noise=args.eval_input_noise,
        easy_task=True,
        **reg_opts,
    )
    return model, config, history, metrics
