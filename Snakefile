# Selectivity pipeline (spec and plan: notes/idea.md). Run with `pixi run pipeline`.
# checkpoint -> per-unit table -> per-network summary -> Wrong-if verdicts / figure.

from cmc.paths import MOO_ZOOM_RUNS_DIR

SETS = {
    "150k": dict(dir=MOO_ZOOM_RUNS_DIR / "dale_all_woutfree_fanin_4pt_5seed_150k",
                 points=["control", "A3", "B0", "C0"]),
    "50k": dict(dir=MOO_ZOOM_RUNS_DIR / "dale_all_4pt_12seed_50k",
                points=["control", "rate_limited", "wiring_limited", "knee"]),
}
THRESHOLDS = ["0.3", "0.5", "0.7"]
OUT = "results/selectivity"


def seeds(s, point):
    # final checkpoints only: {point}/seed_XX.pt, not the step_025000/ snapshots
    return sorted(p.stem.removeprefix("seed_") for p in (SETS[s]["dir"] / point).glob("seed_*.pt"))


def unit_tables(s, thr):
    return [f"{OUT}/{s}/thr{thr}/units/{p}/seed_{sd}.csv"
            for p in SETS[s]["points"] for sd in seeds(s, p)]


wildcard_constraints:
    set="|".join(SETS), thr=r"0\.\d+", seed=r"\d+"


rule all:
    input:
        expand(f"{OUT}/{{set}}/thr{{thr}}/networks.csv", set=SETS, thr=THRESHOLDS),
        f"{OUT}/wrong_if.csv",
        f"{OUT}/breadth.png",


rule unit_table:
    input:
        lambda w: SETS[w.set]["dir"] / w.point / f"seed_{w.seed}.pt",
    output:
        f"{OUT}/{{set}}/thr{{thr}}/units/{{point}}/seed_{{seed}}.csv",
    run:
        from cmc.selectivity import unit_table
        unit_table(input[0], float(wildcards.thr)).to_csv(output[0], index=False)


rule summary:
    input:
        lambda w: unit_tables(w.set, w.thr),
    output:
        f"{OUT}/{{set}}/thr{{thr}}/networks.csv",
    run:
        import pandas as pd
        from cmc.selectivity import network_summary
        pd.DataFrame([network_summary(pd.read_csv(f)) for f in input]).to_csv(output[0], index=False)


rule verdicts:
    input:
        expand(f"{OUT}/{{set}}/thr{{thr}}/networks.csv", set=["150k"], thr=THRESHOLDS),
        f"{OUT}/50k/thr0.5/networks.csv",
    output:
        f"{OUT}/wrong_if.csv",
    run:
        import pandas as pd
        from cmc.selectivity import wrong_if
        nets = {(f.split("/")[-3], f.split("/")[-2].removeprefix("thr")): pd.read_csv(f) for f in input}
        wrong_if(nets).to_csv(output[0], index=False)


rule figure:
    input:
        expand(f"{OUT}/150k/thr{{thr}}/networks.csv", thr=THRESHOLDS),
        f"{OUT}/50k/thr0.5/networks.csv",
    output:
        f"{OUT}/breadth.png",
    run:
        import pandas as pd
        from cmc.selectivity import plot_breadth
        nets = {(f.split("/")[-3], f.split("/")[-2].removeprefix("thr")): pd.read_csv(f) for f in input}
        plot_breadth(nets, {s: SETS[s]["points"] for s in SETS}, output[0])
