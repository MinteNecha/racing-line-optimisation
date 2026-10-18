"""Which algorithm configurations are compared and which parameters are swept."""

NAMES = {
    ("ga", "blx"): "GA (BLX)", ("ga", "arith"): "GA (arithmetic)", ("ga", "uniform"): "GA (uniform)",
    ("de", "rand1"): "DE/rand/1", ("de", "best1"): "DE/best/1", ("de", "ctb1"): "DE/current-to-best/1",
    ("pso", "gbest"): "PSO (gbest)", ("pso", "lbest"): "PSO (lbest)",
    ("random", ""): "Random search",
}
VARIANT_PARAM = {"ga": "cross", "de": "strategy", "pso": "topology", "random": None}
HEADLINE = [("ga", "blx"), ("de", "rand1"), ("pso", "gbest")]
STUDY = HEADLINE + [("random", "")]

SWEEP_GRIDS = {
    "ga": ("pop", [20, 40, 80, 160], "pm", [0.01, 0.05, 0.1, 0.2]),
    "de": ("F", [0.3, 0.5, 0.7, 0.9], "CR", [0.1, 0.5, 0.9]),
    "pso": ("w", [0.4, 0.6, 0.72, 0.9], "pop", [20, 60, 120]),
}


def main_configs():
    """Every algorithm and variant at default parameters: {label: (algorithm, params)}."""
    configs = {}
    for algo, variant in NAMES:
        param = VARIANT_PARAM[algo]
        configs[(algo, variant)] = (algo, {param: variant} if param else {})
    return configs


def study_configs():
    """The headline algorithms and the control at default parameters, for the cross circuit study."""
    configs = main_configs()
    return {key: configs[key] for key in STUDY}


def sweep_configs():
    """Two parameter grid per algorithm: {(algorithm, a, b): (algorithm, params)}."""
    configs = {}
    for algo, (name_a, values_a, name_b, values_b) in SWEEP_GRIDS.items():
        for a in values_a:
            for b in values_b:
                configs[(algo, a, b)] = (algo, {name_a: a, name_b: b})
    return configs
