from argparse import Namespace

import numpy as np
import pytest

import run_experiments
from racing_line.config import Settings
from racing_line.experiments import plots
from racing_line.experiments.circuits import study
from racing_line.experiments.configs import NAMES, STUDY, main_configs, study_configs, sweep_configs
from racing_line.experiments.report import best_evolved, circuit_rows, summary_rows, write_circuits
from racing_line.experiments.runner import finals, run_all
from racing_line.optimisers import OPTIMISERS

TINY = {("ga", "blx"): ("ga", {}), ("de", "rand1"): ("de", {}), ("random", ""): ("random", {})}


def test_every_configured_algorithm_exists_and_is_named():
    configs = main_configs()
    assert set(configs) == set(NAMES)
    assert all(algo in OPTIMISERS for algo, _ in configs.values())


def test_sweep_grids_cover_every_algorithm():
    configs = sweep_configs()
    assert len(configs) == 16 + 12 + 12
    assert {algo for algo, _ in configs.values()} == {"ga", "de", "pso"}


def test_run_all_returns_one_result_per_seed(problem):
    runs = run_all(problem, TINY, seeds=3, budget=600, workers=1)
    assert set(runs) == set(TINY)
    assert all(len(results) == 3 for results in runs.values())


def test_parallel_runs_match_serial_runs(problem):
    serial = run_all(problem, TINY, seeds=2, budget=600, workers=1)
    parallel = run_all(problem, TINY, seeds=2, budget=600, workers=2)
    for label in TINY:
        assert np.allclose(finals(serial[label]), finals(parallel[label]))


def test_report_helpers(problem):
    runs = run_all(problem, TINY, seeds=2, budget=600, workers=1)
    base_t = {"min_curvature": 62.0, "centreline": 64.0}
    rows = summary_rows(runs, base_t)
    assert len(rows) == len(TINY)
    assert best_evolved(runs).fitness == min(
        r.fitness for key, rs in runs.items() if key[0] != "random" for r in rs)


def test_study_configs_are_the_headline_algorithms_and_the_control():
    assert set(study_configs()) == set(STUDY)
    assert ("random", "") in study_configs()


def test_study_runs_every_algorithm_on_every_circuit():
    results = study(["technical", "flowing"], seeds=2, budget=600, workers=1)
    assert list(results) == ["technical", "flowing"]
    for base_t, runs in results.values():
        assert set(runs) == set(STUDY)
        assert all(len(r) == 2 for r in runs.values())
        assert base_t["min_curvature"] < base_t["centreline"]
    rows = circuit_rows(results)
    assert len(rows) == 2 * len(STUDY)
    row = rows[0]
    assert np.isclose(row["gain_s"], row["min_curvature_s"] - row["mean_s"])
    assert np.isclose(row["gain_pct"], 100 * row["gain_s"] / row["min_curvature_s"])


def test_study_results_are_written_and_plotted(tmp_path):
    results = study(["technical", "flowing"], seeds=2, budget=600, workers=1)
    rows = circuit_rows(results)
    write_circuits(rows, tmp_path / "circuits.csv")
    plots.circuits(rows, results, tmp_path / "circuits.png")
    assert len((tmp_path / "circuits.csv").read_text().splitlines()) == 1 + len(rows)
    assert (tmp_path / "circuits.png").stat().st_size > 10_000


TINY_SETTINGS = Settings(budget=600, seeds=2, sweep_seeds=1)


def test_command_line_study_writes_its_outputs(tmp_path):
    out = run_experiments.circuit_study(Namespace(out=tmp_path, workers=1), TINY_SETTINGS)
    assert (out / "circuits.csv").exists() and (out / "circuits.png").exists()


def test_command_line_benchmark_for_another_circuit_goes_to_its_own_folder(tmp_path):
    out = run_experiments.benchmark(Namespace(out=tmp_path, circuit="technical", workers=1), TINY_SETTINGS)
    assert out == tmp_path / "technical"
    assert {"summary.csv", "best_line.npy", "convergence.png", "sweeps.png"} <= {f.name for f in out.iterdir()}


def test_command_line_rejects_an_unknown_circuit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_experiments.py", "--circuit", "monza"])
    with pytest.raises(SystemExit):
        run_experiments.parse_args()
