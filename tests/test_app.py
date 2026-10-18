from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from racing_line.ui import app

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def at(monkeypatch):
    monkeypatch.setattr(app, "LIVE_FRAMES", 3)
    monkeypatch.setattr(app, "PLAYBACK_FRAMES", 6)
    monkeypatch.setattr(app, "FRAME_SECONDS", 0)
    return AppTest.from_file(APP, default_timeout=180)


def train(at, algo="pso", budget=6_000, **settings):
    at.run()
    at.sidebar.selectbox(key="single_algo").set_value(algo).run()
    at.sidebar.slider(key="single_budget").set_value(budget)
    for name, value in settings.items():
        at.sidebar.slider(key=f"single_{algo}_{name}").set_value(value)
    at.sidebar.button(key="train").click().run()
    return at


def test_default_view_waits_for_the_user_to_train(at):
    at.run()
    assert not at.exception
    assert any("press Train" in i.value for i in at.info)
    assert len(at.metric) == 0


def test_training_one_algorithm_shows_its_result_and_its_settings(at):
    train(at)
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Best lap", "Gain vs minimum curvature"]
    assert len(at.table) == 1
    assert at.session_state.single_run["algo"] == "pso"


def test_the_sidebar_only_offers_the_settings_of_the_chosen_algorithm(at):
    at.run()
    ga_keys = {s.key for s in at.sidebar.slider}
    assert {"single_ga_pm", "single_ga_sigma", "single_ga_elite"} <= ga_keys
    assert not any(k.startswith("single_pso_") for k in ga_keys)

    at.sidebar.selectbox(key="single_algo").set_value("pso").run()
    pso_keys = {s.key for s in at.sidebar.slider}
    assert {"single_pso_w", "single_pso_c1", "single_pso_c2", "single_pso_vmax"} <= pso_keys
    assert not any(k.startswith("single_ga_") for k in pso_keys)


def test_changing_a_setting_changes_the_trained_result(at):
    first = train(at, w=0.4).session_state.single_run["trace"].best_f[-1]
    second = train(at, w=0.9).session_state.single_run["trace"].best_f[-1]
    assert first != second


def test_the_settings_table_reports_what_was_used(at):
    train(at, algo="ga", pm=0.12)
    rows = {row["Setting"]: row["Value"] for _, row in at.table[0].value.reset_index().iterrows()}
    assert rows["Mutation rate per gene"] == "0.12"
    assert rows["Evaluation budget"] == "6,000"


def test_play_replays_a_trained_run_to_the_end(at):
    train(at)
    at.button(key="play").click().run()
    assert not at.exception
    last_label = at.select_slider(key="progress").options[-1]
    assert at.session_state.progress == int(last_label.replace(",", ""))


def test_compare_mode_runs_every_chosen_algorithm(at):
    at.run()
    at.sidebar.radio(key="mode").set_value(app.COMPARE).run()
    assert not at.exception
    assert {m.label for m in at.metric} == {"Genetic algorithm", "Differential evolution", "Particle swarm"}


def test_compare_mode_can_be_narrowed_and_replayed(at):
    at.run()
    at.sidebar.radio(key="mode").set_value(app.COMPARE).run()
    at.sidebar.multiselect(key="compare_algos").set_value(["ga", "pso"])
    at.sidebar.slider(key="compare_budget").set_value(6_000)
    at.sidebar.button(key="run").click().run()
    assert [m.label for m in at.metric] == ["Genetic algorithm", "Particle swarm"]
    at.button(key="play").click().run()
    assert not at.exception


def test_compare_mode_without_algorithms_shows_a_hint(at):
    at.run()
    at.sidebar.radio(key="mode").set_value(app.COMPARE).run()
    at.sidebar.multiselect(key="compare_algos").set_value([])
    at.sidebar.button(key="run").click().run()
    assert not at.exception
    assert any("Select at least one algorithm" in i.value for i in at.info)


def test_the_circuit_selector_lists_every_circuit(at):
    at.run()
    assert at.sidebar.selectbox(key="circuit").value == "grand_prix"
    assert set(at.sidebar.selectbox(key="circuit").options) == {
        "Grand Prix (4.6 km, mixed)", "Technical (2.2 km, tight hairpins)", "Flowing (4.7 km, fast sweepers)"}


def test_changing_circuit_discards_the_earlier_run_and_trains_on_the_new_one(at):
    first = train(at).session_state.single_run
    assert first["circuit"] == "grand_prix"
    at.sidebar.selectbox(key="circuit").set_value("technical").run()
    assert any("press Train" in i.value for i in at.info)
    second = train(at).session_state.single_run
    assert second["circuit"] == "technical"
    assert second["trace"].best_f[-1] < first["trace"].best_f[-1] - 10


def test_compare_mode_gives_each_circuit_its_own_results(at):
    at.run()
    at.sidebar.radio(key="mode").set_value(app.COMPARE).run()
    grand_prix = [m.value for m in at.metric]
    at.sidebar.selectbox(key="circuit").set_value("flowing").run()
    assert not at.exception
    assert [m.value for m in at.metric] != grand_prix


def test_benchmark_tab_points_to_the_command_when_a_circuit_has_no_results(at, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "RESULT_FOLDERS", (tmp_path,))
    at.run()
    at.sidebar.selectbox(key="circuit").set_value("technical").run()
    assert any("--circuit technical" in i.value for i in at.info)
