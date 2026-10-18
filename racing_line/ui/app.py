"""Streamlit dashboard: train one algorithm live and inspect its settings, or compare several."""
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from racing_line.baselines import reference_times
from racing_line.problem import LineProblem
from racing_line.track import CIRCUIT_TITLES, DEFAULT_CIRCUIT, build_track
from racing_line.ui.parameters import BLURBS, PARAMS, defaults, describe
from racing_line.ui.rendering import TITLES, Dashboard
from racing_line.ui.simulation import evals_to_beat, simulate, single_view_limits, timeline, train_live

ROOT = Path(__file__).resolve().parents[2]
RESULT_FOLDERS = (ROOT / "results", ROOT / "reference_results")
PLAYBACK_FRAMES = 80
LIVE_FRAMES = 40
FRAME_SECONDS = 0.25
PLAY_DPI, REST_DPI = 90, 130

SINGLE, COMPARE = "Watch one algorithm train", "Compare algorithms"
BUDGET_HELP = "Total laps the algorithm may simulate. More laps give it longer to learn."
SEED_HELP = "The same seed and settings always give the same run."
BENCHMARK_FIGURES = [
    ("convergence.png", "Mean best lap time against evaluations over 10 seeds (shaded band is one standard deviation)"),
    ("variations.png", "Final lap times for each crossover type, DE strategy and PSO topology"),
    ("sweeps.png", "Effect of the main parameters on the mean final lap time"),
    ("track_lines.png", "Reference lines and the best evolved line, coloured by speed"),
    ("speed_profile.png", "Speed along the lap: minimum curvature line against the best evolved line"),
]


@st.cache_resource(show_spinner="Building the track and reference lines")
def load_environment(circuit):
    problem = LineProblem(build_track(circuit))
    return (problem, *reference_times(problem))


@st.cache_data(show_spinner=False)
def cached_trace(_problem, circuit, algo, params, budget, seed):
    return simulate(_problem, algo, dict(params), budget, seed)


def param_controls(prefix, algo):
    """One widget per setting of the algorithm, returning {setting name: chosen value}."""
    values = {}
    for p in PARAMS[algo]:
        key = f"{prefix}_{algo}_{p.name}"
        if p.options:
            choices = list(p.options)
            values[p.name] = st.selectbox(p.label, choices, choices.index(p.default),
                                          format_func=p.options.get, help=p.help, key=key)
        else:
            values[p.name] = st.slider(p.label, p.low, p.high, p.default, p.step, help=p.help, key=key)
    return values


def replay(problem, lines, base_t, traces, controls, slot, limits=None):
    """Play, pause and scrub through finished runs. controls and slot are placeholders made by the caller."""
    grid = timeline(traces, PLAYBACK_FRAMES)
    if st.session_state.get("progress") not in grid:
        st.session_state.progress = grid[-1]

    several = len(traces) > 1
    with controls:
        columns = st.columns([1, 1, 2, 3] if several else [1, 1, 2])
        play = columns[0].button("Play", type="primary", key="play")
        columns[1].button("Pause", key="pause")
        speed = columns[2].select_slider("Playback speed", [0.5, 1, 2, 4], value=1,
                                         format_func=lambda s: f"{s}x", key="speed")
        focus = (columns[3].selectbox("Speed profile of", list(traces), format_func=TITLES.get, key="focus")
                 if several else None)
        slider_slot = st.container()

    dashboard = Dashboard(problem, lines, base_t, traces, focus, **(limits or {}))
    if play:
        start = grid.index(st.session_state.progress)
        for evals in grid[0 if start == len(grid) - 1 else start:]:
            st.session_state.progress = evals
            slot.image(dashboard.png(evals, PLAY_DPI))
            time.sleep(FRAME_SECONDS / speed)

    with slider_slot:
        evals = st.select_slider("Learning progress (evaluations used)", grid, key="progress",
                                 format_func=lambda e: f"{e:,}")
    if not play:
        slot.image(dashboard.png(evals, REST_DPI))


def single_sidebar():
    algo = st.sidebar.selectbox("Algorithm to train", list(TITLES), format_func=TITLES.get, key="single_algo")
    st.sidebar.caption(BLURBS[algo])
    with st.sidebar.form(f"single_{algo}"):
        st.subheader(f"{TITLES[algo]} settings")
        params = param_controls("single", algo)
        budget = st.slider("Evaluation budget", 6_000, 50_000, 30_000, 2_000, key="single_budget", help=BUDGET_HELP)
        seed = int(st.number_input("Random seed", 0, 9_999, 0, key="single_seed", help=SEED_HELP))
        train = st.form_submit_button("Train", type="primary", key="train")
    return algo, params, budget, seed, train


def run_summary(base_t, run):
    trace, algo = run["trace"], run["algo"]
    best = float(trace.best_f[-1])
    beat = evals_to_beat(trace, base_t["min_curvature"])
    columns = st.columns(4)
    columns[0].metric("Best lap", f"{best:.3f} s")
    columns[1].metric("Gain vs minimum curvature", f"{base_t['min_curvature'] - best:+.3f} s")
    columns[2].metric("Evaluations to beat it", f"{beat:,}" if beat else "not reached")
    columns[3].metric("Final population spread", f"{trace.diversity[-1]:.3f}")

    st.subheader(f"{TITLES[algo]}: the settings used")
    rows = describe(algo, run["params"]) + [
        {"Setting": "Evaluation budget", "Value": f"{run['budget']:,}", "What it does": BUDGET_HELP},
        {"Setting": "Random seed", "Value": str(run["seed"]), "What it does": SEED_HELP},
    ]
    st.table(pd.DataFrame(rows).set_index("Setting"))


def single_page(problem, lines, base_t):
    algo, params, budget, seed, train = single_sidebar()
    st.title("Racing line optimisation")
    st.caption("Pick an algorithm, adjust its settings and press Train to watch it learn the fastest line.")
    training, benchmark = st.tabs(["Training", "Benchmark over 10 seeds"])
    with benchmark:
        benchmark_tab(problem.track.name)

    with training:
        controls, slot = st.container(), st.empty()
        if train:
            bar = st.progress(0.0, text=f"Training {TITLES[algo]}")

            def draw(png, evals):
                slot.image(png)
                bar.progress(min(evals / budget, 1.0))

            trace = train_live(problem, lines, base_t, algo, params, budget, seed, draw, LIVE_FRAMES)
            bar.empty()
            st.session_state.single_run = {"circuit": problem.track.name, "algo": algo, "params": params,
                                           "budget": budget, "seed": seed, "trace": trace}
            st.session_state.progress = int(trace.evals[-1])

        run = st.session_state.get("single_run")
        if run is None or run["circuit"] != problem.track.name:
            st.info("Choose an algorithm in the sidebar, adjust its settings and press Train.")
            return
        replay(problem, lines, base_t, {run["algo"]: run["trace"]}, controls, slot,
               single_view_limits(base_t, run["budget"]))
        run_summary(base_t, run)


def compare_sidebar():
    chosen = st.sidebar.multiselect("Algorithms to compare", list(TITLES), default=["ga", "de", "pso"],
                                    format_func=TITLES.get, key="compare_algos")
    with st.sidebar.form("compare"):
        budget = st.slider("Evaluation budget", 6_000, 50_000, 30_000, 2_000, key="compare_budget", help=BUDGET_HELP)
        seed = int(st.number_input("Random seed", 0, 9_999, 0, key="compare_seed", help=SEED_HELP))
        params = {"random": defaults("random")}
        for algo in ("ga", "de", "pso"):
            with st.expander(f"{TITLES[algo]} settings"):
                params[algo] = param_controls("compare", algo)
        st.form_submit_button("Run", type="primary", key="run")
    return chosen, budget, seed, params


def summary_tab(base_t, traces):
    st.subheader("Result of this run")
    for column, (algo, trace) in zip(st.columns(len(traces)), traces.items()):
        gain = base_t["min_curvature"] - trace.best_f[-1]
        column.metric(TITLES[algo], f"{trace.best_f[-1]:.3f} s", f"{gain:+.3f} s vs minimum curvature")

    rows = []
    for algo, trace in traces.items():
        best = trace.best_f[-1]
        rows.append({
            "Algorithm": TITLES[algo],
            "Best lap (s)": round(float(best), 3),
            "Gain vs minimum curvature (s)": round(float(base_t["min_curvature"] - best), 3),
            "Gain vs centreline (s)": round(float(base_t["centreline"] - best), 3),
            "Evaluations to beat minimum curvature": evals_to_beat(trace, base_t["min_curvature"]),
            "Evaluations used": int(trace.evals[-1]),
        })
    table = pd.DataFrame(rows)
    table["Evaluations to beat minimum curvature"] = table["Evaluations to beat minimum curvature"].astype("Int64")
    st.dataframe(table, hide_index=True)

    st.subheader("Reference lines")
    reference = pd.DataFrame({"Line": [n.replace("_", " ").capitalize() for n in base_t],
                              "Lap time (s)": [round(t, 3) for t in base_t.values()]})
    st.dataframe(reference, hide_index=True)
    st.caption("One run with one seed. The benchmark tab shows ten seeds with statistics.")


def compare_page(problem, lines, base_t):
    chosen, budget, seed, params = compare_sidebar()
    st.title("Racing line optimisation")
    st.caption("Run several algorithms on the same budget and watch them side by side. "
               "Press Play, or drag the progress slider.")
    if not chosen:
        st.info("Select at least one algorithm in the sidebar and press Run.")
        return

    with st.spinner("Running the optimisers"):
        traces = {algo: cached_trace(problem, problem.track.name, algo, tuple(sorted(params[algo].items())),
                                     budget, seed)
                  for algo in chosen}

    live, summary, benchmark = st.tabs(["Live learning", "Performance summary", "Benchmark over 10 seeds"])
    with live:
        replay(problem, lines, base_t, traces, st.container(), st.empty())
    with summary:
        summary_tab(base_t, traces)
    with benchmark:
        benchmark_tab(problem.track.name)


def results_folder(circuit, marker):
    """First results folder holding the marker file for this circuit (the default circuit sits at the top)."""
    sub = "" if circuit == DEFAULT_CIRCUIT else circuit
    return next((f / sub for f in RESULT_FOLDERS if (f / sub / marker).exists()), None)


def benchmark_tab(circuit):
    folder = results_folder(circuit, "summary.csv")
    if folder is None:
        st.info(f"No full benchmark for this circuit yet. Run `python run_experiments.py --circuit {circuit}`.")
    else:
        st.caption(f"Loaded from {folder.relative_to(ROOT)}/")
        st.dataframe(pd.read_csv(folder / "summary.csv"), hide_index=True)
        for name, caption in BENCHMARK_FIGURES:
            if (folder / name).exists():
                st.image(str(folder / name), caption=caption)

    across = results_folder(DEFAULT_CIRCUIT, "circuits.csv")
    if across is not None:
        st.subheader("The same algorithms on every circuit")
        st.dataframe(pd.read_csv(across / "circuits.csv"), hide_index=True)
        st.image(str(across / "circuits.png"), caption="Best evolved line on each circuit and the gain over "
                                                       "the minimum curvature line")


def main():
    st.set_page_config(page_title="Racing line optimisation", layout="wide")
    mode = st.sidebar.radio("What do you want to see?", [SINGLE, COMPARE], key="mode")
    circuit = st.sidebar.selectbox("Circuit", list(CIRCUIT_TITLES), format_func=CIRCUIT_TITLES.get, key="circuit")
    problem, lines, base_t = load_environment(circuit)
    page = single_page if mode == SINGLE else compare_page
    page(problem, lines, base_t)
