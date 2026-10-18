# Racing Line Optimisation

Evolutionary and swarm algorithms that search for a fast racing line around a closed circuit. Built for the Optimisation honours module research project.

Three fictional circuits (mixed, tight and technical, fast and flowing) are included so the findings can be checked beyond a single track.

A genetic algorithm (GA), differential evolution (DE) and particle swarm optimisation (PSO) each choose sideways offsets from the track centreline. A simple car model turns every candidate line into a lap time, and that lap time is the fitness to minimise. The results are compared against three classical reference lines and a random search control.

## Project layout

```
racing-line-optimisation/
  run_experiments.py            command line entry point for the benchmark
  app.py                        starts the live dashboard (streamlit run app.py)
  racing_line/
    config.py                   car constants and experiment settings
    track.py                    the three circuits, their geometry and usable track width
    vehicle.py                  lap time model (numba compiled)
    problem.py                  decision vector to line geometry to lap time
    baselines.py                centreline, shortest path, minimum curvature
    recording.py                records every generation of a run for replay
    optimisers/
      base.py                   Result type and shared helpers
      genetic.py                real coded GA (BLX, arithmetic, uniform crossover)
      differential.py           DE (rand/1, best/1, current-to-best/1)
      swarm.py                  PSO (gbest, lbest ring)
      random_search.py          control model
    ui/
      app.py                    Streamlit dashboard (single algorithm and compare views)
      parameters.py             each algorithm's settings with plain English explanations
      rendering.py              dashboard figure (tracks, close-ups, curves, diversity, speed)
      simulation.py             live training, recorded runs and the replay timeline
    experiments/
      configs.py                algorithm variants and parameter grids
      runner.py                 multi-seed, multi-process execution
      circuits.py               the headline algorithms on every circuit
      plots.py                  figures
      report.py                 tables and significance tests
  tests/                        pytest suite
  results/                      your generated figures and tables
  reference_results/            output of the full run, to compare against
```

## Setup

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt     # runtime + dashboard + pytest
```

If you only need the benchmark and not the dashboard, `pip install -r requirements.txt` is enough.

## Run the tests

```
pytest
```

The suite takes about 2 minutes. The dashboard tests are skipped if Streamlit is not installed. It checks the geometry of every circuit, the lap time model against circles with known answers, the problem encoding, the baselines, every optimiser variant (validity, reproducibility, budget accounting, improvement over random search), the experiment runner and the cross circuit study, the run recorder, every setting range exposed in the dashboard, live training, the dashboard figure and the dashboard app itself (training, changing settings, Play and compare mode).

## Live dashboard

```
streamlit run app.py
```

It opens in your browser (if Streamlit asks for an email on the first run, press Enter). The sidebar has two views and a **Circuit** selector, so any algorithm can be trained on any of the three circuits:

| Circuit | Length | Character |
|---|---|---|
| Grand Prix (default) | 4.6 km | mixed corners, one tight chicane |
| Technical | 2.2 km | tight hairpins, small gains from top speed |
| Flowing | 4.7 km | long straights and fast sweepers, speed often capped by the car's top speed |

Changing the circuit clears the previous run, and every circuit has its own reference lines (centreline, shortest path, minimum curvature).

**Watch one algorithm train (default).** Choose an algorithm, adjust its own settings, set the evaluation budget and seed, and press Train. The algorithm then trains in front of you and the dashboard redraws as it learns:

- the track with a sample of the current population (faint lines), the best line so far (bold) and the best lap time;
- a close-up of the tightest corner, where early on the population is spread across the track and later collapses onto one line;
- the learning curve against the minimum curvature and centreline reference lines;
- a diversity chart showing how spread out the population is, so you can see the algorithm explore first and then settle (differential evolution settles slowly, particle swarm quickly);
- the speed of the best line against the minimum curvature line.

When training ends you can press Play to replay it, drag the progress slider to any point, and read the result below: best lap, gain over the minimum curvature line, evaluations needed to beat it and a table of the settings used with a plain English explanation of each.

Settings you can change per algorithm:

| Algorithm | Settings |
|---|---|
| Genetic algorithm | population size, crossover probability, mutation rate, mutation size, elites kept, tournament size, crossover type |
| Differential evolution | population size, scale factor F, crossover rate CR, mutation strategy |
| Particle swarm | swarm size, inertia weight, personal pull c1, group pull c2, speed limit, neighbourhood |
| Random search | batch size |

Hover the question mark next to a setting for its explanation.

**Compare algorithms.** Runs several algorithms on the same budget and replays them side by side, with a performance summary tab (final lap times, gains, evaluations needed to beat the reference line).

Both views also have a benchmark tab with the ten seed statistics and figures from `results/` (or `reference_results/` if you have not run the benchmark yet), followed by the cross circuit study when `circuits.csv` exists. Full benchmark figures exist for the Grand Prix circuit by default; for another circuit the tab shows the command that creates them.

`reference_results/` also holds example dashboard frames for the report: `single_ga_end.png`, `single_de_end.png`, `single_pso_early.png`, `single_pso_end.png` and the three side by side frames `dashboard_early.png`, `dashboard_mid.png`, `dashboard_end.png`.

Training with live drawing takes about 20 seconds at the default budget, because drawing each frame takes longer than the optimisation itself.

## Run the experiments

```
python run_experiments.py --quick      # smoke test, about 1.5 minutes
python run_experiments.py              # full run, about 10 minutes on 2 cores
python run_experiments.py --circuit technical   # the same benchmark on another circuit
python run_experiments.py --circuits   # GA, DE, PSO and random search on every circuit, about 5 minutes
```

Options: `--workers N` sets the number of processes (use `--workers 1` to debug), `--circuit NAME` picks the circuit (`grand_prix`, `technical` or `flowing`), `--circuits` runs the cross circuit study instead of the full benchmark, `--out DIR` changes the output folder. A circuit other than the default writes to `results/<circuit>/`.

The full run uses 50,000 lap evaluations per run, 10 seeds per algorithm variant and 5 seeds per parameter sweep cell. Seeds are fixed, so repeated runs give identical numbers. Settings live in `racing_line/config.py`.

## Outputs (in results/)

| File | Content |
|---|---|
| `summary.csv` | mean, std, best and worst lap time per method, gain over the minimum curvature line |
| `convergence.png` | best lap time against evaluations, mean and standard deviation over seeds |
| `variations.png` | box plots for crossover types, DE strategies and PSO topologies |
| `sweeps.png` | parameter sweeps: GA population and mutation rate, DE F and CR, PSO inertia and population |
| `track_lines.png` | all lines on the circuit, and the best line coloured by speed |
| `speed_profile.png` | speed along the lap for the minimum curvature line and the best evolved line |
| `best_line.npy` | decision vector of the best evolved line |
| `circuits.csv` | (from `--circuits`) mean, std and gain over the minimum curvature line for each algorithm on each circuit |
| `circuits.png` | (from `--circuits`) the best evolved line on each circuit and the gain as a percentage of lap time |

`reference_results/` holds the output of one full run. After your own full run, `results/summary.csv` should match `reference_results/summary.csv` exactly, because the seeds are fixed. The evolved lap times match exactly. The minimum curvature reference line does not: it sits in a nearly flat valley, so tiny numerical differences between machines move its lap time by a few hundredths of a second (up to about 0.05 s on the Technical circuit and 0.07 s on the Flowing one). Compare the evolved results, and quote the reference lap time from your own run.

The console output also prints a Mann-Whitney U test between the three headline algorithms.

## How it works

**Representation.** A solution is 50 numbers in [-1, 1]. They are sideways offsets at evenly spaced points along the track, joined by a periodic cubic spline and scaled to the usable half width (track half width minus half the car width). Every candidate therefore stays on the track and is smooth.

**Fitness.** The lap time model treats the car as a point mass. Corner speed is limited by lateral grip, `v = sqrt(A_LAT / curvature)`. Acceleration is limited by traction and engine power with drag, braking by a fixed deceleration, and both share grip with cornering through a friction circle. A forward pass applies acceleration limits, a backward pass applies braking limits, and both wrap around the lap twice so the start line carries no bias.

**Baselines.** The centreline, the shortest path and the minimum curvature line (minimising the integral of curvature squared, solved with L-BFGS-B). Random search with the same evaluation budget is the control for the evolutionary methods.

## Limitations

- The car model is deliberately simple: constant grip, no tyre wear, no downforce curve, no weight transfer. Lap times are comparable between methods but are not real F1 lap times.
- The circuits are fictional and defined in `racing_line/track.py`. None is a real F1 circuit. On the flowing circuit the car spends much of the lap at its top speed, so the line matters less there and gains are smaller.
- The reported gain of the evolved lines over the minimum curvature line is under one second on a lap of about 62 seconds. Parameter sweep cells use 5 seeds, so small differences between cells should not be over interpreted.
- The minimum curvature reference lap time depends slightly on the machine (see above). The gains of the evolved lines over it are larger than this variation, but quote the reference and gains from one run on one machine.
- Evolutionary methods optimise the lap time directly, while the minimum curvature line optimises a proxy. Part of the gap is therefore expected by construction.

## Use of AI tools

Claude (Anthropic, a large language model) assisted with the graphs and with the mechanics and physics formulas used to calculate the speed limits of the car, such as braking, acceleration and cornering.