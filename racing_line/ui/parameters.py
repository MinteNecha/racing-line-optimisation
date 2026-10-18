"""Tunable settings of each algorithm with plain English explanations for the dashboard."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    default: object
    help: str
    low: float = None
    high: float = None
    step: float = None
    options: dict = None  # value -> label, for choices


POP_HELP = ("How many candidate racing lines exist at once. Think of the size of a team of drivers "
            "trying different lines. More drivers explore more, but each generation costs more laps.")

PARAMS = {
    "ga": [
        Param("pop", "Population size", 60, POP_HELP, 20, 160, 10),
        Param("pc", "Crossover probability", 0.9,
              "Chance that two parents are blended into a child. Otherwise the child is a copy of the first parent.",
              0.0, 1.0, 0.05),
        Param("pm", "Mutation rate per gene", 0.05,
              "Chance that each of the 50 offsets in a child is nudged at random. It keeps new ideas arriving, "
              "like a driver trying a slightly different braking point.", 0.0, 0.3, 0.01),
        Param("sigma", "Mutation size", 0.1,
              "How big that nudge is. Small nudges fine tune a line, large ones make bold changes. "
              "The full width of the track is 2 in these units.", 0.01, 0.5, 0.01),
        Param("elite", "Elites kept", 2,
              "Number of best lines copied unchanged into the next generation, so the best result so far is never lost.",
              0, 10, 1),
        Param("k", "Tournament size", 3,
              "How many random lines compete to become a parent. The fastest one wins. Bigger groups mean stricter selection.",
              2, 8, 1),
        Param("cross", "Crossover type", "blx", "How the two parents are mixed to make a child.",
              options={"blx": "BLX-0.5 (blend with overshoot)", "arith": "Arithmetic (weighted average)",
                       "uniform": "Uniform (each offset from one parent)"}),
    ],
    "de": [
        Param("pop", "Population size", 60, POP_HELP, 20, 160, 10),
        Param("F", "Scale factor F", 0.6,
              "Step size. A trial line is made by adding F times the difference between two random lines to a third. "
              "Small F takes careful steps, large F jumps far.", 0.1, 1.0, 0.05),
        Param("CR", "Crossover rate CR", 0.9,
              "Chance that each offset of the trial line comes from the new mix instead of the old line. "
              "High values change many offsets at once.", 0.0, 1.0, 0.05),
        Param("strategy", "Mutation strategy", "rand1",
              "Which lines the difference is added to. Random lines explore, the best line follows the leader.",
              options={"rand1": "rand/1 (explores)", "best1": "best/1 (follows the leader)",
                       "ctb1": "current-to-best/1 (a mix)"}),
    ],
    "pso": [
        Param("pop", "Swarm size", 60, "How many particles fly around. Each particle is one racing line.", 20, 160, 10),
        Param("w", "Inertia weight w", 0.72,
              "Momentum. How much each particle keeps its previous direction. High values keep exploring, "
              "low values settle quickly.", 0.2, 1.0, 0.02),
        Param("c1", "Personal pull c1", 1.49,
              "How strongly a particle is drawn back to the best line it has found itself.", 0.0, 3.0, 0.05),
        Param("c2", "Group pull c2", 1.49,
              "How strongly a particle is drawn to the best line found by its neighbours.", 0.0, 3.0, 0.05),
        Param("vmax", "Speed limit", 0.5,
              "Largest move per step in offset units, where the whole track width is 2.", 0.05, 1.0, 0.05),
        Param("topology", "Neighbourhood", "gbest", "Who each particle listens to.",
              options={"gbest": "Global best (everyone sees the leader)", "lbest": "Ring (two neighbours each)"}),
    ],
    "random": [
        Param("pop", "Batch size", 60, "How many random lines are tried in each batch.", 20, 160, 10),
    ],
}

BLURBS = {
    "ga": "Evolves a population of racing lines. Faster lines are more likely to become parents, parents are mixed "
          "to make children, and random mutation adds variety. The best lines always survive.",
    "de": "Builds each new trial line by adding the difference between two random lines to a third. "
          "The trial replaces its parent only if it gives a faster lap.",
    "pso": "A swarm of particles flies through the space of racing lines. Each is pulled towards the best line it has "
           "found and the best line its neighbours found, while keeping some of its earlier momentum.",
    "random": "Draws random lines and keeps the best. It learns nothing, so it is the control the other "
              "algorithms must beat.",
}


def defaults(algo):
    return {p.name: p.default for p in PARAMS[algo]}


def describe(algo, values):
    """Rows of (setting, value, explanation) for the settings an algorithm was run with."""
    rows = []
    for p in PARAMS[algo]:
        value = values[p.name]
        rows.append({"Setting": p.label, "Value": str(p.options[value] if p.options else value),
                     "What it does": p.help})
    return rows
