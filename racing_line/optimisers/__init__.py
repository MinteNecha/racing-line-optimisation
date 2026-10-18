from racing_line.optimisers.base import Result
from racing_line.optimisers.differential import de
from racing_line.optimisers.genetic import ga
from racing_line.optimisers.random_search import random_search
from racing_line.optimisers.swarm import pso

OPTIMISERS = {"ga": ga, "de": de, "pso": pso, "random": random_search}

__all__ = ["OPTIMISERS", "Result", "de", "ga", "pso", "random_search"]
