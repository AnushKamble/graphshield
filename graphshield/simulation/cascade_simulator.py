"""
Monte Carlo Cascade Simulator for Vulnerability Propagation in Dependency Networks.
Simulates epidemic propagation (Susceptible - Infected - Recovered) across directed DAG edges,
parameterized by semantic versioning constraint types (pinned vs caret vs wildcard).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np


# Infection transmission probabilities by constraint type
CONSTRAINT_TRANSMISSION_PROBS = {
    "pinned": 0.05,     # Exact pin ==x.y.z resists automated transitive injection
    "tilde": 0.35,      # ~x.y accepts patch bumps
    "caret": 0.75,      # ^x.y accepts minor & patch bumps automatically
    "wildcard": 0.95,   # * pulls latest indiscriminately
}


@dataclass
class SimulationResult:
    iterations: int
    seed: int
    infected_counts_per_step: List[float]
    recovered_counts_per_step: List[float]
    susceptible_counts_per_step: List[float]
    package_infection_probabilities: Dict[str, float]
    mean_cascade_size: float
    max_cascade_size: int
    root_causes: List[str]


class CascadeSimulator:
    """
    Monte Carlo stochastic cascade simulator across dynamic dependency graphs.
    Simulates how an upstream vulnerability or malicious backdoor cascades
    downstream to dependent consumer packages.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        recovery_prob: float = 0.15,
        default_transmission_prob: float = 0.5,
    ):
        self.graph = graph
        # In our graph, edge (u, v) means u depends on v.
        # Vulnerability cascades from upstream dependency v to downstream dependent u.
        # So transmission follows reverse edges: v -> u.
        self.infection_graph = graph.reverse()
        self.recovery_prob = recovery_prob
        self.default_transmission_prob = default_transmission_prob

    def run_simulation(
        self,
        seed_infected_packages: List[str],
        num_steps: int = 10,
        num_mc_runs: int = 100,
        random_seed: int = 42,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulations of the cascade.
        """
        if num_mc_runs <= 0:
            raise ValueError(f"num_mc_runs must be at least 1, got {num_mc_runs}")
        if num_steps < 0:
            raise ValueError(f"num_steps cannot be negative, got {num_steps}")

        rng = np.random.RandomState(random_seed)
        all_nodes = list(self.graph.nodes())
        total_nodes = len(all_nodes)

        infection_frequencies: Dict[str, int] = {node: 0 for node in all_nodes}
        cascade_sizes = []
        step_infected_histories = np.zeros((num_mc_runs, num_steps + 1))
        step_recovered_histories = np.zeros((num_mc_runs, num_steps + 1))
        step_susceptible_histories = np.zeros((num_mc_runs, num_steps + 1))

        for run in range(num_mc_runs):
            # States: 'S' (Susceptible), 'I' (Infected), 'R' (Recovered/Patched)
            state: Dict[str, str] = {node: "S" for node in all_nodes}

            # Initialize seed infections
            for root in seed_infected_packages:
                if root in state:
                    state[root] = "I"

            step_infected_histories[run, 0] = sum(1 for s in state.values() if s == "I")
            step_recovered_histories[run, 0] = 0
            step_susceptible_histories[run, 0] = total_nodes - step_infected_histories[run, 0]

            for step in range(1, num_steps + 1):
                new_state = state.copy()
                currently_infected = [node for node, s in state.items() if s == "I"]

                # 1. Propagation: infected nodes attempt to infect their downstream dependents
                for inf_node in currently_infected:
                    # Look at outgoing edges in infection_graph (dependents of inf_node)
                    if inf_node in self.infection_graph:
                        for dependent in self.infection_graph.neighbors(inf_node):
                            if state[dependent] == "S":
                                # Edge attribute in original graph was (dependent -> inf_node)
                                edge_data = self.graph.get_edge_data(dependent, inf_node, default={})
                                ctype = edge_data.get("constraint", "caret").lower()
                                trans_prob = CONSTRAINT_TRANSMISSION_PROBS.get(
                                    ctype, self.default_transmission_prob
                                )

                                if rng.rand() < trans_prob:
                                    new_state[dependent] = "I"

                # 2. Recovery / Patch adoption: infected nodes may upgrade or patch
                for inf_node in currently_infected:
                    if rng.rand() < self.recovery_prob:
                        new_state[inf_node] = "R"

                state = new_state
                n_i = sum(1 for s in state.values() if s == "I")
                n_r = sum(1 for s in state.values() if s == "R")
                n_s = sum(1 for s in state.values() if s == "S")

                step_infected_histories[run, step] = n_i
                step_recovered_histories[run, step] = n_r
                step_susceptible_histories[run, step] = n_s

            # Track total nodes that were ever infected in this run (I or R)
            ever_infected = [node for node, s in state.items() if s in ("I", "R")]
            cascade_sizes.append(len(ever_infected))
            for node in ever_infected:
                infection_frequencies[node] += 1

        package_probs = {
            node: round(count / num_mc_runs, 4)
            for node, count in infection_frequencies.items()
        }

        return SimulationResult(
            iterations=num_mc_runs,
            seed=random_seed,
            infected_counts_per_step=np.mean(step_infected_histories, axis=0).round(2).tolist(),
            recovered_counts_per_step=np.mean(step_recovered_histories, axis=0).round(2).tolist(),
            susceptible_counts_per_step=np.mean(step_susceptible_histories, axis=0).round(2).tolist(),
            package_infection_probabilities=package_probs,
            mean_cascade_size=round(float(np.mean(cascade_sizes)), 2),
            max_cascade_size=int(np.max(cascade_sizes)),
            root_causes=seed_infected_packages,
        )
