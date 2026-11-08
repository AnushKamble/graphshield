"""Unit tests for Monte Carlo Cascade Simulator."""

import networkx as nx
import pytest

from graphshield.simulation.cascade_simulator import CascadeSimulator, SimulationResult


def test_cascade_simulator_propagation():
    # Construct a dependency chain:
    # app -> web -> core (app depends on web, web depends on core)
    g = nx.DiGraph()
    g.add_edge("app", "web", constraint="caret")
    g.add_edge("web", "core", constraint="caret")

    sim = CascadeSimulator(graph=g, recovery_prob=0.0, default_transmission_prob=0.8)
    res = sim.run_simulation(
        seed_infected_packages=["core"],
        num_steps=5,
        num_mc_runs=50,
        random_seed=42,
    )

    assert isinstance(res, SimulationResult)
    assert res.iterations == 50
    assert "core" in res.package_infection_probabilities
    assert res.package_infection_probabilities["core"] == 1.0  # Seed is always infected
    # Downstream dependents should have high probability of infection
    assert res.package_infection_probabilities["web"] > 0.5
    assert res.package_infection_probabilities["app"] > 0.3
    assert res.mean_cascade_size >= 1.5


def test_simulation_reproducibility():
    g = nx.DiGraph()
    g.add_edge("pkgB", "pkgA", constraint="caret")
    sim = CascadeSimulator(graph=g)

    res1 = sim.run_simulation(["pkgA"], num_steps=3, num_mc_runs=20, random_seed=999)
    res2 = sim.run_simulation(["pkgA"], num_steps=3, num_mc_runs=20, random_seed=999)

    assert res1.mean_cascade_size == res2.mean_cascade_size
    assert res1.package_infection_probabilities == res2.package_infection_probabilities


def test_pinned_constraint_resistance():
    # Pinned dependency should have lower transmission than caret
    g_pinned = nx.DiGraph()
    g_pinned.add_edge("downstream", "upstream", constraint="pinned")
    sim_pinned = CascadeSimulator(graph=g_pinned, recovery_prob=0.0)
    res_pinned = sim_pinned.run_simulation(["upstream"], num_steps=3, num_mc_runs=100, random_seed=123)

    g_caret = nx.DiGraph()
    g_caret.add_edge("downstream", "upstream", constraint="caret")
    sim_caret = CascadeSimulator(graph=g_caret, recovery_prob=0.0)
    res_caret = sim_caret.run_simulation(["upstream"], num_steps=3, num_mc_runs=100, random_seed=123)

    assert res_pinned.package_infection_probabilities["downstream"] < res_caret.package_infection_probabilities["downstream"]


def test_cascade_simulator_invalid_inputs():
    """Verify input validation for Monte Carlo parameters."""
    g = nx.DiGraph()
    g.add_edge("B", "A", constraint="caret")
    sim = CascadeSimulator(graph=g)

    with pytest.raises(ValueError, match="num_mc_runs must be at least 1"):
        sim.run_simulation(["A"], num_mc_runs=0)

    with pytest.raises(ValueError, match="num_steps cannot be negative"):
        sim.run_simulation(["A"], num_steps=-1)
