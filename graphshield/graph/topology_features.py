"""
Topological Feature Extraction for Dependency Graphs.
Computes In/Out-Degree, PageRank, Betweenness, k-Core, Downstream Blast Radius, and Unpinned Ratio.
"""

from typing import Any, Dict, List
import networkx as nx
import numpy as np


def compute_downstream_blast_radius(g: nx.DiGraph, node: str) -> int:
    """
    Compute the downstream blast radius: the number of packages that depend
    directly or transitively on `node`.
    In our convention, edge (u, v) denotes 'u depends on v'.
    Therefore, packages depending on `node` are ancestors of `node` in g.
    """
    if node not in g:
        return 0
    try:
        ancestors = nx.ancestors(g, node)
        return len(ancestors)
    except Exception:
        return 0


def compute_transitive_depth(g: nx.DiGraph, node: str) -> int:
    """Compute the maximum dependency depth reachable from `node`."""
    if node not in g:
        return 0
    try:
        lengths = nx.single_source_shortest_path_length(g, node)
        return max(lengths.values()) if lengths else 0
    except Exception:
        return 0


def extract_topological_features(g: nx.DiGraph, all_nodes: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Extract comprehensive topological graph metrics for all nodes in graph `g`.
    Returns dict mapping node_name -> dict of metric values.
    """
    n_nodes = len(all_nodes)
    if n_nodes == 0:
        return {}

    # Degrees
    in_degrees = dict(g.in_degree()) if len(g) > 0 else {}
    out_degrees = dict(g.out_degree()) if len(g) > 0 else {}

    # Reverse graph where edge (v, u) denotes 'infection flow from v to u'
    g_rev = g.reverse() if len(g) > 0 else nx.DiGraph()

    # PageRank on reverse graph reflects downstream centrality
    try:
        if len(g_rev) > 0 and g_rev.number_of_edges() > 0:
            pagerank = nx.pagerank(g_rev, alpha=0.85, max_iter=200, tol=1e-5)
        else:
            pagerank = {n: 1.0 / n_nodes for n in all_nodes}
    except Exception:
        pagerank = {n: 1.0 / n_nodes for n in all_nodes}

    # Betweenness centrality
    try:
        if len(g) > 0 and g.number_of_edges() > 0:
            betweenness = nx.betweenness_centrality(g, normalized=True)
        else:
            betweenness = {n: 0.0 for n in all_nodes}
    except Exception:
        betweenness = {n: 0.0 for n in all_nodes}

    # k-Core decomposition (on undirected copy)
    try:
        if len(g) > 0 and g.number_of_edges() > 0:
            undirected_g = g.to_undirected()
            # remove self-loops if any
            undirected_g.remove_edges_from(nx.selfloop_edges(undirected_g))
            k_core = nx.core_number(undirected_g)
        else:
            k_core = {n: 0 for n in all_nodes}
    except Exception:
        k_core = {n: 0 for n in all_nodes}

    # Assemble per-node dictionary
    results: Dict[str, Dict[str, float]] = {}
    for node in all_nodes:
        in_d = float(in_degrees.get(node, 0))
        out_d = float(out_degrees.get(node, 0))
        pr = float(pagerank.get(node, 0.0))
        bet = float(betweenness.get(node, 0.0))
        core = float(k_core.get(node, 0))
        blast = float(compute_downstream_blast_radius(g, node))
        depth = float(compute_transitive_depth(g, node))

        # Compute unpinned constraint ratio for outgoing dependencies
        unpinned = 0
        total_out = 0
        if node in g:
            for _, _, data in g.out_edges(node, data=True):
                ctype = data.get("constraint", "caret")
                total_out += 1
                if ctype in ("caret", "tilde", "wildcard"):
                    unpinned += 1
        unpinned_ratio = float(unpinned / total_out) if total_out > 0 else 0.0

        results[node] = {
            "in_degree": in_d,
            "out_degree": out_d,
            "pagerank": pr,
            "betweenness": bet,
            "k_core": core,
            "blast_radius": blast,
            "transitive_depth": depth,
            "unpinned_ratio": unpinned_ratio,
        }

    return results
