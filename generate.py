"""Data generation script for graph trajectory training data.

Generates Erdos-Renyi random graphs, samples trajectories as node sequences,
and produces train/val/test splits with a vocabulary file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import hydra
from omegaconf import DictConfig

SPECIAL_TOKENS = ["<pad>", "<eos>", "<start_goal>", "<end_goal>", "<unk>"]


def generate_graph(num_nodes: int, edge_probability: float, seed: int) -> nx.Graph:
    """Generate an Erdos-Renyi graph and prune to the largest connected component."""
    graph = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)

    # Keep only the largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()

    # Relabel nodes to consecutive integers starting from 0
    mapping = {old: new for new, old in enumerate(sorted(graph.nodes()))}
    graph = nx.relabel_nodes(graph, mapping)

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
          f"(pruned from {num_nodes} nodes)")
    return graph


def save_graph(graph: nx.Graph, output_dir: str) -> None:
    """Save graph structure as JSON with nodes and adjacency lists."""
    nodes = sorted(graph.nodes())
    adjacency = {str(node): sorted(list(graph.neighbors(node))) for node in nodes}

    graph_data = {
        "nodes": nodes,
        "adjacency": adjacency,
    }

    path = Path(output_dir) / "graph.json"
    with open(path, "w") as f:
        json.dump(graph_data, f, indent=2)
    print(f"Saved graph to {path}")


def format_trajectory(path: list[int]) -> str:
    """Format a node path as a trajectory string.

    Format: <start_goal> START END <end_goal> START A B C ... END
    """
    start = path[0]
    end = path[-1]
    path_str = " ".join(str(n) for n in path)
    return f"<start_goal> {start} {end} <end_goal> {path_str}"


def find_random_path(
    graph: nx.Graph,
    start: int,
    end: int,
    max_length: int,
    rng: random.Random,
) -> list[int] | None:
    """Find a path from start to end using a biased random walk.

    Uses a random walk biased toward the target node. Falls back to BFS
    shortest path if the random walk fails to reach the target.
    """
    # First check if a path exists and is within length limit
    try:
        shortest = nx.shortest_path(graph, start, end)
    except nx.NetworkXNoPath:
        return None

    if len(shortest) > max_length:
        return None

    # For short paths or with some probability, just return the shortest path
    if len(shortest) >= max_length - 1 or rng.random() < 0.3:
        return shortest

    # Try a random walk with detours for path diversity
    path = [start]
    current = start
    visited_set = {start}
    steps = 0

    while current != end and steps < max_length - 1:
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break

        # Bias toward unvisited nodes and toward the target
        weights = []
        for n in neighbors:
            w = 1.0
            if n == end:
                w = 5.0  # Strong bias toward target
            elif n not in visited_set:
                w = 2.0  # Prefer unvisited nodes
            weights.append(w)

        next_node = rng.choices(neighbors, weights=weights, k=1)[0]
        path.append(next_node)
        visited_set.add(next_node)
        current = next_node
        steps += 1

    # If we didn't reach the target, fall back to shortest path
    if path[-1] != end:
        return shortest

    return path


def generate_trajectories(
    graph: nx.Graph,
    num_trajectories: int,
    max_path_length: int,
    seed: int,
) -> list[str]:
    """Generate trajectory strings ensuring all nodes are covered at least once."""
    rng = random.Random(seed)
    nodes = sorted(graph.nodes())
    trajectories = []
    uncovered = set(nodes)

    # Phase 1: coverage pass — ensure every node appears in at least one trajectory
    attempts = 0
    max_attempts = len(nodes) * 10
    while uncovered and attempts < max_attempts:
        start = rng.choice(list(uncovered))
        end = rng.choice(nodes)
        while end == start:
            end = rng.choice(nodes)

        path = find_random_path(graph, start, end, max_path_length, rng)
        if path is not None:
            trajectories.append(format_trajectory(path))
            uncovered -= set(path)
        attempts += 1

    if uncovered:
        print(f"Warning: {len(uncovered)} nodes could not be covered in trajectories")

    print(f"Coverage phase: {len(trajectories)} trajectories "
          f"(covered {len(nodes) - len(uncovered)}/{len(nodes)} nodes)")

    # Phase 2: fill remaining trajectories with random paths
    while len(trajectories) < num_trajectories:
        start = rng.choice(nodes)
        end = rng.choice(nodes)
        while end == start:
            end = rng.choice(nodes)

        path = find_random_path(graph, start, end, max_path_length, rng)
        if path is not None:
            trajectories.append(format_trajectory(path))

    return trajectories


def save_vocabulary(graph: nx.Graph, output_dir: str) -> None:
    """Save vocabulary.json with node IDs and special tokens."""
    node_tokens = [str(n) for n in sorted(graph.nodes())]
    all_tokens = SPECIAL_TOKENS + node_tokens

    token_to_id = {token: idx for idx, token in enumerate(all_tokens)}

    vocab_data = {
        "tokens": all_tokens,
        "token_to_id": token_to_id,
    }

    path = Path(output_dir) / "vocabulary.json"
    with open(path, "w") as f:
        json.dump(vocab_data, f, indent=2)
    print(f"Saved vocabulary ({len(all_tokens)} tokens) to {path}")


def split_and_save(
    trajectories: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    output_dir: str,
    seed: int,
) -> None:
    """Shuffle and split trajectories into train/val/test files."""
    rng = random.Random(seed)
    rng.shuffle(trajectories)

    n = len(trajectories)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train.txt": trajectories[:train_end],
        "val.txt": trajectories[train_end:val_end],
        "test.txt": trajectories[val_end:],
    }

    output_path = Path(output_dir)
    for filename, data in splits.items():
        filepath = output_path / filename
        with open(filepath, "w") as f:
            for line in data:
                f.write(line + "\n")
        print(f"Saved {len(data)} trajectories to {filepath}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main data generation entrypoint."""
    output_dir = cfg.data.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate and save graph
    graph = generate_graph(
        num_nodes=cfg.data.num_nodes,
        edge_probability=cfg.data.edge_probability,
        seed=cfg.data.seed,
    )
    save_graph(graph, output_dir)

    # Generate trajectories
    trajectories = generate_trajectories(
        graph=graph,
        num_trajectories=cfg.data.num_trajectories,
        max_path_length=cfg.data.max_path_length,
        seed=cfg.data.seed,
    )

    # Save vocabulary
    save_vocabulary(graph, output_dir)

    # Split and save
    split_and_save(
        trajectories=trajectories,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        output_dir=output_dir,
        seed=cfg.data.seed,
    )

    print(f"\nData generation complete. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
