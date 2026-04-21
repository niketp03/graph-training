"""
eval_length_generalization.py

Evaluates length generalisation for pretrain-only and posttrain models.

Two modes:

  1. Single-pair mode (original): pass --pretrain_checkpoint and
     --posttrain_checkpoint directly.

  2. Directory mode: pass --pretrain_dir and --posttrain_dir.
     The script discovers all run_*/ subdirectories, reads metadata.json /
     the checkpoint's embedded metadata, and produces comparison plots for
     pairs matched by:
       (a) same max data length  (pretrain max_path_length == posttrain n_max)
       (b) same total data seen  (pretrain samples_seen ≈ posttrain total_samples_seen)

Usage examples:
    # Single-pair mode
    python eval_length_generalization.py \\
        --pretrain_checkpoint checkpoints/run_20260419_120000/best.ckpt \\
        --posttrain_checkpoint postrain_checkpoints/run_20260419_140000/postrain_final.pt \\
        --tokenizer_dir data_output/tokenizer \\
        --graph_path data_output/graph.json

    # Directory mode
    python eval_length_generalization.py \\
        --pretrain_dir checkpoints \\
        --posttrain_dir postrain_checkpoints \\
        --tokenizer_dir data_output/tokenizer \\
        --graph_path data_output/graph.json
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from model import GraphTrajectoryLM
from tokenizer_utils import load_tokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Length generalisation evaluation")

    # --- checkpoint inputs (single-pair mode) ---
    p.add_argument("--pretrain_checkpoint", default=None,
                   help="Path to pretrain-only checkpoint (.ckpt)")
    p.add_argument("--posttrain_checkpoint", default=None,
                   help="Path to posttrain checkpoint (.pt)")

    # --- checkpoint inputs (directory mode) ---
    p.add_argument("--pretrain_dir", default=None,
                   help="Base directory of pretrain runs (contains run_*/ subdirs)")
    p.add_argument("--posttrain_dir", default=None,
                   help="Base directory of posttrain runs (contains run_*/ subdirs)")

    # --- shared required args ---
    p.add_argument("--tokenizer_dir", required=True,
                   help="Directory containing the HuggingFace tokenizer")
    p.add_argument("--graph_path", required=True, help="Path to graph.json")

    # --- config (used only in single-pair mode when no run-level config.yaml exists) ---
    p.add_argument("--config_path", default="conf/config.yaml",
                   help="Fallback Hydra config YAML")

    p.add_argument("--output_dir", default=".",
                   help="Directory where plots are saved")
    p.add_argument("--M", type=int, default=50,
                   help="(u,v) pairs sampled per distance")
    p.add_argument("--num_attempts", type=int, default=1,
                   help="Generation attempts per pair")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--max_dist", type=int, default=30,
                   help="Evaluate distances up to this value")
    p.add_argument("--sample_sources", type=int, default=500,
                   help="Number of source nodes for BFS")
    p.add_argument("--train_max", type=int, default=None,
                   help="Override pretrain max path length for plot divider")
    p.add_argument("--posttrain_max", type=int, default=None,
                   help="Override posttrain max path length for plot divider")
    p.add_argument("--total_data_tolerance", type=float, default=0.15,
                   help="Fractional tolerance for matching by total data (default 0.15)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="Device override, e.g. 'cpu' or 'cuda:1'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def discover_pretrain_runs(base_dir: Path) -> list[dict]:
    """Scan base_dir/run_*/ for pretrain checkpoints and their metadata."""
    runs = []
    for run_dir in sorted(base_dir.glob("run_*")):
        ckpt = None
        for fname in ["best.ckpt", "last.ckpt"]:
            p = run_dir / fname
            if p.exists():
                ckpt = p
                break
        if ckpt is None:
            continue

        meta = _load_run_metadata(run_dir, ckpt, kind="pretrain")
        if meta is not None:
            runs.append(meta)
    return runs


def discover_posttrain_runs(base_dir: Path) -> list[dict]:
    """Scan base_dir/run_*/ for posttrain checkpoints and their metadata."""
    runs = []
    for run_dir in sorted(base_dir.glob("run_*")):
        ckpt = None
        for fname in ["postrain_final.pt"]:
            p = run_dir / fname
            if p.exists():
                ckpt = p
                break
        if ckpt is None:
            continue

        meta = _load_run_metadata(run_dir, ckpt, kind="posttrain")
        if meta is not None:
            runs.append(meta)
    return runs


def _load_run_metadata(run_dir: Path, ckpt_path: Path, kind: str) -> dict | None:
    """Return a metadata dict for a run, preferring metadata.json over checkpoint contents."""
    meta_path = run_dir / "metadata.json"
    config_path = run_dir / "config.yaml"

    meta: dict = {}

    # Try metadata.json first
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        # Fall back to peeking inside the checkpoint
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if kind == "pretrain":
                meta["samples_seen"] = raw.get("samples_seen", 0)
                meta["max_path_length"] = raw.get("max_path_length")
            else:
                meta["n_max"] = raw.get("n_max")
                meta["max_path_length"] = raw.get("max_path_length")
                meta["pretrain_samples_seen"] = raw.get("pretrain_samples_seen", 0)
                meta["posttrain_samples_seen"] = raw.get("posttrain_samples_seen", 0)
                meta["total_samples_seen"] = raw.get("total_samples_seen", 0)
            del raw
        except Exception as e:
            print(f"  WARNING: could not read metadata from {ckpt_path}: {e}")
            return None

    meta["checkpoint_path"] = str(ckpt_path)
    meta["run_dir"] = str(run_dir)
    if config_path.exists():
        meta["config_path"] = str(config_path)

    return meta


# ---------------------------------------------------------------------------
# Checkpoint matching
# ---------------------------------------------------------------------------


def match_by_max_length(pretrain_runs: list[dict],
                        posttrain_runs: list[dict]) -> list[tuple[dict, dict]]:
    """Pair runs where pretrain max_path_length == posttrain n_max."""
    pairs = []
    for pre in pretrain_runs:
        pre_len = pre.get("max_path_length")
        if pre_len is None:
            continue
        for post in posttrain_runs:
            post_len = post.get("n_max")
            if post_len is not None and pre_len == post_len:
                pairs.append((pre, post))
    return pairs


def match_by_total_data(pretrain_runs: list[dict],
                        posttrain_runs: list[dict],
                        tolerance: float = 0.15) -> list[tuple[dict, dict]]:
    """Pair runs where total samples seen are within *tolerance* of each other.

    pretrain total  = samples_seen
    posttrain total = pretrain_samples_seen + posttrain_samples_seen
    """
    pairs = []
    for pre in pretrain_runs:
        pre_total = pre.get("samples_seen", 0)
        if pre_total <= 0:
            continue
        for post in posttrain_runs:
            post_total = post.get("total_samples_seen",
                                  post.get("pretrain_samples_seen", 0)
                                  + post.get("posttrain_samples_seen", 0))
            if post_total <= 0:
                continue
            if abs(pre_total - post_total) / pre_total <= tolerance:
                pairs.append((pre, post))
    return pairs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_config(run_meta: dict, fallback_config_path: str):
    cfg_path = run_meta.get("config_path", fallback_config_path)
    return OmegaConf.load(cfg_path)


def load_pretrain_model(checkpoint_path: str, tokenizer, cfg, device):
    model = GraphTrajectoryLM(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        model_config=cfg.model,
        train_config=cfg.train,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model_state_dict" in ckpt:
        model.model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    del ckpt
    model.eval()
    model.to(device)
    return model


def load_posttrain_model(checkpoint_path: str, tokenizer, cfg, device):
    # posttrain checkpoints store model_state_dict under the HF model
    model = GraphTrajectoryLM(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        model_config=cfg.model,
        train_config=cfg.train,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model_state_dict" in ckpt:
        model.model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    del ckpt
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def make_generate_fn(model, tokenizer, device, temperature, top_k, max_new_tokens):
    @torch.no_grad()
    def generate_trajectory(start_node: int, goal_node: int) -> list:
        prompt = f"<start_goal> {start_node} {goal_node} <end_goal>"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        generated = input_ids
        for _ in range(max_new_tokens):
            logits = model.model(input_ids=generated).logits[:, -1, :] / temperature
            logits[:, pad_id] = -float("inf")

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, -1:]] = -float("inf")

            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == eos_id:
                break

        path_tokens = tokenizer.decode(
            generated[0, input_ids.shape[1]:], skip_special_tokens=True
        ).split()
        return path_tokens

    return generate_trajectory


def is_valid_and_reaches_goal(G, start: int, goal: int, path_tokens: list[str]) -> bool:
    try:
        path_tokens_temp = []
        for t in path_tokens:
            if t == "<eos>":
                break
            path_tokens_temp.append(int(t))
            if int(t) == goal:
                break
        path_nodes = path_tokens_temp
    except ValueError:
        return False
    if not path_nodes or path_nodes[0] != start or path_nodes[-1] != goal:
        return False
    return all(
        G.has_edge(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def run_eval(generate_fn, G, pairs_by_dist, M, num_attempts, max_dist):
    results = {}
    for dist in tqdm(sorted(pairs_by_dist.keys()), desc="Evaluating distances"):
        if dist > max_dist:
            break

        pool = pairs_by_dist[dist]
        selected = random.sample(pool, min(M, len(pool)))
        successes = 0

        for u, v in tqdm(selected, desc=f"  d={dist}", leave=False):
            solved = False
            for _ in range(num_attempts):
                path_tokens = generate_fn(u, v)
                if is_valid_and_reaches_goal(G, u, v, path_tokens):
                    solved = True
                    break
            if solved:
                successes += 1

        results[dist] = {"successes": successes, "total": len(selected)}
        acc = successes / len(selected) * 100
        print(f"  d={dist:>2d}:  {successes}/{len(selected)}  ({acc:.1f}%)")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results, train_max, posttrain_max, num_attempts, title, save_path):
    distances = sorted(results.keys())
    accuracies = [
        results[d]["successes"] / results[d]["total"] * 100 for d in distances
    ]
    counts = [results[d]["total"] for d in distances]

    if posttrain_max is not None and posttrain_max != train_max:
        colors = [
            (
                "#4c72b0"
                if d <= train_max
                else "#c44e52" if d <= posttrain_max else "green"
            )
            for d in distances
        ]
    else:
        colors = ["#4c72b0" if d <= train_max else "#c44e52" for d in distances]

    fig, ax = plt.subplots(figsize=(max(10, len(distances) * 0.7), 5))
    bars = ax.bar(distances, accuracies, color=colors, edgecolor="white", linewidth=0.5)

    for bar, acc, n in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, -4,
                f"n={n}", ha="center", va="top", fontsize=7, color="gray")

    ax.axvline(x=train_max + 0.5, color="black", linestyle="--", linewidth=1,
               label=f"Pretraining max path length = {train_max}")
    if posttrain_max is not None and posttrain_max != train_max:
        ax.axvline(x=posttrain_max + 0.5, color="gray", linestyle="--", linewidth=1,
                   label=f"Posttraining max path length = {posttrain_max}")

    ax.set_xlabel("Shortest-path distance between (u, v)", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(distances)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def plot_comparison(pretrain_results, posttrain_results,
                    train_max, posttrain_max,
                    num_attempts, save_path, subtitle=""):
    """Side-by-side bar charts for a matched (pretrain, posttrain) pair."""
    fig, axes = plt.subplots(1, 2, figsize=(max(20, len(pretrain_results) * 1.4), 5),
                             sharey=True)

    for ax, results, title_suffix, pt_max in [
        (axes[0], pretrain_results, "Pretrain only", None),
        (axes[1], posttrain_results, "After posttraining", posttrain_max),
    ]:
        distances = sorted(results.keys())
        accuracies = [results[d]["successes"] / results[d]["total"] * 100 for d in distances]
        counts = [results[d]["total"] for d in distances]

        if pt_max is not None and pt_max != train_max:
            colors = ["#4c72b0" if d <= train_max else "#c44e52" if d <= pt_max else "green"
                      for d in distances]
        else:
            colors = ["#4c72b0" if d <= train_max else "#c44e52" for d in distances]

        bars = ax.bar(distances, accuracies, color=colors, edgecolor="white", linewidth=0.5)
        for bar, acc, n in zip(bars, accuracies, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
            ax.text(bar.get_x() + bar.get_width() / 2, -4,
                    f"n={n}", ha="center", va="top", fontsize=6, color="gray")

        ax.axvline(x=train_max + 0.5, color="black", linestyle="--", linewidth=1,
                   label=f"Pretrain max = {train_max}")
        if pt_max is not None and pt_max != train_max:
            ax.axvline(x=pt_max + 0.5, color="gray", linestyle="--", linewidth=1,
                       label=f"Posttrain max = {pt_max}")

        ax.set_title(f"{title_suffix} (pass@{num_attempts}){subtitle}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Shortest-path distance", fontsize=10)
        ax.set_ylabel("Success rate (%)", fontsize=10)
        ax.set_xticks(distances)
        ax.set_ylim(0, 115)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {save_path}")


# ---------------------------------------------------------------------------
# Build graph + BFS pairs (shared between models)
# ---------------------------------------------------------------------------


def build_graph_and_pairs(graph_path: str, sample_sources: int, max_dist: int, seed: int):
    with open(graph_path) as f:
        graph_data = json.load(f)

    directed = graph_data.get("directed", False)
    G = nx.DiGraph() if directed else nx.Graph()
    for node, neighbors in graph_data["adjacency"].items():
        for nb in neighbors:
            G.add_edge(int(node), int(nb))

    kind = "directed" if directed else "undirected"
    print(f"Graph ({kind}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    nodes = list(G.nodes())
    pairs_by_dist: dict = defaultdict(list)
    src_sample = random.sample(nodes, min(sample_sources, len(nodes)))

    for src in tqdm(src_sample, desc="Computing shortest paths"):
        lengths = nx.single_source_shortest_path_length(G, src)
        for tgt, d in lengths.items():
            if tgt != src:
                pairs_by_dist[d].append((src, tgt))

    print(f"Max distance found: {max(pairs_by_dist.keys())}")
    return G, pairs_by_dist


# ---------------------------------------------------------------------------
# Per-model evaluation helper
# ---------------------------------------------------------------------------


def eval_checkpoint(checkpoint_path, is_posttrain, tokenizer, cfg, device, args,
                    G, pairs_by_dist):
    print(f"\n  Loading: {checkpoint_path}")
    if is_posttrain:
        model = load_posttrain_model(checkpoint_path, tokenizer, cfg, device)
    else:
        model = load_pretrain_model(checkpoint_path, tokenizer, cfg, device)

    gen_fn = make_generate_fn(model, tokenizer, device,
                              args.temperature, args.top_k, args.max_new_tokens)
    results = run_eval(gen_fn, G, pairs_by_dist, args.M, args.num_attempts, args.max_dist)
    del model
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    fallback_cfg = OmegaConf.load(args.config_path)

    G, pairs_by_dist = build_graph_and_pairs(
        args.graph_path, args.sample_sources, args.max_dist, args.seed
    )

    # ── Single-pair mode ───────────────────────────────────────────────────
    if args.pretrain_checkpoint and args.posttrain_checkpoint:
        cfg = fallback_cfg
        train_max = args.train_max if args.train_max is not None else cfg.data.max_path_length
        posttrain_max = args.posttrain_max if args.posttrain_max is not None else train_max

        print(f"Pretraining max path length : {train_max}")
        print(f"Posttraining max path length: {posttrain_max}")

        print(f"\n{'='*60}\nEvaluating PRETRAIN: {args.pretrain_checkpoint}\n{'='*60}")
        pretrain_results = eval_checkpoint(args.pretrain_checkpoint, False,
                                           tokenizer, cfg, device, args, G, pairs_by_dist)

        print(f"\n{'='*60}\nEvaluating POSTTRAIN: {args.posttrain_checkpoint}\n{'='*60}")
        posttrain_results = eval_checkpoint(args.posttrain_checkpoint, True,
                                            tokenizer, cfg, device, args, G, pairs_by_dist)

        plot_results(pretrain_results, train_max=train_max, posttrain_max=None,
                     num_attempts=args.num_attempts,
                     title=f"Length Generalisation — Pretrain only (pass@{args.num_attempts})",
                     save_path=os.path.join(args.output_dir, "length_generalization_eval.png"))

        plot_results(posttrain_results, train_max=train_max, posttrain_max=posttrain_max,
                     num_attempts=args.num_attempts,
                     title=f"Length Generalisation — After posttraining (pass@{args.num_attempts})",
                     save_path=os.path.join(args.output_dir, "length_generalization_eval_posttrain.png"))

        plot_comparison(pretrain_results, posttrain_results,
                        train_max=train_max, posttrain_max=posttrain_max,
                        num_attempts=args.num_attempts,
                        save_path=os.path.join(args.output_dir, "length_generalization_comparison.png"))

        print("\nDone.")
        return

    # ── Directory mode ─────────────────────────────────────────────────────
    if not (args.pretrain_dir and args.posttrain_dir):
        raise ValueError(
            "Provide either (--pretrain_checkpoint + --posttrain_checkpoint) "
            "or (--pretrain_dir + --posttrain_dir)."
        )

    pretrain_runs = discover_pretrain_runs(Path(args.pretrain_dir))
    posttrain_runs = discover_posttrain_runs(Path(args.posttrain_dir))
    print(f"Found {len(pretrain_runs)} pretrain runs, {len(posttrain_runs)} posttrain runs")

    if not pretrain_runs or not posttrain_runs:
        print("No runs found — nothing to evaluate.")
        return

    # --- Criterion 1: same max data length ---
    length_pairs = match_by_max_length(pretrain_runs, posttrain_runs)
    print(f"\nCriterion 1 (same max length): {len(length_pairs)} matching pairs")

    # --- Criterion 2: same total data ---
    data_pairs = match_by_total_data(pretrain_runs, posttrain_runs,
                                     tolerance=args.total_data_tolerance)
    print(f"Criterion 2 (same total data ±{args.total_data_tolerance:.0%}): {len(data_pairs)} matching pairs")

    # Cache eval results keyed by checkpoint path to avoid re-running
    results_cache: dict[str, dict] = {}

    def get_results(meta, is_posttrain):
        ckpt = meta["checkpoint_path"]
        if ckpt not in results_cache:
            cfg = OmegaConf.load(meta["config_path"]) if "config_path" in meta else fallback_cfg
            results_cache[ckpt] = eval_checkpoint(
                ckpt, is_posttrain, tokenizer, cfg, device, args, G, pairs_by_dist
            )
        return results_cache[ckpt]

    # --- Plot criterion 1 pairs ---
    for i, (pre, post) in enumerate(length_pairs):
        max_len = pre["max_path_length"]
        n_max = post.get("n_max", max_len)
        pre_cfg = OmegaConf.load(pre["config_path"]) if "config_path" in pre else fallback_cfg
        post_cfg = OmegaConf.load(post["config_path"]) if "config_path" in post else fallback_cfg

        print(f"\n{'='*60}")
        print(f"[Length match {i+1}/{len(length_pairs)}] max_path_length={max_len}")
        print(f"  Pretrain : {pre['checkpoint_path']}")
        print(f"  Posttrain: {post['checkpoint_path']}")
        print(f"{'='*60}")

        pre_res = get_results(pre, is_posttrain=False)
        post_res = get_results(post, is_posttrain=True)

        train_max = args.train_max if args.train_max is not None else max_len
        posttrain_max = args.posttrain_max if args.posttrain_max is not None else n_max

        stem = f"length_match_{i+1:02d}_maxlen{max_len}"
        plot_comparison(
            pre_res, post_res,
            train_max=train_max, posttrain_max=posttrain_max,
            num_attempts=args.num_attempts,
            save_path=os.path.join(args.output_dir, f"{stem}_comparison.png"),
            subtitle=f"\nmaxlen={max_len}",
        )
        plot_results(pre_res, train_max=train_max, posttrain_max=None,
                     num_attempts=args.num_attempts,
                     title=f"Pretrain only — max_path_length={max_len} (pass@{args.num_attempts})",
                     save_path=os.path.join(args.output_dir, f"{stem}_pretrain.png"))
        plot_results(post_res, train_max=train_max, posttrain_max=posttrain_max,
                     num_attempts=args.num_attempts,
                     title=f"After posttraining — n_max={n_max} (pass@{args.num_attempts})",
                     save_path=os.path.join(args.output_dir, f"{stem}_posttrain.png"))

    # --- Plot criterion 2 pairs ---
    for i, (pre, post) in enumerate(data_pairs):
        pre_total = pre.get("samples_seen", 0)
        post_total = post.get("total_samples_seen", 0)
        max_len = pre.get("max_path_length", args.train_max or 0)
        n_max = post.get("n_max", args.posttrain_max or max_len)

        print(f"\n{'='*60}")
        print(f"[Data match {i+1}/{len(data_pairs)}] pretrain={pre_total:,} ≈ posttrain_total={post_total:,}")
        print(f"  Pretrain : {pre['checkpoint_path']}")
        print(f"  Posttrain: {post['checkpoint_path']}")
        print(f"{'='*60}")

        pre_res = get_results(pre, is_posttrain=False)
        post_res = get_results(post, is_posttrain=True)

        train_max = args.train_max if args.train_max is not None else max_len
        posttrain_max = args.posttrain_max if args.posttrain_max is not None else n_max

        stem = f"data_match_{i+1:02d}_pretrain{pre_total}_total{post_total}"
        plot_comparison(
            pre_res, post_res,
            train_max=train_max, posttrain_max=posttrain_max,
            num_attempts=args.num_attempts,
            save_path=os.path.join(args.output_dir, f"{stem}_comparison.png"),
            subtitle=f"\npretrain_samples={pre_total:,} | posttrain_total={post_total:,}",
        )
        plot_results(pre_res, train_max=train_max, posttrain_max=None,
                     num_attempts=args.num_attempts,
                     title=f"Pretrain only — {pre_total:,} samples (pass@{args.num_attempts})",
                     save_path=os.path.join(args.output_dir, f"{stem}_pretrain.png"))
        plot_results(post_res, train_max=train_max, posttrain_max=posttrain_max,
                     num_attempts=args.num_attempts,
                     title=f"After posttraining — {post_total:,} total samples (pass@{args.num_attempts})",
                     save_path=os.path.join(args.output_dir, f"{stem}_posttrain.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
