"""
eval_comparison.py

Evaluates pretrain-only (Run A) and all posttrain-sweep (Run B) models up to
N_MAX_EVAL distance, then plots all results as lines on a single figure.

Discovery:
  Auto-mode  — pass --base_dir and --posttrain_percents (matches run_comparison.sh layout)
  Manual mode — pass --posttrain_dirs and optionally --posttrain_labels

Usage (auto-mode):
    python eval_comparison.py \\
        --pretrain_ckpt ./checkpoints_pretrain_only/last.ckpt \\
        --base_dir . \\
        --posttrain_percents 10 20 30 40 50 60 70 80 90 \\
        --tokenizer_dir ./data_output_more_trajectories/tokenizer \\
        --graph_path ./data_output_more_trajectories/graph.json \\
        --n_max_eval 30 \\
        --train_max 15

Usage (manual mode):
    python eval_comparison.py \\
        --pretrain_ckpt ./checkpoints_pretrain_only/last.ckpt \\
        --posttrain_dirs ./checkpoints_posttrain_prop10 ./checkpoints_posttrain_prop50 \\
        --posttrain_labels "10% posttrain" "50% posttrain" \\
        --tokenizer_dir ./data_output_more_trajectories/tokenizer \\
        --graph_path ./data_output_more_trajectories/graph.json \\
        --n_max_eval 30 --train_max 15
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from eval_length_generalization import (
    build_graph_and_pairs,
    load_posttrain_model,
    load_pretrain_model,
    make_generate_fn,
    run_eval,
)
from tokenizer_utils import load_tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Comparison eval: pretrain-only vs posttrain-proportion sweep"
    )

    p.add_argument("--pretrain_ckpt", required=True,
                   help="Path to pretrain-only checkpoint (.ckpt)")

    # Auto-discovery (matches run_comparison.sh naming)
    p.add_argument("--base_dir", default=".",
                   help="Base directory containing checkpoints_posttrain_prop* dirs")
    p.add_argument("--posttrain_percents", nargs="+", type=int, default=[],
                   help="Post-train proportion percentages to discover, e.g. 10 20 30")

    # Manual specification
    p.add_argument("--posttrain_dirs", nargs="+", default=[],
                   help="Explicit posttrain checkpoint directories")
    p.add_argument("--posttrain_labels", nargs="+", default=[],
                   help="Labels for --posttrain_dirs (defaults to dir name)")

    # Required
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--graph_path", required=True)

    # Eval parameters
    p.add_argument("--n_max_eval", type=int, default=30,
                   help="Evaluate paths at distances 1..n_max_eval")
    p.add_argument("--train_max", type=int, default=None,
                   help="Pretraining max path length (vertical line on plot). "
                        "Defaults to data.max_path_length from config.")
    p.add_argument("--M", type=int, default=50,
                   help="(u,v) pairs sampled per distance")
    p.add_argument("--num_attempts", type=int, default=1,
                   help="Generation attempts per pair (pass@k)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--sample_sources", type=int, default=500,
                   help="Source nodes for BFS when building pairs-by-distance")

    p.add_argument("--config_path", default=".hydra/config.yaml",
                   help="Hydra config YAML (used as fallback for model/train sections)")
    p.add_argument("--output_dir", default=".",
                   help="Directory where the plot and results JSON are saved")
    p.add_argument("--results_json", default=None,
                   help="Load previously computed results from this JSON and skip eval "
                        "(useful for re-plotting without re-running inference)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def find_posttrain_ckpt(directory: str) -> str | None:
    """Return the path to postrain_final.pt inside the newest run_*/ subdir."""
    base = Path(directory)
    # Prefer run_*/ subdirs (newest first)
    candidates = sorted(base.glob("run_*/postrain_final.pt"), reverse=True)
    if candidates:
        return str(candidates[0])
    # Fall back to a file directly in directory
    direct = base / "postrain_final.pt"
    if direct.exists():
        return str(direct)
    return None


def resolve_posttrain_entries(args) -> list[tuple[str, str]]:
    """Return list of (label, directory) for all posttrain runs to evaluate."""
    entries: list[tuple[str, str]] = []

    # Auto-discovery from base_dir + percentages
    for pct in args.posttrain_percents:
        d = os.path.join(args.base_dir, f"checkpoints_posttrain_prop{pct}")
        entries.append((f"{pct}% posttrain", d))

    # Manual specification
    if args.posttrain_dirs:
        labels = args.posttrain_labels or [Path(d).name for d in args.posttrain_dirs]
        for label, d in zip(labels, args.posttrain_dirs):
            entries.append((label, d))

    return entries


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def eval_model(ckpt_path, is_posttrain, tokenizer, cfg, device, args, G, pairs_by_dist):
    if is_posttrain:
        model = load_posttrain_model(ckpt_path, tokenizer, cfg, device)
    else:
        model = load_pretrain_model(ckpt_path, tokenizer, cfg, device)

    # max_new_tokens == n_max_eval (paths up to n_max_eval nodes long)
    gen_fn = make_generate_fn(
        model, tokenizer, device,
        args.temperature, args.top_k, args.n_max_eval
    )
    results = run_eval(gen_fn, G, pairs_by_dist, args.M, args.num_attempts, args.n_max_eval)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sweep(all_results: dict, train_max: int | None, n_max_eval: int,
               num_attempts: int, output_dir: str):
    """Single line plot — one line per model, dashed for pretrain-only."""
    fig, ax = plt.subplots(figsize=(max(12, n_max_eval * 0.65), 5))

    cmap = plt.cm.tab10.colors
    color_idx = 0

    for label, results in all_results.items():
        dists = sorted(results.keys())
        accs = [results[d]["successes"] / results[d]["total"] * 100 for d in dists]

        is_pretrain = label == "Pretrain only"
        ax.plot(
            dists, accs,
            label=label,
            linestyle="--" if is_pretrain else "-",
            linewidth=2.5 if is_pretrain else 1.8,
            color="black" if is_pretrain else cmap[color_idx],
            marker="o" if is_pretrain else ".",
            markersize=5 if is_pretrain else 4,
            zorder=10 if is_pretrain else 5,
        )
        if not is_pretrain:
            color_idx = (color_idx + 1) % len(cmap)

    if train_max is not None:
        ax.axvline(x=train_max, color="gray", linestyle=":", linewidth=1.5,
                   label=f"Pretrain max path length = {train_max}")

    ax.set_xlabel("Shortest-path distance between (u, v)", fontsize=12)
    ax.set_ylabel(f"Success rate (%)  [pass@{num_attempts}]", fontsize=12)
    ax.set_title(
        "Length Generalisation: Pretrain-only vs Posttrain proportion sweep",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(range(1, n_max_eval + 1))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    save_path = os.path.join(output_dir, "eval_sweep_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Optionally skip inference and re-plot from saved JSON
    if args.results_json and os.path.exists(args.results_json):
        print(f"Loading saved results from {args.results_json}")
        with open(args.results_json) as f:
            all_results = json.load(f)
        # Reconstruct int keys (JSON serialises them as strings)
        all_results = {
            label: {int(d): v for d, v in res.items()}
            for label, res in all_results.items()
        }
        cfg = OmegaConf.load(args.config_path)
        train_max = args.train_max if args.train_max is not None else cfg.data.max_path_length
        plot_sweep(all_results, train_max, args.n_max_eval, args.num_attempts, args.output_dir)
        return

    tokenizer = load_tokenizer(args.tokenizer_dir)
    cfg = OmegaConf.load(args.config_path)
    train_max = args.train_max if args.train_max is not None else cfg.data.max_path_length

    G, pairs_by_dist = build_graph_and_pairs(
        args.graph_path, args.sample_sources, args.n_max_eval, args.seed
    )

    posttrain_entries = resolve_posttrain_entries(args)

    if not posttrain_entries:
        print("WARNING: no posttrain runs specified. Pass --posttrain_percents or --posttrain_dirs.")

    # ── Evaluate all models ─────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    print(f"\n{'='*60}")
    print(f"Evaluating Pretrain-only: {args.pretrain_ckpt}")
    print(f"{'='*60}")
    all_results["Pretrain only"] = eval_model(
        args.pretrain_ckpt, False, tokenizer, cfg, device, args, G, pairs_by_dist
    )

    for label, d in posttrain_entries:
        ckpt = find_posttrain_ckpt(d)
        if ckpt is None:
            print(f"WARNING: no postrain_final.pt found under {d} — skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating {label}: {ckpt}")
        print(f"{'='*60}")
        # Load the run's own config if present, otherwise fall back to global cfg
        run_cfg_path = Path(ckpt).parent / "config.yaml"
        run_cfg = OmegaConf.load(str(run_cfg_path)) if run_cfg_path.exists() else cfg
        all_results[label] = eval_model(
            ckpt, True, tokenizer, run_cfg, device, args, G, pairs_by_dist
        )

    # ── Save raw results ─────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "eval_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results → {results_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_sweep(all_results, train_max, args.n_max_eval, args.num_attempts, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
