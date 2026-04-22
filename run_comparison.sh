#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate multi_312

# Comparison experiment: pretrain-only (Run A) vs pretrain+posttrain (Run B).
#
# Fixed total sequence budget is split between pretraining and post-training.
# Run A (pretrain-only) always consumes the full budget.
# Run B sweeps over what fraction of the budget goes to post-training.
#
# For a given post-train proportion p:
#   Pretrain seqs  = (1-p) * TOTAL_SEQS  → converted to epochs (floor)
#   Posttrain seqs =     p * TOTAL_SEQS  → converted to steps  (floor)

# ─── Fixed parameters ────────────────────────────────────────────────────────
N_TRAIN=800000        # sequences per pretraining epoch (num_trajectories * train_ratio)
POSTRAIN_PAIRS=256    # postrain.num_pairs_per_step
N_MAX=15              # n_max == max_new_tokens for posttraining

TOTAL_SEQS=10560000   # total sequence budget shared by all runs

# Post-training proportions to sweep (integer percentages, e.g. 10 = 10%)
POSTTRAIN_PERCENTS=(10 20 30 40 50 60 70 80 90)

# ─── Pretrain-only (Run A) epochs — fixed, covers full budget ────────────────
PRETRAIN_ONLY_EPOCHS=$(( (TOTAL_SEQS + N_TRAIN - 1) / N_TRAIN ))

echo "============================================================"
echo "Total sequence budget:          ${TOTAL_SEQS}"
echo "N_TRAIN (seqs/epoch):           ${N_TRAIN}"
echo "POSTRAIN_PAIRS (seqs/step):     ${POSTRAIN_PAIRS}"
echo "Run A pretrain-only epochs:     ${PRETRAIN_ONLY_EPOCHS}"
echo "  (actual seqs: $(( PRETRAIN_ONLY_EPOCHS * N_TRAIN )))"
echo "Sweep proportions (%):          ${POSTTRAIN_PERCENTS[*]}"
echo "============================================================"
echo ""

# ─── Data generation ─────────────────────────────────────────────────────────
DATA_DIR="./data_output_more_trajectories"
if [ -f "${DATA_DIR}/train.txt" ]; then
    echo ">>> Data already exists at ${DATA_DIR}, skipping generation"
else
    echo ">>> Generating training data"
    python generate.py \
        data.num_nodes=10000 \
        data.edge_probability=0.0001 \
        data.directed=true \
        data.num_trajectories=1000000 \
        data.max_path_length=30 \
        data.output_dir="${DATA_DIR}"
fi

# ─── Run A: Pretrain-only ─────────────────────────────────────────────────────
echo ""
echo ">>> Run A: pretrain-only for ${PRETRAIN_ONLY_EPOCHS} epochs"
python train.py \
    train.max_epochs="${PRETRAIN_ONLY_EPOCHS}" \
    train.wandb_run_name="pretrain_only_${PRETRAIN_ONLY_EPOCHS}ep_total${TOTAL_SEQS}" \
    train.checkpoint_dir="./checkpoints_pretrain_only" \
    data.output_dir="${DATA_DIR}"

# ─── Run B: Sweep over post-training proportion ───────────────────────────────
for PCT in "${POSTTRAIN_PERCENTS[@]}"; do
    POSTTRAIN_SEQS=$(( TOTAL_SEQS * PCT / 100 ))
    PRETRAIN_SEQS=$(( TOTAL_SEQS - POSTTRAIN_SEQS ))
    PRETRAIN_EPOCHS=$(( PRETRAIN_SEQS / N_TRAIN ))       # floor
    POSTTRAIN_STEPS=$(( POSTTRAIN_SEQS / POSTRAIN_PAIRS ))  # floor

    echo ""
    echo "============================================================"
    echo ">>> Run B | posttrain proportion = ${PCT}%"
    echo "    Pretrain  seqs: ${PRETRAIN_SEQS}  → ${PRETRAIN_EPOCHS} epochs"
    echo "    Posttrain seqs: ${POSTTRAIN_SEQS} → ${POSTTRAIN_STEPS} steps"
    echo "============================================================"

    CKPT_DIR="./checkpoints_posttrain_prop${PCT}"
    PRETRAIN_CKPT_DIR="./checkpoints_for_posttrain_prop${PCT}"

    # Phase 1: pretrain
    echo ">>> Run B prop=${PCT}%: pretrain for ${PRETRAIN_EPOCHS} epochs"
    python train.py \
        train.max_epochs="${PRETRAIN_EPOCHS}" \
        train.wandb_run_name="pretrain_for_posttrain_prop${PCT}pct" \
        train.checkpoint_dir="${PRETRAIN_CKPT_DIR}" \
        data.output_dir="${DATA_DIR}"

    # Phase 2: posttrain
    echo ">>> Run B prop=${PCT}%: posttrain for ${POSTTRAIN_STEPS} steps"
    python postrain.py \
        postrain.checkpoint_path="${PRETRAIN_CKPT_DIR}/last.ckpt" \
        postrain.max_steps="${POSTTRAIN_STEPS}" \
        postrain.num_pairs_per_step="${POSTRAIN_PAIRS}" \
        postrain.n_max="${N_MAX}" \
        postrain.max_new_tokens="${N_MAX}" \
        postrain.wandb_run_name="posttrain_prop${PCT}pct_${POSTTRAIN_STEPS}steps" \
        postrain.checkpoint_dir="${CKPT_DIR}" \
        data.output_dir="${DATA_DIR}"
done

echo ""
echo "Done. All runs complete."

# ─── Eval: compare all models on a single plot ────────────────────────────────
N_MAX_EVAL=30      # evaluate paths up to this distance

echo ""
echo ">>> Evaluating all models up to distance ${N_MAX_EVAL}"
POSTTRAIN_PCT_ARGS=""
for PCT in "${POSTTRAIN_PERCENTS[@]}"; do
    POSTTRAIN_PCT_ARGS="${POSTTRAIN_PCT_ARGS} ${PCT}"
done

python eval_comparison.py \
    --pretrain_ckpt "./checkpoints_pretrain_only/last.ckpt" \
    --base_dir "." \
    --posttrain_percents ${POSTTRAIN_PCT_ARGS} \
    --tokenizer_dir "${DATA_DIR}/tokenizer" \
    --graph_path "${DATA_DIR}/graph.json" \
    --n_max_eval "${N_MAX_EVAL}" \
    --train_max "${N_MAX}" \
    --output_dir "./eval_results"

echo ">>> Eval complete. Plot saved to ./eval_results/eval_sweep_comparison.png"
