#!/bin/bash
# sweep_lr.sh — Launch one SLURM job for lr=1e-4 with n_max=10
ACCOUNT="torch_pr_236_cds"
CONSTRAINT="a100|h200|l40s"
LR="1e-4"

JOB_NAME="postrain_lr${LR}_nmax10"
OUTLOG="$SCRATCH/graph-training/postrain_lr${LR}_nmax10.log"
CMD="cd \$SCRATCH/graph-training && source .venv/bin/activate && python postrain.py postrain.lr=${LR} postrain.n_max=10 postrain.wandb_run_name=lr_${LR}_nmax10"

JOB_ID=$(sbatch \
    --account="${ACCOUNT}" \
    --constraint="${CONSTRAINT}" \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=4 \
    --time=48:00:00 \
    --job-name="${JOB_NAME}" \
    --output="${OUTLOG}" \
    --wrap="${CMD}" \
    --parsable)

echo "Submitted lr=${LR} n_max=10  ->  job ${JOB_ID}  (log: ${OUTLOG})"
