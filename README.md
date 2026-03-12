# Graph Training

Next-token prediction on graph trajectories using a randomly-initialized Qwen2 transformer.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Data

Generate a random graph and trajectory data:

```bash
python generate.py
```

Override defaults via Hydra CLI:

```bash
python generate.py data.num_nodes=200 data.edge_probability=0.15 data.num_trajectories=20000 data.max_path_length=30 data.output_dir=./my_data

# Generate a directed graph
python generate.py data.directed=true data.num_nodes=200 data.edge_probability=0.15 data.output_dir=./my_directed_data
```

This produces in the output directory:
- `graph.json` — graph structure (nodes + adjacency lists)
- `vocabulary.json` — token vocabulary (node IDs + special tokens)
- `train.txt`, `val.txt`, `test.txt` — trajectory splits (one per line)

### 2. Train

```bash
python train.py
```

Override training config:

```bash
python train.py train.lr=3e-4 train.batch_size=64 train.max_epochs=20 model.num_hidden_layers=12
```

Training uses Weights & Biases for logging. Set `WANDB_API_KEY` or run `wandb login` first.

### Configuration

All parameters are managed via Hydra. See `conf/config.yaml` for defaults. Sections:

- **data** — graph size, edge probability, directed/undirected, trajectory count, path length, splits, seed
- **model** — Qwen2 architecture params (hidden size, layers, heads, etc.)
- **train** — learning rate, batch size, epochs, gradient clipping, precision, W&B settings

## Project Structure

```
├── conf/
│   └── config.yaml          # Hydra configuration
├── generate.py               # Data generation script
├── train.py                  # Training entrypoint
├── model.py                  # LightningModule (Qwen2ForCausalLM)
├── data.py                   # LightningDataModule + Dataset
├── tokenizer_utils.py        # HuggingFace tokenizer builder
├── requirements.txt          # Dependencies
├── pyproject.toml            # Project metadata
└── CLAUDE.md                 # Specification
```
