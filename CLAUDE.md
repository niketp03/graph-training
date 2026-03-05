Build a Python project with two main components: a data generation script and a PyTorch Lightning training pipeline for next-token prediction on graph trajectories.

## 1. `generate.py` — Data Generation

This script generates synthetic training data from random graphs.

**Graph generation:**
- Sample a large Erdos-Renyi random graph (configurable number of nodes and edge probability/density). Prune the graph to keep only the largest connected component. 
- Save the entire graph structure as a JSON file (e.g., `graph.json`) with nodes and adjacency info.
- Graph structure in the JSON file should have nodes and then for each node a list of all the other nodes it is connected to. 

**Trajectory generation:**
- Generate a configurable number of trajectories of configurable max length.
- Each trajectory is a string/sequence with this format:
  `<start_goal> START_NODE END_NODE <end_goal> START_NODE NODE_A NODE_B NODE_C ... END_NODE`
  So it begins with a `<start_goal>` special token, then the start node, then the target/end node, then an `<end_goal>` special token, followed by the actual path of nodes from start to end.
- Sample paths across varying lengths up to the specified max length.
- IMPORTANT: The set of all generated trajectories must collectively cover every node in the graph at least once.
- Split trajectories into three files: `train.txt`, `val.txt`, `test.txt` — each file contains one trajectory string per line.

**Vocabulary generation:**
- Produce a `vocabulary.json` file listing all vocabulary items: the node names/IDs plus special tokens (`<start_goal>`, `<end_goal>`, `<pad>`, `<eos>`, etc.).

**CLI args for generate.py:** number of nodes, edge probability, number of trajectories, max trajectory length, train/val/test split ratios, output directory, random seed.


## 2. Training Pipeline — PyTorch Lightning + HuggingFace Tokenizer

**Tokenizer:**
- Read `vocabulary.json` and build a HuggingFace-compatible tokenizer (use the `tokenizers` library to create a custom tokenizer — e.g., a WordLevel tokenizer with the provided vocab).
- Save the tokenizer in HuggingFace format so it can be loaded with `AutoTokenizer.from_pretrained()` or similar.

**Dataset / DataModule:**
- A Lightning DataModule that reads `train.txt`, `val.txt`, `test.txt`.
- Each line is one trajectory string. Tokenize using the HuggingFace tokenizer.
- Prepare batches for causal next-token prediction (input = tokens[:-1], target = tokens[1:]), with padding/collation.

**Model Specification**
- For the actual model, use the huggingface class for Qwen 2.5
- You can use the config for Qwen 2.5, 0.8B to get started.
- Randomly initialize the model with our custom vocabulary
**Model:**
- Configurable: embedding dim, number of layers, number of heads, feedforward dim, dropout, max sequence length (pick these from the qwen config)
- Use cross-entropy loss over the vocabulary.

**Training:**
- Use PyTorch Lightning Trainer.
- Use Weights & Biases (`WandbLogger`) as the default logger.
- Support standard Lightning CLI args (max epochs, gpus, precision, etc.).
- Include validation loop with val loss logging.


**Config**
Use hydra to deal with the configs of the project. Each run will be associated with a single config.py that should have the following sections: 

Data:
- Total number of nodes 
- connectivity of the Erdos-Renyi graph 
- max path-length in the training data 
- sampling probability for an edge p

Train:
- lr
- gradient accumulation 
- gradient clip 
... and anything else about the model 




**Project structure:**
project/
├── generate.py
├── train.py
├── model.py
├── data.py
├── tokenizer_utils.py
├── requirements.txt
└── README.md

Use clean, well-documented code. All hyperparameters should be configurable via CLI arguments (use argparse or LightningCLI). Include a README with usage examples for both generate.py and train.py.