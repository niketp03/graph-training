"""RLVR-style post-training for graph trajectory models.

Samples (u, v) node pairs (via uniform_pairs or uniform_specified sampling),
generates trajectories with the model, verifies path validity against the
graph, and trains using policy gradient methods (REINFORCE or GRPO).
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import hydra
import networkx as nx
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from model import GraphTrajectoryLM
from tokenizer_utils import load_tokenizer  
from teacher import TeacherModel

class GraphEnvironment:
    """Graph structure for sampling node pairs and verifying generated paths."""

    def __init__(self, graph_path: str):
        with open(graph_path) as f:
            data = json.load(f)
        self.directed = data["directed"]
        self.nodes: list[int] = data["nodes"]
        self.adj: dict[int, set[int]] = {}
        for node_str, neighbors in data["adjacency"].items():
            self.adj[int(node_str)] = set(neighbors)

        self.graph = nx.DiGraph() if self.directed else nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        for node, neighbors in self.adj.items():
            for nbr in neighbors:
                self.graph.add_edge(node, nbr)

    def sample_pairs(self, n: int, rng: random.Random) -> list[tuple[int, int]]:
        """Uniformly sample n (u, v) pairs where u != v."""
        pairs = []
        for _ in range(n):
            u, v = rng.sample(self.nodes, 2)
            pairs.append((u, v))
        return pairs

    def sample_pairs_connected(self, n: int, rng: random.Random, max_retries: int = 500) -> list[tuple[int, int]]:
        """Uniformly sample n (u, v) pairs where u is connected to v."""
        pairs = []
        for _ in range(n):
            for _ in range(max_retries):
                u, v = rng.sample(self.nodes, 2)
                if nx.has_path(self.graph, u, v):
                    pairs.append((u, v))
                    break

        #print(f"Sampled {len(pairs)} pairs, desired {n}")
        return pairs

    def _random_walk(
        self, length: int, rng: random.Random, max_retries: int = 100
    ) -> list[int] | None:
        """Return a random walk of exactly *length* edges (length+1 nodes).

        Retries from a fresh random start on dead-ends.  Returns ``None``
        if no valid walk is found within *max_retries* attempts.
        """
        for _ in range(max_retries):
            start = rng.choice(self.nodes)
            path = [start]
            ok = True
            for _ in range(length):
                nbrs = self.adj.get(path[-1], set())
                if not nbrs:
                    ok = False
                    break
                path.append(rng.choice(list(nbrs)))
            if ok and len(path) == length + 1:
                return path
        return None

    def sample_specified_pairs(
        self, n: int, n_max: int, rng: random.Random
    ) -> list[tuple[int, int]]:
        """Uniformly sample a path length in 1..n_max, find a path of that
        length via random walk, and return (start, end) of the path.

        Repeats *n* times to produce *n* pairs.
        """
        pairs: list[tuple[int, int]] = []
        for _ in range(n):
            length = rng.randint(1, n_max)
            path = self._random_walk(length, rng)
            if path is None:
                u, v = rng.sample(self.nodes, 2)
                pairs.append((u, v))
            else:
                pairs.append((path[0], path[-1]))
        return pairs

    def verify_path(self, start: int, end: int, path_nodes: list[int]) -> bool:
        """Check whether path_nodes is a valid walk from start to end."""
        if not path_nodes:
            return False
        if path_nodes[0] != start or path_nodes[-1] != end:
            return False
        for i in range(len(path_nodes) - 1):
            if path_nodes[i + 1] not in self.adj.get(path_nodes[i], set()):
                return False
        return True

class TeacherEnvironment:
    """Teacher model for task (node-pair) sampling and optional policy-gradient updates."""
    def __init__(self, teacher_model: TeacherModel):
        self.teacher_model = teacher_model

    def sample_pairs(self, n: int, rng: random.Random) -> list[tuple[int, int]]:
        return self.teacher_model.sample_tasks(n)

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def build_prompt_strings(pairs: list[tuple[int, int]]) -> list[str]:
    return [f"<start_goal> {u} {v} <end_goal>" for u, v in pairs]


@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    prompt_strings: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Generate completions for a batch of prompts.

    Returns (generated_ids [B, T], prompt_len) where prompt_len is the
    padded prompt length (identical for every row thanks to left-padding).
    """
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    encodings = tokenizer(
        prompt_strings, return_tensors="pt", padding=True
    ).to(device)
    prompt_len = encodings.input_ids.shape[1]

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = model.generate(
        input_ids=encodings.input_ids,
        attention_mask=encodings.attention_mask,
        **gen_kwargs,
    )

    tokenizer.padding_side = orig_side
    return generated_ids, prompt_len


def parse_and_verify(
    generated_ids: torch.Tensor,
    prompt_len: int,
    pairs: list[tuple[int, int]],
    tokenizer,
    env: GraphEnvironment,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Parse generated sequences and assign binary rewards.

    Returns (rewards [B], info_dict).
    """
    B = generated_ids.shape[0]
    rewards = torch.full((B,), incorrect_reward)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    num_valid = 0
    num_reached_end = 0
    total_gen_len = 0

    for i in range(B):
        start, end = pairs[i]
        response_ids = generated_ids[i, prompt_len:].tolist()

        clean_ids: list[int] = []
        for tid in response_ids:
            if tid in (eos_id, pad_id):
                break
            clean_ids.append(tid)

        total_gen_len += len(clean_ids)

        try:
            path_nodes = [int(tokenizer.decode([tid]).strip()) for tid in clean_ids]
        except (ValueError, KeyError):
            continue

        if path_nodes and path_nodes[-1] == end:
            num_reached_end += 1

        if env.verify_path(start, end, path_nodes):
            rewards[i] = correct_reward
            num_valid += 1

    info = {
        "accuracy": num_valid / B if B > 0 else 0.0,
        "reached_end_rate": num_reached_end / B if B > 0 else 0.0,
        "avg_gen_length": total_gen_len / B if B > 0 else 0.0,
        "num_valid": num_valid,
    }
    return rewards, info


# ---------------------------------------------------------------------------
# Log-prob / masking utilities
# ---------------------------------------------------------------------------

def compute_token_log_probs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Per-token log probs, shape [B, T-1].

    Position t holds log P(token[t+1] | token[:t+1]).
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    return F.log_softmax(shift_logits, dim=-1).gather(
        -1, shift_targets.unsqueeze(-1)
    ).squeeze(-1)


def get_response_mask(
    input_ids: torch.Tensor, prompt_len: int, pad_token_id: int
) -> torch.Tensor:
    """Mask that is 1 for response tokens in the shifted (T-1) space."""
    B, T = input_ids.shape
    targets = input_ids[:, 1:]  # [B, T-1]
    mask = torch.zeros(B, T - 1, device=input_ids.device)
    if prompt_len - 1 < T - 1:
        mask[:, prompt_len - 1 :] = (targets[:, prompt_len - 1 :] != pad_token_id).float()
    return mask


# ---------------------------------------------------------------------------
# Algorithm implementations
# ---------------------------------------------------------------------------

def reinforce_step(
    model,
    optimizer,
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    baseline: str,
    gradient_clip_val: float,
) -> dict:
    """REINFORCE policy gradient update (per-token formulation)."""
    device = generated_ids.device
    rewards = rewards.to(device)

    token_log_probs = compute_token_log_probs(model, generated_ids, attention_mask)

    if baseline == "mean":
        advantages = rewards - rewards.mean()
    else:
        advantages = rewards

    token_advantages = advantages.unsqueeze(1) * response_mask
    loss = -(token_log_probs * token_advantages).sum() / response_mask.sum().clamp(min=1)

    optimizer.zero_grad()
    loss.backward()
    if gradient_clip_val > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
    optimizer.step()

    return {"loss": loss.item(), "mean_advantage": advantages.mean().item()}


def grpo_step(
    model,
    ref_model,
    optimizer,
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
    clip_range: float,
    kl_coeff: float,
    num_ppo_epochs: int,
    gradient_clip_val: float,
) -> dict:
    """Group Relative Policy Optimization update."""
    device = generated_ids.device
    rewards = rewards.to(device)
    B = generated_ids.shape[0]
    num_groups = B // group_size

    # --- group-normalised advantages ---
    rg = rewards.view(num_groups, group_size)
    g_mean = rg.mean(dim=1, keepdim=True)
    g_std = rg.std(dim=1, keepdim=True)
    advantages = torch.where(
        g_std > 1e-6,
        (rg - g_mean) / g_std,
        torch.zeros_like(rg),
    ).view(B)

    # old / ref log-probs (frozen)
    with torch.no_grad():
        old_token_lp = compute_token_log_probs(model, generated_ids, attention_mask)
        ref_token_lp = compute_token_log_probs(ref_model, generated_ids, attention_mask)

    total_policy = 0.0
    total_kl = 0.0

    for _ in range(num_ppo_epochs):
        new_token_lp = compute_token_log_probs(model, generated_ids, attention_mask)

        ratio = torch.exp(new_token_lp - old_token_lp.detach())
        tok_adv = advantages.unsqueeze(1) * response_mask

        surr1 = ratio * tok_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * tok_adv
        policy_loss = -torch.min(surr1, surr2).sum() / response_mask.sum().clamp(min=1)

        # KL(pi_theta || pi_ref) — unbiased estimator (DeepSeek GRPO)
        r_ref = torch.exp(ref_token_lp - new_token_lp)
        kl_per_tok = r_ref - torch.log(r_ref) - 1
        kl_loss = (kl_per_tok * response_mask).sum() / response_mask.sum().clamp(min=1)

        loss = policy_loss + kl_coeff * kl_loss

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        optimizer.step()

        total_policy += policy_loss.item()
        total_kl += kl_loss.item()

    n = max(num_ppo_epochs, 1)
    return {
        "loss": (total_policy + kl_coeff * total_kl) / n,
        "policy_loss": total_policy / n,
        "kl_loss": total_kl / n,
        "mean_advantage": advantages.mean().item(),
        "mean_group_std": g_std.mean().item(),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    env: GraphEnvironment,
    num_pairs: int,
    max_new_tokens: int,
    device: torch.device,
    rng: random.Random,
    batch_size: int = 64,
) -> dict:
    """Evaluate on random (u,v) pairs with greedy decoding."""
    model.eval()

    pairs = env.sample_pairs(num_pairs, rng)
    all_rewards: list[torch.Tensor] = []
    total_valid = 0
    total_gen_len = 0.0

    for start in range(0, len(pairs), batch_size):
        bp = pairs[start : start + batch_size]
        prompts = build_prompt_strings(bp)

        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        plen = enc.input_ids.shape[1]
        gen_ids = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        tokenizer.padding_side = orig_side

        rew, info = parse_and_verify(gen_ids, plen, bp, tokenizer, env)
        all_rewards.append(rew)
        total_valid += info["num_valid"]
        total_gen_len += info["avg_gen_length"] * len(bp)

    model.train()
    cat = torch.cat(all_rewards)
    return {
        "eval/accuracy": cat.mean().item(),
        "eval/num_valid": total_valid,
        "eval/avg_gen_length": total_gen_len / max(len(pairs), 1),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, step: int, model, optimizer, scheduler, cfg):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def load_postrain_checkpoint(path: str, model, optimizer, scheduler):
    """Resume from a previous post-training checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pc = cfg.postrain

    random.seed(pc.seed)
    torch.manual_seed(pc.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if "bf16" in pc.precision else torch.float32

    # ---- tokenizer & graph --------------------------------------------------
    data_dir = cfg.data.output_dir
    tokenizer = load_tokenizer(os.path.join(data_dir, "tokenizer"))
    env = GraphEnvironment(os.path.join(data_dir, "graph.json"))
    print(f"Graph: {len(env.nodes)} nodes, directed={env.directed}")
    if pc.sampling_algorithm == "teacher_stationary":
        teacher_model = TeacherModel(pc.checkpoint_path, tokenizer, device, dtype) # initialize teacher model from the same checkpoint as before!
        teacher_env = TeacherEnvironment(teacher_model)

    # ---- model from supervised checkpoint -----------------------------------
    print(f"Loading checkpoint: {pc.checkpoint_path}")
    lit_model = GraphTrajectoryLM.load_from_checkpoint(
        pc.checkpoint_path, map_location="cpu", weights_only=False
    )
    model = lit_model.model.to(device=device, dtype=dtype)
    model.train()

    ref_model = None
    if pc.algorithm == "grpo":
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # ---- optimiser / scheduler ----------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=pc.lr, weight_decay=pc.weight_decay
    )
    warmup = pc.warmup_steps

    def _lr_lambda(step):
        return float(step) / float(max(1, warmup)) if step < warmup else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    start_step = 0
    if pc.get("resume_from"):
        start_step = load_postrain_checkpoint(
            pc.resume_from, model, optimizer, scheduler
        )
        print(f"Resumed from step {start_step}")

    # ---- checkpoint dir -----------------------------------------------------
    ckpt_dir = Path(pc.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- W&B ----------------------------------------------------------------
    wandb.init(
        project=pc.wandb_project,
        name=pc.wandb_run_name or None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    rng = random.Random(pc.seed)
    eval_rng = random.Random(pc.seed + 1)

    # ---- training loop ------------------------------------------------------
    sampling_algo = pc.get("sampling_algorithm", "uniform_pairs") # or "uniform_connected" or "uniform_specified"
    print(
        f"Starting RLVR post-training — algorithm={pc.algorithm.upper()}, "
        f"sampling={sampling_algo}"
    )

    for step in tqdm(range(start_step + 1, pc.max_steps + 1), desc="Post-train"):
        model.train()

        # 1) sample (u, v) pairs
        sampling = pc.get("sampling_algorithm", "uniform_pairs")
        if sampling == "uniform_pairs":
            pairs = env.sample_pairs(pc.num_pairs_per_step, rng)
        elif sampling == "uniform_specified":
            pairs = env.sample_specified_pairs(
                pc.num_pairs_per_step, pc.n_max, rng
            )
        elif sampling == "uniform_connected":
            pairs = env.sample_pairs_connected(pc.num_pairs_per_step, rng)
        elif sampling == "teacher_stationary":
            pairs = teacher_env.sample_pairs(pc.num_pairs_per_step, rng)
            
        else:
            raise ValueError(f"Unknown sampling_algorithm: {sampling}")

        # drop None entries (failed decodes from teacher sampling)
        pairs = [p for p in pairs if p is not None]
        if len(pairs) == 0:
            continue

        # for GRPO, repeat each pair group_size times
        if pc.algorithm == "grpo":
            gen_pairs = [p for p in pairs for _ in range(pc.group_size)]
        else:
            gen_pairs = pairs

        # 2) generate rollouts
        generated_ids, prompt_len = generate_rollouts(
            model=model,
            tokenizer=tokenizer,
            prompt_strings=build_prompt_strings(gen_pairs),
            max_new_tokens=pc.max_new_tokens,
            temperature=pc.temperature,
            top_k=pc.get("top_k", 0),
            top_p=pc.get("top_p", 1.0),
            device=device,
        )

        # 3) verify & reward
        rewards, gen_info = parse_and_verify(
            generated_ids,
            prompt_len,
            gen_pairs,
            tokenizer,
            env,
            correct_reward=pc.correct_reward,
            incorrect_reward=pc.incorrect_reward,
        )

        # 4) masks
        attn_mask = (generated_ids != tokenizer.pad_token_id).long()
        resp_mask = get_response_mask(generated_ids, prompt_len, tokenizer.pad_token_id)

        # 5) policy-gradient update
        if pc.algorithm == "reinforce":
            step_info = reinforce_step(
                model,
                optimizer,
                generated_ids,
                attn_mask,
                rewards,
                resp_mask,
                baseline=pc.baseline,
                gradient_clip_val=pc.gradient_clip_val,
            )
        elif pc.algorithm == "grpo":
            step_info = grpo_step(
                model,
                ref_model,
                optimizer,
                generated_ids,
                attn_mask,
                rewards,
                resp_mask,
                group_size=pc.group_size,
                clip_range=pc.clip_range,
                kl_coeff=pc.kl_coeff,
                num_ppo_epochs=pc.get("num_ppo_epochs", 1),
                gradient_clip_val=pc.gradient_clip_val,
            )
        else:
            raise ValueError(f"Unknown algorithm: {pc.algorithm}")

        scheduler.step()

        # 6) logging
        if step % pc.log_every_steps == 0:
            log_data = {
                "step": step,
                "train/loss": step_info["loss"],
                "train/accuracy": gen_info["accuracy"],
                "train/avg_gen_length": gen_info["avg_gen_length"],
                "train/mean_reward": rewards.mean().item(),
                "train/lr": scheduler.get_last_lr()[0],
            }
            for k in ("policy_loss", "kl_loss", "mean_advantage", "mean_group_std"):
                if k in step_info:
                    log_data[f"train/{k}"] = step_info[k]
            wandb.log(log_data, step=step)
            tqdm.write(
                f"[Step {step}] loss={step_info['loss']:.4f} "
                f"acc={gen_info['accuracy']:.3f} "
                f"reward={rewards.mean():.3f}"
            )

        # 7) evaluation
        if step % pc.eval_every_steps == 0:
            eval_info = evaluate(
                model,
                tokenizer,
                env,
                num_pairs=pc.eval_num_pairs,
                max_new_tokens=pc.max_new_tokens,
                device=device,
                rng=eval_rng,
            )
            wandb.log(eval_info, step=step)
            tqdm.write(
                f"[Eval @ {step}] accuracy={eval_info['eval/accuracy']:.3f}"
            )

        # 8) checkpoint
        if step % pc.save_every_steps == 0:
            p = ckpt_dir / f"postrain_step_{step}.pt"
            save_checkpoint(p, step, model, optimizer, scheduler, cfg)
            tqdm.write(f"Checkpoint → {p}")

    # ---- final save ---------------------------------------------------------
    final = ckpt_dir / "postrain_final.pt"
    save_checkpoint(final, pc.max_steps, model, optimizer, scheduler, cfg)
    print(f"Final checkpoint → {final}")

    wandb.finish()


if __name__ == "__main__":
    main()
