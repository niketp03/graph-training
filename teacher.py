"""Teacher model for task (node-pair) sampling and optional policy-gradient updates.

Default mode is a *fixed* teacher whose weights never change.  Tasks are
sampled by feeding the ``<start_goal>`` token and generating two tokens
autoregressively — these two tokens are the (start, end) node pair.

If ``trainable=True``, the teacher exposes a ``gradient_step`` method that
accepts (pairs, rewards) and performs a REINFORCE update on the task-generation
policy, analogous to the student loop in ``postrain.py``.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from model import GraphTrajectoryLM


class TeacherModel:
    """Wraps a checkpoint-loaded causal LM for task sampling and optional RL."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trainable: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        gradient_clip_val: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.trainable = trainable
        self.gradient_clip_val = gradient_clip_val
        self._step = 0

        lit_model = GraphTrajectoryLM.load_from_checkpoint(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.model = lit_model.model.to(device=self.device, dtype=self.dtype)

        if trainable:
            self.model.train()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay,
            )
            self._warmup_steps = warmup_steps

            def _lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, _lr_lambda,
            )
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.optimizer = None
            self.scheduler = None
            self.ref_model = None

    # ------------------------------------------------------------------
    # Task sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_tasks(
        self,
        num_tasks: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[tuple[int, int]]:
        """Sample ``num_tasks`` (start, end) node pairs from the teacher.

        Feeds ``<start_goal>`` as the prompt and generates exactly 2 tokens.
        Each generated pair of tokens is decoded and parsed into integer node
        IDs.
        """
        prompt = "<start_goal>"
        prompt_strings = [prompt] * num_tasks

        orig_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            prompt_strings, return_tensors="pt", padding=True,
        ).to(self.device)
        self.tokenizer.padding_side = orig_side

        gen_kwargs: dict = dict(
            max_new_tokens=2,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = self.model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            **gen_kwargs,
        )

        prompt_len = enc.input_ids.shape[1]
        pairs: list[tuple[int, int]] = []
        for i in range(num_tasks):
            new_tokens = generated_ids[i, prompt_len:].tolist()
            try:
                node_strs = [
                    self.tokenizer.decode([tid]).strip() for tid in new_tokens
                ]
                if len(node_strs) >= 2:
                    pairs.append((int(node_strs[0]), int(node_strs[1])))
                else:
                    pairs.append(None)
            except (ValueError, KeyError):
                pairs.append(None)

        return pairs

    # ------------------------------------------------------------------
    # Policy-gradient update
    # ------------------------------------------------------------------

    def _build_task_prompts(
        self, pairs: list[tuple[int, int]]
    ) -> list[str]:
        """Build prompt strings for scoring generated tasks."""
        return ["<start_goal>" for _ in pairs]

    def _build_task_sequences(
        self, pairs: list[tuple[int, int]]
    ) -> list[str]:
        """Full sequences: prompt + the two generated node tokens."""
        return [f"<start_goal> {u} {v}" for u, v in pairs]

    def _compute_token_log_probs(
        self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token log probs, shape [B, T-1]."""
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        return F.log_softmax(shift_logits, dim=-1).gather(
            -1, shift_targets.unsqueeze(-1),
        ).squeeze(-1)

    def _get_response_mask(
        self, input_ids: torch.Tensor, prompt_len: int,
    ) -> torch.Tensor:
        """Mask over response tokens in the shifted (T-1) space."""
        B, T = input_ids.shape
        pad_id = self.tokenizer.pad_token_id
        targets = input_ids[:, 1:]
        mask = torch.zeros(B, T - 1, device=input_ids.device)
        if prompt_len - 1 < T - 1:
            mask[:, prompt_len - 1:] = (
                targets[:, prompt_len - 1:] != pad_id
            ).float()
        return mask

    def gradient_step(
        self,
        pairs: list[tuple[int, int]],
        rewards: torch.Tensor,
        baseline: str = "mean",
    ) -> dict:
        """REINFORCE policy-gradient step on the task-generation policy.

        Args:
            pairs: The (start, end) node pairs that were sampled.
            rewards: Tensor of shape ``[B]`` with a scalar reward per pair.
            baseline: ``"mean"`` to subtract the batch mean, or ``"none"``.

        Returns:
            Dict with ``loss`` and ``mean_advantage``.
        """
        if not self.trainable:
            raise RuntimeError(
                "gradient_step called on a fixed (non-trainable) teacher. "
                "Instantiate with trainable=True to enable updates."
            )

        sequences = self._build_task_sequences(pairs)

        orig_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            sequences, return_tensors="pt", padding=True,
        ).to(self.device)
        self.tokenizer.padding_side = orig_side

        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        prompt_strs = self._build_task_prompts(pairs)
        self.tokenizer.padding_side = "left"
        prompt_enc = self.tokenizer(
            prompt_strs, return_tensors="pt", padding=True,
        )
        self.tokenizer.padding_side = orig_side
        prompt_len = prompt_enc.input_ids.shape[1]

        response_mask = self._get_response_mask(input_ids, prompt_len)

        rewards = rewards.to(self.device)

        token_log_probs = self._compute_token_log_probs(
            self.model, input_ids, attention_mask,
        )

        if baseline == "mean":
            advantages = rewards - rewards.mean()
        else:
            advantages = rewards

        token_advantages = advantages.unsqueeze(1) * response_mask
        loss = -(token_log_probs * token_advantages).sum() / response_mask.sum().clamp(min=1)

        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val,
            )
        self.optimizer.step()
        self.scheduler.step()
        self._step += 1

        return {
            "loss": loss.item(),
            "mean_advantage": advantages.mean().item(),
            "lr": self.scheduler.get_last_lr()[0],
        }
