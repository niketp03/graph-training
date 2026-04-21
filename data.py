"""Lightning DataModule for graph trajectory next-token prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from transformers import PreTrainedTokenizerFast

from tokenizer_utils import build_tokenizer, save_tokenizer, load_tokenizer


class TrajectoryDataset(Dataset):
    """Dataset that reads trajectory strings and tokenizes them for causal LM training."""

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> dict:
        line = self.lines[idx]

        encoding = self.tokenizer(
            line,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        return {"input_ids": encoding["input_ids"]}


class TrajectoryDataModule(pl.LightningDataModule):
    """Lightning DataModule for graph trajectory data.

    Reads train/val/test.txt files, tokenizes trajectories, and prepares
    batches for causal next-token prediction.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer_dir: str,
        vocab_path: str,
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_dir = tokenizer_dir
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Build or load tokenizer
        tokenizer_path = Path(self.tokenizer_dir)
        if tokenizer_path.exists() and (tokenizer_path / "tokenizer.json").exists():
            self.tokenizer = load_tokenizer(self.tokenizer_dir)
        else:
            self.tokenizer = build_tokenizer(self.vocab_path)
            save_tokenizer(self.tokenizer, self.tokenizer_dir)

        if stage == "fit" or stage is None:
            self.train_dataset = TrajectoryDataset(
                os.path.join(self.data_dir, "train.txt"),
                self.tokenizer,
                self.max_length,
            )
            self.val_dataset = TrajectoryDataset(
                os.path.join(self.data_dir, "val.txt"),
                self.tokenizer,
                self.max_length,
            )

        if stage == "test" or stage is None:
            self.test_dataset = TrajectoryDataset(
                os.path.join(self.data_dir, "test.txt"),
                self.tokenizer,
                self.max_length,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(self, batch: list[dict]) -> dict:
        """Collate with padding for causal LM training.

        The HuggingFace Qwen2ForCausalLM handles the input/target shifting
        internally when labels are provided, so we pass the same token IDs
        as both input_ids and labels. Padding positions use -100 in labels
        so they are ignored by the cross-entropy loss.
        """
        pad_id = self.tokenizer.pad_token_id
        all_ids = [item["input_ids"] for item in batch]
        max_len = max(len(ids) for ids in all_ids)

        input_ids = []
        labels = []
        attention_masks = []

        for ids in all_ids:
            pad_length = max_len - len(ids)

            attention_mask = [1] * len(ids) + [0] * pad_length
            inp_padded = ids + [pad_id] * pad_length
            lbl_padded = ids + [-100] * pad_length  # -100 = ignore index

            input_ids.append(inp_padded)
            labels.append(lbl_padded)
            attention_masks.append(attention_mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
