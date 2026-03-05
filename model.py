"""Lightning Module wrapping a randomly-initialized Qwen2ForCausalLM."""

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import Qwen2Config, Qwen2ForCausalLM
from omegaconf import DictConfig


class GraphTrajectoryLM(pl.LightningModule):
    """Lightning Module for next-token prediction on graph trajectories.

    Uses a randomly-initialized Qwen2ForCausalLM with a custom vocabulary size
    determined by the graph nodes + special tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
        model_config: DictConfig,
        train_config: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build Qwen2Config with custom vocab_size
        config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=model_config.hidden_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            intermediate_size=model_config.intermediate_size,
            hidden_act=model_config.hidden_act,
            max_position_embeddings=model_config.max_position_embeddings,
            rms_norm_eps=model_config.rms_norm_eps,
            rope_theta=model_config.rope_theta,
            attention_dropout=model_config.attention_dropout,
            initializer_range=model_config.initializer_range,
            tie_word_embeddings=model_config.tie_word_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=False,
        )

        # Randomly initialize the model (no pretrained weights)
        self.model = Qwen2ForCausalLM(config)

        self.train_config = train_config

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("val/loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("test/loss", outputs.loss, on_step=False, on_epoch=True, sync_dist=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )

        warmup_steps = self.train_config.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
