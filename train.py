"""Training entrypoint for graph trajectory next-token prediction."""

import json
import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data import TrajectoryDataModule
from model import GraphTrajectoryLM
from tokenizer_utils import build_tokenizer, save_tokenizer, load_tokenizer


class MetadataCallback(pl.Callback):
    """Embeds samples_seen into each checkpoint and writes metadata.json on train end."""

    def __init__(self, run_dir: Path, cfg: DictConfig):
        self.run_dir = run_dir
        self.cfg = cfg

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["samples_seen"] = (
            trainer.global_step
            * self.cfg.train.batch_size
            * self.cfg.train.gradient_accumulation
        )
        checkpoint["max_path_length"] = self.cfg.data.max_path_length

    def on_train_end(self, trainer, pl_module):
        metadata = {
            "max_path_length": self.cfg.data.max_path_length,
            "samples_seen": (
                trainer.global_step
                * self.cfg.train.batch_size
                * self.cfg.train.gradient_accumulation
            ),
            "global_step": trainer.global_step,
            "batch_size": self.cfg.train.batch_size,
            "gradient_accumulation": self.cfg.train.gradient_accumulation,
        }
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata written to {self.run_dir / 'metadata.json'}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entrypoint."""
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.data.seed, workers=True)

    # Paths
    data_dir = cfg.data.output_dir
    vocab_path = os.path.join(data_dir, "vocabulary.json")
    tokenizer_dir = os.path.join(data_dir, "tokenizer")

    # Build or load tokenizer to get vocab_size and special token IDs
    tokenizer_path = Path(tokenizer_dir)
    if tokenizer_path.exists() and (tokenizer_path / "tokenizer.json").exists():
        tokenizer = load_tokenizer(tokenizer_dir)
    else:
        tokenizer = build_tokenizer(vocab_path)
        save_tokenizer(tokenizer, tokenizer_dir)

    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    print(f"Vocabulary size: {vocab_size}")
    print(f"Pad token ID: {pad_token_id}, EOS token ID: {eos_token_id}")

    # Unique run directory — one per run, under the configured checkpoint base dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.train.checkpoint_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the config that produced this run
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"Run checkpoint dir: {run_dir}")

    # DataModule
    datamodule = TrajectoryDataModule(
        data_dir=data_dir,
        tokenizer_dir=tokenizer_dir,
        vocab_path=vocab_path,
        batch_size=cfg.train.batch_size,
        max_length=cfg.model.max_position_embeddings,
        num_workers=cfg.train.num_workers,
    )

    # Model
    model = GraphTrajectoryLM(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        model_config=cfg.model,
        train_config=cfg.train,
    )

    # Logger
    wandb_logger = WandbLogger(
        project=cfg.train.wandb_project,
        name=cfg.train.wandb_run_name or None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Callbacks — save only the single best checkpoint for this run
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    metadata_cb = MetadataCallback(run_dir, cfg)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.gradient_accumulation,
        gradient_clip_val=cfg.train.gradient_clip_val,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, metadata_cb],
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
