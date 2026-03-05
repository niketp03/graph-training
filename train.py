"""Training entrypoint for graph trajectory next-token prediction."""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data import TrajectoryDataModule
from model import GraphTrajectoryLM
from tokenizer_utils import build_tokenizer, save_tokenizer, load_tokenizer


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

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        filename="epoch-{epoch:02d}-val_loss-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.gradient_accumulation,
        gradient_clip_val=cfg.train.gradient_clip_val,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
