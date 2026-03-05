"""Tokenizer utilities for building and loading a HuggingFace-compatible tokenizer."""

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast


def build_tokenizer(vocab_path: str) -> PreTrainedTokenizerFast:
    """Build a HuggingFace-compatible WordLevel tokenizer from vocabulary.json.

    Args:
        vocab_path: Path to the vocabulary.json file containing tokens and token_to_id.

    Returns:
        A PreTrainedTokenizerFast instance ready for encoding/decoding.
    """
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    token_to_id = vocab_data["token_to_id"]

    # Create WordLevel tokenizer
    tokenizer_model = WordLevel(vocab=token_to_id, unk_token="<unk>")
    tokenizer = Tokenizer(tokenizer_model)

    # Split only on whitespace, preserving tokens like <start_goal> intact
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Wrap into PreTrainedTokenizerFast for HuggingFace compatibility
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )

    return wrapped_tokenizer


def save_tokenizer(tokenizer: PreTrainedTokenizerFast, save_dir: str) -> None:
    """Save tokenizer in HuggingFace format for later loading."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)


def load_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    """Load a previously saved tokenizer."""
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
