"""Dataset classes for research training with family JSONL files.

Supports loading math problems from family-specific JSONL files and
tokenizing them for causal language model training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset


def load_family_data(
    family_dir: Path,
    families: Optional[List[str]] = None,
    family_type: Optional[str] = None,  # "target" | "control" | None (both)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load family data from JSONL files.

    Args:
        family_dir: Directory containing target/ and control/ subdirs
        families: Optional list of family IDs to load (None = all)
        family_type: Optional filter for "target" or "control" only

    Returns:
        Dict mapping family_id to list of items
    """
    data = {}

    subdirs = []
    if family_type is None or family_type == "target":
        target_dir = family_dir / "target"
        if target_dir.exists():
            subdirs.append(target_dir)
    if family_type is None or family_type == "control":
        control_dir = family_dir / "control"
        if control_dir.exists():
            subdirs.append(control_dir)

    for subdir in subdirs:
        for jsonl_file in subdir.glob("*.jsonl"):
            family_id = jsonl_file.stem
            if families is not None and family_id not in families:
                continue

            items = []
            with jsonl_file.open() as f:
                for line in f:
                    items.append(json.loads(line))
            data[family_id] = items

    return data


def format_problem_for_training(item: Dict[str, Any]) -> str:
    """
    Format a problem item into a training string.

    Uses the problem text and generated solution from OpenMathReasoning.
    """
    problem = item.get("content", {}).get("problem", "")
    solution = item.get("provenance", {}).get("generated_solution", "")

    if solution:
        # Format as problem-solution pair
        return f"Problem: {problem}\n\nSolution: {solution}"
    else:
        return f"Problem: {problem}"


class FamilyDataset(Dataset):
    """Dataset that loads from family JSONL files with tokenization."""

    def __init__(
        self,
        family_dir: Path,
        tokenizer: Any,
        max_seq_len: int = 2048,
        families: Optional[List[str]] = None,
        family_type: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle_seed: int = 42,
    ):
        """
        Args:
            family_dir: Directory containing families/target and families/control
            tokenizer: HuggingFace tokenizer or compatible
            max_seq_len: Maximum sequence length
            families: Optional list of families to include
            family_type: "target" | "control" | None (both)
            max_samples: Maximum total samples to use
            shuffle_seed: Seed for shuffling
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load data
        family_data = load_family_data(family_dir, families, family_type)

        # Flatten into single list with family labels
        self.items = []
        self.family_ids = []
        for family_id, items in family_data.items():
            for item in items:
                self.items.append(item)
                self.family_ids.append(family_id)

        # Shuffle
        rng = random.Random(shuffle_seed)
        indices = list(range(len(self.items)))
        rng.shuffle(indices)
        self.items = [self.items[i] for i in indices]
        self.family_ids = [self.family_ids[i] for i in indices]

        # Limit samples
        if max_samples is not None and max_samples < len(self.items):
            self.items = self.items[:max_samples]
            self.family_ids = self.family_ids[:max_samples]

        print(f"Loaded {len(self.items)} items from {len(family_data)} families")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        family_id = self.family_ids[idx]

        # Format text
        text = format_problem_for_training(item)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "family_id": family_id,
        }


class InfiniteFamilyDataset(IterableDataset):
    """Infinite streaming dataset from family data for training."""

    def __init__(
        self,
        family_dir: Path,
        tokenizer: Any,
        max_seq_len: int = 2048,
        families: Optional[List[str]] = None,
        family_type: Optional[str] = None,
        shuffle_seed: int = 42,
    ):
        self.family_dir = family_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.families = families
        self.family_type = family_type
        self.shuffle_seed = shuffle_seed

        # Load data
        self.family_data = load_family_data(family_dir, families, family_type)
        self.all_items = []
        for family_id, items in self.family_data.items():
            for item in items:
                self.all_items.append((family_id, item))

        if not self.all_items:
            raise ValueError(f"No items found in {family_dir}")

        print(f"InfiniteFamilyDataset: {len(self.all_items)} items from {len(self.family_data)} families")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.shuffle_seed)
        indices = list(range(len(self.all_items)))

        while True:
            rng.shuffle(indices)
            for idx in indices:
                family_id, item = self.all_items[idx]
                text = format_problem_for_training(item)

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids = encoding["input_ids"].squeeze(0)

                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "family_id": family_id,
                }


def get_tokenizer(vocab_size: int = 32000):
    """
    Get a tokenizer for training.

    For research reproducibility, we use a simple byte-level BPE tokenizer
    or fall back to a character-level tokenizer.
    """
    try:
        from transformers import AutoTokenizer

        # Use GPT-2 tokenizer as default (50257 vocab)
        # Can be overridden with a custom tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        # Fallback: simple character tokenizer
        return SimpleCharTokenizer(vocab_size)


class SimpleCharTokenizer:
    """Simple character-level tokenizer as fallback."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 2048,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # Simple character-to-id mapping
        ids = [min(ord(c), self.vocab_size - 1) for c in text]

        # Truncate
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]

        # Pad
        if padding == "max_length":
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len

        input_ids = torch.tensor([ids], dtype=torch.long)

        return {"input_ids": input_ids}

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(i) if 32 <= i < 127 else "?" for i in ids)
