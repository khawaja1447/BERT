import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenized text + labels."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids", torch.zeros(1))[idx]
            if "token_type_ids" in self.encodings
            else torch.zeros(self.encodings["input_ids"].shape[1], dtype=torch.long),
            "labels": self.labels[idx],
        }


def load_sst2(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    train_split: str = "train",
    val_split: str = "validation",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> tuple:
    """Load SST-2 from HuggingFace datasets hub."""
    logger.info("Loading SST-2 dataset from HuggingFace Hub...")
    raw = load_dataset("stanfordnlp/sst2")

    def _extract(split: str, limit: Optional[int]):
        ds = raw[split]
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        texts = [ex["sentence"] for ex in ds]
        labels = [ex["label"] for ex in ds]
        logger.info(f"  {split}: {len(texts)} examples")
        return texts, labels

    train_texts, train_labels = _extract(train_split, max_train_samples)
    val_texts, val_labels = _extract(val_split, max_val_samples)

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
    return train_dataset, val_dataset
