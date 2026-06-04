import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
)
from typing import Optional, Dict, Tuple, List
import logging
import os

logger = logging.getLogger(__name__)


class BertSentimentClassifier(nn.Module):
    """BERT fine-tuned for sequence classification with attention export."""

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,  # enabled only at inference time
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=output_attentions,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "attentions": outputs.attentions if output_attentions else None,
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, token_type_ids, output_attentions=return_attention)
        probs = torch.softmax(out["logits"], dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs, out["attentions"]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.bert.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, num_labels: int = 2) -> "BertSentimentClassifier":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.num_labels = num_labels
        obj.bert = BertForSequenceClassification.from_pretrained(
            path, num_labels=num_labels,
            output_attentions=False,
            attn_implementation="eager",  # SDPA silently drops attentions; eager returns them
        )
        logger.info(f"Model loaded from {path}")
        return obj


def get_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizer:
    return BertTokenizer.from_pretrained(model_name)


def compute_attention_rollout(attentions: tuple) -> torch.Tensor:
    """Aggregate attention across all heads/layers via rollout."""
    rollout = torch.eye(attentions[0].shape[-1])
    for attn in attentions:
        attn_avg = attn.mean(dim=1)  # avg over heads
        rollout = torch.bmm(
            (attn_avg + torch.eye(attn_avg.shape[-1]).unsqueeze(0)).softmax(dim=-1),
            rollout.unsqueeze(0).expand(attn_avg.shape[0], -1, -1),
        )
    return rollout  # (batch, seq, seq)


def export_onnx(model: BertSentimentClassifier, path: str, max_length: int = 128, opset: int = 17) -> None:
    """Export model to ONNX for fast CPU/GPU inference."""
    import torch.onnx

    model.eval()
    dummy_ids = torch.zeros(1, max_length, dtype=torch.long)
    dummy_mask = torch.ones(1, max_length, dtype=torch.long)
    dummy_types = torch.zeros(1, max_length, dtype=torch.long)

    torch.onnx.export(
        model.bert,
        (dummy_ids, dummy_mask, dummy_types),
        path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )
    logger.info(f"ONNX model exported to {path}")
