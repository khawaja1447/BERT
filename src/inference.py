"""
Inference engine supporting both PyTorch and ONNX Runtime backends.

Usage:
    from src.inference import SentimentPipeline
    pipeline = SentimentPipeline.from_pretrained("checkpoints/best_model")
    results = pipeline.predict(["I love this!", "Terrible experience."])
"""

import logging
import time
from typing import List, Dict, Optional, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """High-level inference pipeline with attention extraction."""

    LABELS = {0: "negative", 1: "positive"}
    EMOJI = {0: "😞", 1: "😊"}

    def __init__(
        self,
        model_path: str,
        use_onnx: bool = False,
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        from src.model import BertSentimentClassifier, get_tokenizer

        self.max_length = max_length
        self.use_onnx = use_onnx

        self.tokenizer = get_tokenizer(model_path)

        if use_onnx:
            self._load_onnx(model_path)
        else:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            self.model = BertSentimentClassifier.load(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"PyTorch model loaded on {self.device}")

    def _load_onnx(self, model_path: str) -> None:
        import onnxruntime as ort

        onnx_file = model_path.replace("best_model", "model.onnx")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_file, providers=providers)
        self.device = "onnx"
        logger.info(f"ONNX Runtime session loaded from {onnx_file}")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "SentimentPipeline":
        return cls(model_path, **kwargs)

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def predict(
        self, texts: Union[str, List[str]], return_attention: bool = False
    ) -> List[Dict]:
        if isinstance(texts, str):
            texts = [texts]

        t0 = time.perf_counter()
        encodings = self._tokenize(texts)

        if self.use_onnx:
            results = self._predict_onnx(texts, encodings)
        else:
            results = self._predict_torch(texts, encodings, return_attention)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"Inference: {len(texts)} samples in {elapsed:.1f}ms")
        return results

    def _predict_torch(
        self, texts: List[str], encodings: Dict, return_attention: bool
    ) -> List[Dict]:
        ids = encodings["input_ids"].to(self.device)
        mask = encodings["attention_mask"].to(self.device)
        types = encodings.get("token_type_ids", torch.zeros_like(ids)).to(self.device)

        preds, probs, attentions = self.model.predict(
            ids, mask, types, return_attention=return_attention
        )

        results = []
        for i, text in enumerate(texts):
            label_id = preds[i].item()
            prob_arr = probs[i].cpu().tolist()
            result = {
                "text": text,
                "label": self.LABELS[label_id],
                "emoji": self.EMOJI[label_id],
                "confidence": round(prob_arr[label_id], 4),
                "probabilities": {self.LABELS[j]: round(p, 4) for j, p in enumerate(prob_arr)},
            }
            if return_attention and attentions is not None:
                last_layer_attn = attentions[-1][i].mean(dim=0).cpu().tolist()
                tokens = self.tokenizer.convert_ids_to_tokens(ids[i].cpu().tolist())
                result["tokens"] = [t for t in tokens if t != "[PAD]"]
                result["attention"] = last_layer_attn[: len(result["tokens"])]
            results.append(result)
        return results

    def _predict_onnx(self, texts: List[str], encodings: Dict) -> List[Dict]:
        ort_inputs = {
            "input_ids": encodings["input_ids"].numpy().astype(np.int64),
            "attention_mask": encodings["attention_mask"].numpy().astype(np.int64),
            "token_type_ids": encodings.get(
                "token_type_ids", torch.zeros_like(encodings["input_ids"])
            ).numpy().astype(np.int64),
        }
        logits = self.session.run(["logits"], ort_inputs)[0]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        preds = probs.argmax(axis=-1)

        return [
            {
                "text": text,
                "label": self.LABELS[preds[i]],
                "emoji": self.EMOJI[preds[i]],
                "confidence": round(float(probs[i][preds[i]]), 4),
                "probabilities": {
                    self.LABELS[j]: round(float(p), 4) for j, p in enumerate(probs[i])
                },
            }
            for i, text in enumerate(texts)
        ]

    def predict_stream(self, text: str):
        """Yield token-level analysis as a generator (for SSE streaming)."""
        words = text.split()
        accumulated = ""
        for word in words:
            accumulated = (accumulated + " " + word).strip()
            result = self.predict(accumulated)[0]
            yield result
