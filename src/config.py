from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    label_names: List[str] = field(default_factory=lambda: ["negative", "positive"])
    max_length: int = 128
    batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 4
    output_dir: str = "checkpoints"
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 400
    fp16: bool = True
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 2
    seed: int = 42

    # dataset
    dataset_name: str = "stanfordnlp/sst2"
    train_split: str = "train"
    val_split: str = "validation"

    # ONNX export
    onnx_path: str = "checkpoints/model.onnx"
    onnx_opset: int = 17


@dataclass
class InferenceConfig:
    model_path: str = "checkpoints/best_model"
    onnx_path: str = "checkpoints/model.onnx"
    use_onnx: bool = False
    max_length: int = 128
    device: str = "auto"
    batch_size: int = 32
