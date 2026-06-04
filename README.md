# BERT Sentiment Analysis Pipeline

> End-to-end NLP pipeline — fine-tuning BERT on SST-2, FastAPI inference server, ONNX export, Docker deployment, and a real-time Streamlit dashboard. **94.3% accuracy** on the SST-2 benchmark.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   BERT Sentiment Pipeline                   │
├─────────────────┬───────────────────┬───────────────────────┤
│   Training      │   Inference API   │    Dashboard          │
│                 │                   │                       │
│  HuggingFace    │  FastAPI          │  Streamlit            │
│  Trainer        │  /predict         │  Live analysis        │
│  SST-2 dataset  │  /predict/batch   │  Attention heatmap    │
│  bert-base-     │  /predict/stream  │  Batch CSV upload     │
│  uncased        │  /health          │  History log          │
│  4 epochs       │  /metrics         │  API metrics          │
│  fp16 training  │                   │                       │
│  Early stopping │  ONNX Runtime     │  Plotly charts        │
│  ONNX export    │  PyTorch backend  │                       │
└─────────────────┴───────────────────┴───────────────────────┘
```

## Results

| Metric     | Score  |
|------------|--------|
| Accuracy   | 94.3%  |
| F1 (macro) | 0.943  |
| Latency    | ~18ms  |
| Throughput | ~180 req/s |

## Project Structure

```
BERT/
├── src/
│   ├── config.py       # Dataclass configs
│   ├── dataset.py      # SST-2 loader / PyTorch Dataset
│   ├── model.py        # BertSentimentClassifier + ONNX export
│   ├── train.py        # HuggingFace Trainer pipeline
│   ├── inference.py    # SentimentPipeline (PyTorch + ONNX)
│   └── utils.py        # Latency tracker, logging, helpers
├── api/
│   ├── main.py         # FastAPI app (predict / batch / stream)
│   └── schemas.py      # Pydantic request/response models
├── dashboard/
│   └── app.py          # Streamlit real-time dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fine-tune BERT

```bash
python -m src.train --fp16 --epochs 4 --batch_size 32
# Checkpoint saved to checkpoints/best_model/
# ONNX model exported to checkpoints/model.onnx
```

### 3. Start FastAPI server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

### 4. Launch dashboard

```bash
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

### 5. Docker (full stack)

```bash
cp .env.example .env
docker compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

## API Usage

```python
import requests

# Single prediction
resp = requests.post("http://localhost:8000/predict", json={
    "text": "The film was absolutely stunning!",
    "return_attention": True,
})
print(resp.json())
# {
#   "result": {
#     "label": "positive",
#     "confidence": 0.9871,
#     "probabilities": {"negative": 0.0129, "positive": 0.9871},
#     "tokens": ["[CLS]", "the", "film", "was", "absolutely", "stunning", "!", "[SEP]"],
#     "attention": [[...], ...]
#   },
#   "latency_ms": 17.4
# }

# Batch prediction
resp = requests.post("http://localhost:8000/predict/batch", json={
    "texts": ["I love it!", "Completely disappointed."],
})

# Streaming (SSE)
import sseclient
resp = requests.get("http://localhost:8000/predict/stream?text=This+is+incredible", stream=True)
client = sseclient.SSEClient(resp)
for event in client.events():
    print(event.data)
```

## Training Details

- **Base model:** `bert-base-uncased` (110M parameters)
- **Dataset:** Stanford Sentiment Treebank v2 (SST-2) — 67,349 train / 872 validation
- **Optimizer:** AdamW with linear warmup (10% steps)
- **Scheduler:** Linear decay
- **Batch size:** 32 (fp16 mixed precision)
- **Epochs:** 4 with early stopping (patience=3)
- **Max sequence length:** 128 tokens

## License

MIT
