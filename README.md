# SupportLLM — Fine-tuned Customer-Support Intent Classifier

An end-to-end pipeline that **LoRA-fine-tunes a small instruct LLM for customer-support
intent classification** and **serves it behind a production-shaped FastAPI app** — with a
real vLLM engine and a custom continuous-batching fallback, MLflow experiment tracking,
Optuna hyperparameter search, containerization, and CI.

The point of this project is the **full fine-tune-to-serve lifecycle and the serving
engineering**, demonstrated end-to-end on consumer hardware. See
[Results](#results) and [Limitations](#limitations--productionizing) for an honest account
of the training scale.

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Highlights

- **LoRA / QLoRA fine-tuning** of `Qwen/Qwen2.5-0.5B-Instruct` with PEFT — 4-bit QLoRA on
  CUDA, plain LoRA on Apple Silicon (MPS) / CPU, selected automatically at runtime.
- **Pluggable serving layer** — a `BaseServingEngine` abstraction with a real
  `vLLM` implementation and a Hugging Face fallback that implements its **own async
  continuous-batching queue** to approximate vLLM behavior on CPU/MPS.
- **Streaming FastAPI service** with classification, response generation, and a deliberately
  naive sequential endpoint for A/B serving benchmarks.
- **Reproducible MLOps** — Optuna sweep → MLflow tracking → final train → eval → serve →
  benchmark, all runnable with a single script.
- **Shipped like software** — Docker + docker-compose, GitHub Actions CI (ruff + pytest).

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
   Bitext dataset  ──▶  │  dataset.py  (stratified split, chatml prompt)│
   (27 intents)         └───────────────────────┬─────────────────────┘
                                                 │
                      ┌──────────────────────────▼──────────────────────────┐
                      │  sweep.py (Optuna, 20 trials) ─▶ configs/best.yaml    │
                      │  train.py (PEFT LoRA/QLoRA)   ─▶ model_assets/adapter │
                      │            └─ params + metrics ─▶ MLflow              │
                      └──────────────────────────┬──────────────────────────┘
                                                 │
                      ┌──────────────────────────▼──────────────────────────┐
                      │  eval.py  baseline vs fine-tuned                      │
                      │           ─▶ results/metrics.json, confusion_matrix   │
                      └──────────────────────────┬──────────────────────────┘
                                                 │
     FastAPI (src/api.py)                        │
     ┌───────────────────────────────────────────▼───────────────────────┐
     │  get_serving_engine()                                              │
     │     ├─ VLLMServingEngine   (AsyncLLMEngine + LoRARequest)          │
     │     └─ HFServingEngine      (custom 16-req / 20 ms batching queue) │
     │  endpoints: /health  /classify  /naive_classify  /generate        │
     └───────────────────────────────────────────────────────────────────┘
```

The serving layer is the load-bearing piece. `get_serving_engine()` tries to construct a
vLLM engine and transparently falls back to the Hugging Face engine when vLLM or CUDA is
unavailable. The fallback isn't just sequential `model.generate` — it runs a background
`asyncio` loop that accumulates incoming requests into batches of up to **16** within a
**20 ms** window and executes them off the event loop in a thread pool, so the CPU/MPS path
still benefits from batched throughput.

---

## Repository layout

```
.
├── configs/
│   └── train.yaml              # model + hyperparameters (best.yaml written by the sweep)
├── src/
│   ├── api.py                  # FastAPI app: /health /classify /naive_classify /generate
│   ├── dataset.py              # load + stratified split, chatml prompt formatting
│   └── serve_vllm.py           # BaseServingEngine, VLLMServingEngine, HFServingEngine
├── scripts/
│   ├── download_data.py        # pre-cache dataset + base model
│   ├── train.py                # LoRA/QLoRA training + MLflow logging
│   ├── sweep.py                # Optuna hyperparameter search
│   ├── eval.py                 # baseline vs fine-tuned, metrics + confusion matrix
│   └── benchmark_serving.py    # throughput / latency benchmark (batched vs naive)
├── model_assets/adapter/       # saved LoRA adapter
├── results/                    # metrics.json, dataset_stats.json, plots, benchmark csv
├── tests/                      # pytest: API + prompt/label utilities
├── .github/workflows/ci.yml    # ruff lint + pytest
├── Dockerfile
├── docker-compose.yml          # MLflow server + API service
├── requirements.txt
└── run_all.sh                  # one-command end-to-end pipeline
```

---

## Quickstart

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> `vllm` and `bitsandbytes` are included for the GPU path. On macOS/CPU the service
> automatically falls back to the Hugging Face engine, so a missing vLLM install is fine.

### 2. Run the whole pipeline

```bash
bash run_all.sh
```

This downloads the data and base model, runs the Optuna sweep, trains the final adapter,
evaluates baseline vs fine-tuned, boots the API, and runs the serving benchmark. All
artifacts land in `results/`.

### 3. Or run the steps individually

```bash
python scripts/download_data.py                 # cache dataset + base model
python scripts/sweep.py                          # Optuna → configs/best.yaml
python scripts/train.py --config configs/best.yaml
python scripts/eval.py                            # → results/metrics.json
python src/api.py                                 # serve on :8000
python scripts/benchmark_serving.py --num_requests 100 --concurrency 5
```

### 4. Docker

```bash
docker compose up --build
# API   → http://localhost:8000
# MLflow → http://localhost:5000
```

---

## API reference

Base URL: `http://localhost:8000`

| Method | Endpoint          | Purpose                                                        |
|--------|-------------------|----------------------------------------------------------------|
| GET    | `/health`         | Engine readiness + engine type                                 |
| POST   | `/classify`       | Classify a query into one intent (batched engine path)         |
| POST   | `/naive_classify` | Same, forced sequential (batch=1) — the benchmark baseline     |
| POST   | `/generate`       | Generate a support response conditioned on an intent           |

**Classify**

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to cancel my order", "max_tokens": 15}'
# → {"intent": "cancel_order"}
```

Request fields: `query` (required), `max_tokens` (default 15), `temperature` (default 0.0),
`stream` (default false). With `stream: true` the response is streamed as `text/plain`
chunks. Raw output is normalized against the known intent list before it is returned.

**Generate**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is my package?", "intent": "track_order", "max_tokens": 100}'
# → {"response": "..."}
```

Request fields: `query`, `intent` (both required), `max_tokens` (default 100),
`temperature` (default 0.7), `stream` (default false).

---

## Data & training

- **Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` — 26,872 raw
  examples across **27 intents**, split stratified on the intent label.
- **Prompt:** Qwen chatml. The classifier prompt enumerates every intent and constrains the
  model to emit exactly one intent name. During SFT, prompt tokens are masked with `-100` so
  the loss is computed only on the target label.
- **Adapter:** LoRA `r=8`, `alpha=16`, `dropout=0.05`, applied to
  `q/k/v/o_proj` and `gate/up/down_proj`. On CUDA, the base model is loaded 4-bit
  (`nf4`, double-quant, bf16 compute) for QLoRA.
- **Tracking:** parameters and metrics are logged to MLflow; the adapter is logged as a run
  artifact.

---

## Results

Baseline (zero-shot base model) vs LoRA fine-tuned, evaluated on a held-out test sample:

| Metric                     | Baseline | Fine-tuned |
|----------------------------|:--------:|:----------:|
| Accuracy                   |   0.23   |  **0.46**  |
| Macro-F1                   |   0.21   |  **0.52**  |
| Weighted-F1                |   0.24   |  **0.49**  |
| ROUGE-L (response gen)     |   0.183  |    0.174   |

Fine-tuning roughly **doubled** classification accuracy and macro-F1. Response-generation
ROUGE-L is essentially flat — the demonstrated win is intent classification, not free-text
answer quality.

**Serving benchmark** (batched vs naive sequential, local run):

| Metric              | Continuous batching | Naive sequential | Speedup |
|---------------------|:-------------------:|:----------------:|:-------:|
| Throughput (req/s)  |        3.88         |       3.44       |  1.13x  |
| p50 latency (s)     |        0.71         |       0.86       |    —    |

---

## Configuration

`configs/train.yaml` holds the model id and hyperparameters (learning rate, LoRA rank/alpha/
dropout, batch sizes, sequence length, seed, and MLflow reporting). `scripts/sweep.py`
searches over the LoRA and learning-rate space with Optuna and writes the winning
combination to `configs/best.yaml`, which `train.py`, `eval.py`, and `api.py` prefer when
present.

---

## Testing & CI

```bash
ruff check .
pytest tests/
```

CI runs on every push and PR to `main` (GitHub Actions): ruff lint + pytest. The API tests
mock the serving engine so the suite stays fast and dependency-light — no model weights are
downloaded in CI.

---

## Limitations & productionizing

This repository is a **lifecycle and serving demonstration trained at proof-of-concept scale
on a laptop (Apple Silicon / MPS)**, not a converged production model. Being explicit about
that:

- The final training run is short (`max_steps` on the order of tens) and the training set is
  capped well below the full dataset, so metrics reflect a smoke-test amount of training.
- The evaluation uses a small test sample across 27 classes, so per-intent scores are noisy;
  the aggregate baseline-vs-fine-tuned improvement is the defensible signal.
- The serving benchmark ran on the CPU/MPS fallback path, so the 1.13x figure reflects the
  custom batching queue vs sequential generation — not a true GPU vLLM comparison.

To take this to production the natural next steps are: train on the full dataset for a real
number of steps, run on a GPU with genuine vLLM continuous batching, evaluate on the full
held-out test set with confidence intervals, and complete the adapter model card.

---

## License

MIT — see [LICENSE](LICENSE).
