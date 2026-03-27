# AI Reasoning Box

**Multi-Model Ensemble Reasoning Platform** — Query multiple AI models simultaneously, score their responses, synthesize the best answer, and train your own model on the results.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Live Demo

**https://ai-reasoning-box.onrender.com**

> Users bring their own API keys — entered in the browser, stored only in localStorage, never on the server.

## How It Works

```
 Your Question
      |
      v
 +----+----+----+----+----+
 | Claude  | GPT-4o | Gemini | o3-mini | Haiku |   <-- Parallel query
 +----+----+----+----+----+
      |         |        |        |         |
      v         v        v        v         v
 [ Score: accuracy, completeness, reasoning, clarity ]
      |
      v
 [ Synthesize best elements into one answer ]
      |
      v
 [ Collect as training data ]  -->  [ Fine-tune ReasoningBox model ]
      |                                        |
      v                                        v
 Final Answer                    Your own AI model joins the ensemble
```

### The Flywheel

1. **Query** — Send questions to all AI models in parallel
2. **Score** — Each response is evaluated on 4 dimensions
3. **Synthesize** — Best elements combined into a single answer
4. **Collect** — Every high-quality result is saved as training data
5. **Train** — Fine-tune an open-source model on the collected data
6. **Deploy** — Your ReasoningBox model joins the ensemble
7. **Repeat** — The more you use it, the better your model gets

## Features

### Ensemble Reasoning
- Query 6+ AI models simultaneously
- 4-dimensional response scoring (accuracy, completeness, reasoning quality, clarity)
- Consensus and disagreement detection
- 4 synthesis strategies

### ReasoningBox Model Training
- Automatic data collection from every ensemble query
- Quality filtering (confidence >= 0.6, 2+ model responses)
- LoRA fine-tuning pipeline on open-source base models
- Export to HuggingFace Hub or Ollama
- Training progress dashboard in the web UI

## Supported Providers

| Provider | Models | API Key Source |
|----------|--------|----------------|
| **Anthropic** | Claude Sonnet 4.6, Claude Haiku 4.5 | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| **OpenAI** | GPT-4o, o3-mini | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Google** | Gemini 2.5 Pro, Gemini 2.5 Flash | [aistudio.google.com](https://aistudio.google.com/apikey) |
| **DeepSeek** | DeepSeek R1 | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| **ReasoningBox** | Your fine-tuned model | Runs locally or via Ollama |

## Synthesis Strategies

| Strategy | Description |
|----------|-------------|
| **Weighted Merge** | Ranks responses by weighted score, synthesizes best elements |
| **Best of N** | Returns the highest-scored response |
| **Debate** | Models critique each other in rounds, then merge |
| **Chain of Verification** | Weighted merge + verification pass for errors |

## Quick Start

### 1. Run the Web Platform

```bash
git clone https://github.com/aoloo-r/ai_resoning_box.git
cd ai_resoning_box
pip install -r requirements.txt
uvicorn server:app --port 8900
# Open http://localhost:8900 — enter your API keys in the browser
```

### 2. Collect Training Data

Just use the platform! Every query automatically collects training data. The dashboard shows your progress toward the 1,000-pair training threshold.

### 3. Train Your ReasoningBox Model

```bash
# Install training dependencies
pip install -r training/requirements-training.txt

# Prepare the dataset
python training/prepare_data.py

# Fine-tune (default: Qwen2.5-7B with LoRA)
python training/train.py

# Or customize:
python training/train.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --epochs 3 \
  --lora-rank 64 \
  --lr 2e-4
```

### 4. Export & Deploy Your Model

```bash
# Merge LoRA weights into full model
python training/export_model.py

# Convert for Ollama
python training/export_model.py --to-gguf

# Or push to HuggingFace Hub
python training/export_model.py --push-hub your-name/ReasoningBox-7B

# Test it
python training/test_model.py --interactive
```

### 5. Add ReasoningBox to the Ensemble

Enable in `config.yaml`:
```yaml
reasoning_box:
  enabled: true
  models:
    - id: reasoning-box
      name: "ReasoningBox"
      role: reasoning
```

Now your model competes alongside Claude, GPT-4o, and Gemini!

## Deployment

### Docker

```bash
docker build -t ai-reasoning-box .
docker run -p 8900:8900 ai-reasoning-box
```

### Render (One-Click)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aoloo-r/ai_resoning_box)

No environment variables needed. Users provide their own API keys.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/status` | Get active models (with user API keys) |
| `POST` | `/api/query` | Submit an ensemble query |
| `GET` | `/api/strategies` | List synthesis strategies |
| `GET` | `/api/training/stats` | Training data statistics |
| `GET` | `/api/training/recent` | Recent collected queries |

### Query Example

```bash
curl -X POST https://ai-reasoning-box.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain quantum computing",
    "strategy": "weighted_merge",
    "api_keys": {
      "ANTHROPIC_API_KEY": "sk-ant-...",
      "OPENAI_API_KEY": "sk-..."
    }
  }'
```

## Project Structure

```
├── core/
│   ├── models.py                  # Data models and enums
│   ├── orchestrator.py            # Multi-model dispatcher
│   ├── synthesizer.py             # Response scoring and synthesis
│   ├── pipeline.py                # High-level orchestration + data collection
│   ├── collector.py               # Training data collector
│   └── providers/
│       ├── anthropic_provider.py  # Claude
│       ├── openai_provider.py     # GPT-4o, o3
│       ├── google_provider.py     # Gemini
│       ├── deepseek_provider.py   # DeepSeek R1
│       ├── ollama_provider.py     # Local models
│       └── reasoning_box_provider.py  # Your fine-tuned model
├── training/
│   ├── prepare_data.py            # Dataset preparation
│   ├── train.py                   # LoRA fine-tuning
│   ├── export_model.py            # Merge & export weights
│   ├── test_model.py              # Test the fine-tuned model
│   └── requirements-training.txt  # Training dependencies
├── static/index.html              # Web UI with training dashboard
├── server.py                      # FastAPI server
├── main.py                        # CLI interface
├── config.yaml                    # Provider & model config
├── Dockerfile                     # Container deployment
└── render.yaml                    # Render.com config
```

## Security

- API keys stored **only in browser localStorage**
- Keys sent per-request, never logged or persisted server-side
- Fresh provider instances created per request
- No database, no user accounts, no server-side storage
