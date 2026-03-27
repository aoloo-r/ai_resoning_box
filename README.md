# AI Reasoning Box

**Multi-Model Ensemble Reasoning Platform** — Query multiple AI models simultaneously, score their responses, and synthesize the best answer.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## How It Works

1. **Parallel Dispatch** — Your question is sent to all configured AI models at once (Claude, GPT-4o, Gemini, etc.)
2. **Scoring** — Each response is evaluated across 4 dimensions: accuracy, completeness, reasoning quality, and clarity
3. **Synthesis** — The best elements are combined into a single, high-quality answer using one of 4 strategies
4. **Consensus Detection** — Points of agreement and disagreement between models are identified

## Synthesis Strategies

| Strategy | Description |
|----------|-------------|
| **Weighted Merge** | Ranks responses by multi-dimensional score, then synthesizes the best elements |
| **Best of N** | Returns the single highest-scored response |
| **Debate** | Models critique each other's answers in rounds, then merge insights |
| **Chain of Verification** | Weighted merge + verification pass to catch errors |

## Supported Providers

- **Anthropic** — Claude Sonnet 4.6, Claude Haiku 4.5
- **OpenAI** — GPT-4o, o3-mini
- **Google** — Gemini 2.5 Pro, Gemini 2.5 Flash
- **DeepSeek** — DeepSeek R1
- **Ollama** — Any local model (Llama, Qwen, etc.)

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/aoloo-r/ai_resoning_box.git
cd ai_resoning_box
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Web UI

```bash
uvicorn server:app --port 8900
# Open http://localhost:8900
```

### 4. Or Use CLI

```bash
python main.py "Your question here"
python main.py --strategy debate "Compare Python vs Rust"
python main.py --interactive
```

## Configuration

Edit `config.yaml` to:
- Enable/disable providers
- Add or remove models
- Change the synthesis strategy
- Adjust scoring weights
- Set concurrency and timeout limits

## Deployment

### Docker

```bash
docker build -t ai-reasoning-box .
docker run -p 8900:8900 --env-file .env ai-reasoning-box
```

### Render (One-Click)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aoloo-r/ai_resoning_box)

Set your API keys as environment variables in the Render dashboard.

### Railway / Fly.io

Uses the included `Dockerfile` and `Procfile`. Set API keys as environment variables.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | Get active models and providers |
| `POST` | `/api/query` | Submit a query for ensemble reasoning |
| `GET` | `/api/strategies` | List available synthesis strategies |

### Query Example

```bash
curl -X POST http://localhost:8900/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain quantum computing", "strategy": "weighted_merge"}'
```

## Project Structure

```
├── core/
│   ├── models.py          # Data models and enums
│   ├── orchestrator.py    # Multi-model dispatcher
│   ├── synthesizer.py     # Response evaluation and synthesis
│   ├── pipeline.py        # High-level orchestration API
│   └── providers/         # AI provider integrations
├── static/index.html      # Web interface
├── server.py              # FastAPI web server
├── main.py                # CLI interface
├── config.yaml            # Provider and model configuration
├── Dockerfile             # Container deployment
└── render.yaml            # Render.com deployment config
```
