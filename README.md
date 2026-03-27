# AI Reasoning Box

**Multi-Model Ensemble Reasoning Platform** — Query multiple AI models simultaneously, score their responses, and synthesize the best answer.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Live Demo

**https://ai-reasoning-box.onrender.com**

> No server-side API keys needed. Users bring their own keys — entered in the browser and stored only in localStorage.

## How It Works

1. **Enter Your API Keys** — Click "API Keys" in the header and add your keys for any provider (Anthropic, OpenAI, Google, DeepSeek). Keys are stored in your browser only and never saved on the server.
2. **Ask a Question** — Your question is sent to all configured AI models in parallel.
3. **Scoring** — Each response is evaluated across 4 dimensions: accuracy, completeness, reasoning quality, and clarity.
4. **Synthesis** — The best elements are combined into a single answer using your chosen strategy.
5. **Consensus Detection** — Points of agreement and disagreement between models are identified.

## Supported Providers

| Provider | Models | API Key Source |
|----------|--------|----------------|
| **Anthropic** | Claude Sonnet 4.6, Claude Haiku 4.5 | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| **OpenAI** | GPT-4o, o3-mini | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Google** | Gemini 2.5 Pro, Gemini 2.5 Flash | [aistudio.google.com](https://aistudio.google.com/apikey) |
| **DeepSeek** | DeepSeek R1 | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |

You only need **at least one** API key to use the platform. More keys = more models = better synthesis.

## Synthesis Strategies

| Strategy | Description |
|----------|-------------|
| **Weighted Merge** | Ranks responses by multi-dimensional score, then synthesizes the best elements |
| **Best of N** | Returns the single highest-scored response |
| **Debate** | Models critique each other's answers in rounds, then merge insights |
| **Chain of Verification** | Weighted merge + verification pass to catch errors |

## Self-Hosting

### Quick Start

```bash
git clone https://github.com/aoloo-r/ai_resoning_box.git
cd ai_resoning_box
pip install -r requirements.txt
uvicorn server:app --port 8900
# Open http://localhost:8900
```

No `.env` file needed — users enter their own API keys in the web interface.

### Docker

```bash
docker build -t ai-reasoning-box .
docker run -p 8900:8900 ai-reasoning-box
```

### Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aoloo-r/ai_resoning_box)

No environment variables required. Users provide their own API keys in the browser.

### CLI Usage

The CLI still supports environment variables for local use:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py "Your question here"
python main.py --strategy debate "Compare Python vs Rust"
python main.py --interactive
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/status` | Get active models (with user's API keys) |
| `POST` | `/api/query` | Submit a query for ensemble reasoning |
| `GET` | `/api/strategies` | List available synthesis strategies |

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
│   ├── models.py          # Data models and enums
│   ├── orchestrator.py    # Multi-model dispatcher
│   ├── synthesizer.py     # Response evaluation and synthesis
│   ├── pipeline.py        # High-level orchestration API
│   └── providers/         # AI provider integrations
├── static/index.html      # Web interface (BYO API keys)
├── server.py              # FastAPI web server
├── main.py                # CLI interface
├── config.yaml            # Provider and model configuration
├── Dockerfile             # Container deployment
└── render.yaml            # Render.com deployment config
```

## Security

- API keys are stored **only in your browser's localStorage**
- Keys are sent per-request to the server and used immediately — never logged or persisted
- The server creates a fresh provider instance for each request
- CORS is enabled for the web interface
- No database, no user accounts, no server-side storage
