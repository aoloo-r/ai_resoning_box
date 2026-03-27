# AI Reasoning Box

**Multi-Model Ensemble Reasoning Platform with JEPA Architecture** — Query multiple AI models simultaneously, learn where they fail, and train a model that beats them all.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![JEPA](https://img.shields.io/badge/Architecture-JEPA-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Live Demo

**https://ai-reasoning-box.onrender.com**

> Users bring their own API keys — stored only in the browser, never on the server.

## The Vision: Beat Every AI Model

```
┌──────────────────────────────────────────────────────────────────┐
│                     AI REASONING BOX                             │
│                                                                  │
│  1. QUERY    All frontier models in parallel                     │
│              Claude, GPT-4o, Gemini, DeepSeek                    │
│                          │                                       │
│  2. SCORE    4-dimensional evaluation                            │
│              Accuracy, Completeness, Reasoning, Clarity          │
│                          │                                       │
│  3. FIND     Adversarial Weakness Finder                         │
│     GAPS     "Where do frontier models FAIL?"                    │
│                          │                                       │
│  4. JEPA     Learn reasoning patterns in embedding space         │
│     LEARN    (not token prediction — abstract reasoning)         │
│                          │                                       │
│  5. TRAIN    Fine-tune on best answers + weakness gaps            │
│              LoRA + Reward Model + RLHF                          │
│                          │                                       │
│  6. DEPLOY   ReasoningBox joins the ensemble                     │
│              Gets scored alongside frontier models                │
│                          │                                       │
│  7. REPEAT   The more you use it, the better it gets             │
└──────────────────────────────────────────────────────────────────┘
```

## Architecture: JEPA for Reasoning

Inspired by Yann LeCun's **Joint Embedding Predictive Architecture** (JEPA), our model learns to reason in abstract embedding space rather than just predicting tokens.

```
Standard LLM:     Question → [predict tokens] → Answer  (copies patterns)

ReasoningBox:     Question → [JEPA: predict reasoning PATTERN]
                           → [World Model: verify quality]
                           → [Meta-Reasoning: think-critique-refine]
                           → [Guided Decoding: steer toward best reasoning]
                           → Answer  (learns to THINK)
```

### Why JEPA Beats Standard Fine-Tuning

| Approach | What It Learns | Limitation |
|----------|---------------|------------|
| Standard fine-tuning | Surface text patterns | Copies style, not reasoning |
| RLHF | Human preferences | Expensive, preference noise |
| **JEPA (ours)** | Abstract reasoning representations | Learns *how to think* |

### Five Components

#### 1. JEPA Reasoning Predictor (`core/jepa/`)
- **Context Encoder**: Encodes questions into reasoning space
- **Predictor**: Predicts what good reasoning looks like (before seeing the answer)
- **Target Encoder**: EMA-updated from context encoder (prevents collapse)
- **VICReg Loss**: Variance-Invariance-Covariance regularization + contrastive learning

#### 2. Adversarial Weakness Finder (`core/adversarial.py`)
- Analyzes where Claude, GPT-4o, and Gemini fail
- Categorizes failures: logical reasoning, math, ambiguity, ethics, etc.
- Generates targeted training data for those exact weaknesses
- You don't need to beat them at everything — just where they're weak

#### 3. Meta-Reasoning Engine (`core/meta_reasoning.py`)
- 5-stage pipeline: Decompose → Reason → Critique → Verify → Refine
- Self-evaluating: catches its own errors before answering
- Adjustable depth: trivial questions get fast answers, hard ones get deep reasoning
- Iterative refinement: keeps improving until quality threshold is met

#### 4. Dynamic Router (`core/router.py`)
- Classifies question type using learned patterns
- Routes to optimal model subset (saves cost on easy questions)
- Selects best synthesis strategy per question type
- Adjusts temperature and reasoning depth automatically

#### 5. Reward Model (`training/reward_model.py`)
- Trained on the 4-dimensional scoring data from the ensemble
- Enables RLHF fine-tuning of ReasoningBox
- Fast response scoring (replaces expensive LLM-as-judge)
- Best-of-N sampling at inference time

## Quick Start

### 1. Run the Platform

```bash
git clone https://github.com/aoloo-r/ai_resoning_box.git
cd ai_resoning_box
pip install -r requirements.txt
uvicorn server:app --port 8900
```

### 2. Collect Data (Just Use It!)

Every query automatically collects training data. The dashboard tracks your progress.

### 3. Train the JEPA Model

```bash
pip install -r training/requirements-training.txt

# Train JEPA reasoning predictor
python training/train_jepa.py

# Train reward model for RLHF
python training/reward_model.py

# Fine-tune the language model
python training/train.py

# Export for deployment
python training/export_model.py --to-gguf
```

### 4. Analyze Weaknesses

```bash
# The API endpoint shows weakness analysis
curl http://localhost:8900/api/jepa/weaknesses

# Route a query optimally
curl "http://localhost:8900/api/jepa/route?query=Prove+the+Riemann+hypothesis"
```

## Training Pipeline

```
Ensemble Data
     │
     ├──→ train_jepa.py      → JEPA reasoning predictor + World Model
     ├──→ reward_model.py     → Reward model for RLHF
     ├──→ prepare_data.py     → Cleaned training pairs
     ├──→ train.py            → LoRA fine-tuning on best answers
     └──→ adversarial.py      → Targeted weakness training data
            │
            v
     export_model.py → Merged model → Ollama / HuggingFace / vLLM
            │
            v
     ReasoningBox joins the ensemble → Competes with frontier models
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/status` | Active models (with user API keys) |
| `POST` | `/api/query` | Ensemble reasoning query |
| `GET` | `/api/strategies` | List synthesis strategies |
| `GET` | `/api/training/stats` | Training data statistics |
| `GET` | `/api/training/recent` | Recent collected queries |
| `GET` | `/api/jepa/weaknesses` | Frontier model weakness analysis |
| `GET` | `/api/jepa/route?query=...` | Dynamic routing recommendation |

## Project Structure

```
├── core/
│   ├── jepa/                      # JEPA Architecture
│   │   ├── architecture.py        # Encoder + Predictor + VICReg loss
│   │   └── world_model.py         # Quality predictor + Guided decoder
│   ├── adversarial.py             # Weakness finder + adversarial data gen
│   ├── meta_reasoning.py          # Multi-stage reasoning engine
│   ├── router.py                  # Dynamic query router
│   ├── collector.py               # Training data collector
│   ├── models.py                  # Data models
│   ├── orchestrator.py            # Multi-model dispatcher
│   ├── synthesizer.py             # Response scoring + synthesis
│   ├── pipeline.py                # High-level orchestration
│   └── providers/                 # AI provider integrations
│       ├── anthropic_provider.py
│       ├── openai_provider.py
│       ├── google_provider.py
│       ├── deepseek_provider.py
│       ├── ollama_provider.py
│       └── reasoning_box_provider.py
├── training/
│   ├── train_jepa.py              # JEPA training (embedding space)
│   ├── reward_model.py            # Reward model for RLHF
│   ├── train.py                   # LoRA fine-tuning
│   ├── prepare_data.py            # Dataset preparation
│   ├── export_model.py            # Model export (GGUF/HF Hub)
│   └── test_model.py              # Model testing
├── static/index.html              # Web UI + Training dashboard
├── server.py                      # FastAPI server
├── main.py                        # CLI interface
├── config.yaml                    # Configuration
├── Dockerfile                     # Container deployment
└── render.yaml                    # Render.com config
```

## Deployment

### Docker
```bash
docker build -t ai-reasoning-box .
docker run -p 8900:8900 ai-reasoning-box
```

### Render (One-Click)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aoloo-r/ai_resoning_box)

## Security

- API keys stored **only in browser localStorage**
- Keys sent per-request, never logged or persisted server-side
- No database, no user accounts, no server-side storage
