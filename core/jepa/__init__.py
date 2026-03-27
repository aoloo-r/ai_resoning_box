"""JEPA (Joint Embedding Predictive Architecture) for Reasoning.

Instead of predicting tokens, JEPA learns to predict abstract reasoning
representations in embedding space. This lets the model learn *how to think*
rather than *what to say*.

Architecture:
    Context Encoder  -->  Predictor  -->  Predicted Embedding
    Target Encoder   -->  Target Embedding (EMA updated)

    Loss = distance(Predicted Embedding, Target Embedding)

The predictor learns to anticipate what good reasoning looks like in
abstract space — capturing reasoning patterns, not surface text.
"""

from core.jepa.architecture import (
    ReasoningEncoder,
    ReasoningPredictor,
    ReasoningJEPA,
)
from core.jepa.world_model import WorldModel
