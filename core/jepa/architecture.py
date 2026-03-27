"""JEPA Architecture for Reasoning — Core neural network components.

This implements a Joint Embedding Predictive Architecture adapted for
text-based reasoning. Instead of predicting pixels (I-JEPA) or video
frames (V-JEPA), we predict reasoning representations.

Key insight: By predicting in embedding space rather than token space,
the model learns abstract reasoning patterns that transfer across domains.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   Question ──> Context Encoder ──> z_context    │
    │                                     │           │
    │                              ┌──────┘           │
    │                              v                  │
    │                         Predictor               │
    │                              │                  │
    │                              v                  │
    │                     z_predicted (reasoning)     │
    │                              │                  │
    │   Good Answer ──> Target Encoder ──> z_target   │
    │                              │                  │
    │              Loss = VICReg(z_predicted, z_target)│
    │                                                 │
    │   Bad Answer ──> Target Encoder ──> z_negative  │
    │              + Contrastive margin loss           │
    └─────────────────────────────────────────────────┘

    Target Encoder is EMA-updated from Context Encoder (no gradient).
"""

from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasoningEncoder(nn.Module):
    """Encodes text into a reasoning representation space.

    Uses a transformer backbone to produce dense embeddings that
    capture the semantic and reasoning structure of text.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        projection_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.projection_dim = projection_dim

        # Token + positional embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # Project to reasoning representation space
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode input tokens into reasoning representation space.

        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) 1=attend, 0=ignore

        Returns:
            (batch, projection_dim) reasoning embeddings
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        # Create causal mask
        if attention_mask is not None:
            # Convert padding mask to transformer format (True = ignore)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)

        # Pool: use mean of non-masked positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Project to reasoning space
        return self.projector(x)


class ReasoningPredictor(nn.Module):
    """Predicts target reasoning representations from context representations.

    This is the core JEPA component — it learns to predict what good
    reasoning looks like given a question, without seeing the answer.

    Uses cross-attention to condition on multiple reasoning aspects:
    - Logical structure
    - Factual grounding
    - Completeness
    - Clarity of explanation
    """

    def __init__(
        self,
        d_input: int = 512,
        d_hidden: int = 1024,
        d_output: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_reasoning_aspects: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Learnable reasoning aspect tokens
        # These represent: accuracy, completeness, reasoning_quality, clarity
        self.aspect_tokens = nn.Parameter(torch.randn(n_reasoning_aspects, d_hidden) * 0.02)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
        )

        # Cross-attention layers: aspect tokens attend to context
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(d_hidden, n_heads, dropout=dropout, batch_first=True),
                "self_attn": nn.MultiheadAttention(d_hidden, n_heads, dropout=dropout, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(d_hidden, d_hidden * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden * 4, d_hidden),
                    nn.Dropout(dropout),
                ),
                "norm1": nn.LayerNorm(d_hidden),
                "norm2": nn.LayerNorm(d_hidden),
                "norm3": nn.LayerNorm(d_hidden),
            }))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_output),
            nn.LayerNorm(d_output),
        )

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """Predict target reasoning representation from context.

        Args:
            context_embedding: (batch, d_input) context encoder output

        Returns:
            (batch, d_output) predicted reasoning embedding
        """
        B = context_embedding.shape[0]

        # Project context
        context = self.input_proj(context_embedding).unsqueeze(1)  # (B, 1, d_hidden)

        # Initialize aspect tokens for this batch
        aspects = self.aspect_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, n_aspects, d_hidden)

        for layer in self.layers:
            # Aspect tokens cross-attend to context
            residual = aspects
            aspects = layer["norm1"](aspects)
            aspects = residual + layer["cross_attn"](aspects, context, context)[0]

            # Self-attention among aspect tokens
            residual = aspects
            aspects = layer["norm2"](aspects)
            aspects = residual + layer["self_attn"](aspects, aspects, aspects)[0]

            # FFN
            residual = aspects
            aspects = layer["norm3"](aspects)
            aspects = residual + layer["ffn"](aspects)

        # Aggregate aspect predictions
        prediction = aspects.mean(dim=1)  # (B, d_hidden)
        return self.output_proj(prediction)


class ReasoningJEPA(nn.Module):
    """Full JEPA model for reasoning.

    Combines context encoder, target encoder (EMA), and predictor
    to learn abstract reasoning representations.

    Training:
        1. Context encoder encodes the question
        2. Predictor predicts the reasoning representation
        3. Target encoder (EMA) encodes the good answer
        4. Loss pulls prediction toward target, pushes away from bad answers

    Inference:
        1. Encode question with context encoder
        2. Predict reasoning representation
        3. Use predicted representation to guide generation
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        d_projection: int = 512,
        d_predictor: int = 1024,
        encoder_layers: int = 12,
        predictor_layers: int = 6,
        n_heads: int = 16,
        ema_decay: float = 0.996,
        **kwargs,
    ):
        super().__init__()

        # Context encoder (trainable)
        self.context_encoder = ReasoningEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=encoder_layers,
            projection_dim=d_projection,
        )

        # Target encoder (EMA of context encoder — no gradients)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = ReasoningPredictor(
            d_input=d_projection,
            d_hidden=d_predictor,
            d_output=d_projection,
            n_layers=predictor_layers,
            n_heads=n_heads // 2,
        )

        self.ema_decay = ema_decay
        self.d_projection = d_projection

    @torch.no_grad()
    def update_target_encoder(self):
        """Exponential moving average update of target encoder."""
        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_tgt.data.mul_(self.ema_decay).add_(param_ctx.data, alpha=1.0 - self.ema_decay)

    def forward(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor | None,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor | None,
        negative_ids: torch.Tensor | None = None,
        negative_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            question_ids: (B, T) question token IDs
            question_mask: (B, T) question attention mask
            answer_ids: (B, T) good answer token IDs
            answer_mask: (B, T) good answer attention mask
            negative_ids: (B, T) bad answer token IDs (optional)
            negative_mask: (B, T) bad answer attention mask (optional)

        Returns:
            Dict with loss components
        """
        # Encode question with context encoder
        z_context = self.context_encoder(question_ids, question_mask)

        # Predict target reasoning representation
        z_predicted = self.predictor(z_context)

        # Encode good answer with target encoder (no gradient)
        with torch.no_grad():
            z_target = self.target_encoder(answer_ids, answer_mask)

        # === VICReg Loss (Variance-Invariance-Covariance Regularization) ===
        # This prevents representation collapse — a key JEPA innovation
        losses = {}

        # 1. Invariance: pull prediction toward target
        invariance_loss = F.mse_loss(z_predicted, z_target)
        losses["invariance"] = invariance_loss

        # 2. Variance: keep variance of each dimension high (prevent collapse)
        std_pred = z_predicted.std(dim=0)
        std_target = z_target.std(dim=0)
        variance_loss = (
            F.relu(1.0 - std_pred).mean() +
            F.relu(1.0 - std_target).mean()
        )
        losses["variance"] = variance_loss

        # 3. Covariance: decorrelate dimensions (prevent redundancy)
        z_pred_centered = z_predicted - z_predicted.mean(dim=0)
        z_tgt_centered = z_target - z_target.mean(dim=0)
        B = z_predicted.shape[0]

        cov_pred = (z_pred_centered.T @ z_pred_centered) / (B - 1)
        cov_tgt = (z_tgt_centered.T @ z_tgt_centered) / (B - 1)

        # Zero out diagonal (we only penalize off-diagonal correlations)
        D = z_predicted.shape[1]
        mask = ~torch.eye(D, device=z_predicted.device).bool()
        covariance_loss = (
            cov_pred[mask].pow(2).mean() +
            cov_tgt[mask].pow(2).mean()
        )
        losses["covariance"] = covariance_loss

        # 4. Contrastive: push prediction away from bad answers
        if negative_ids is not None:
            with torch.no_grad():
                z_negative = self.target_encoder(negative_ids, negative_mask)

            # Margin loss: prediction should be closer to target than negative
            pos_dist = F.pairwise_distance(z_predicted, z_target)
            neg_dist = F.pairwise_distance(z_predicted, z_negative)
            margin = 1.0
            contrastive_loss = F.relu(pos_dist - neg_dist + margin).mean()
            losses["contrastive"] = contrastive_loss

        # Total loss
        losses["total"] = (
            25.0 * losses["invariance"] +
            25.0 * losses["variance"] +
            1.0 * losses["covariance"] +
            10.0 * losses.get("contrastive", torch.tensor(0.0))
        )

        return losses

    @torch.no_grad()
    def encode_question(self, question_ids: torch.Tensor, question_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a question into reasoning representation space (inference)."""
        z_context = self.context_encoder(question_ids, question_mask)
        z_predicted = self.predictor(z_context)
        return z_predicted

    @torch.no_grad()
    def encode_answer(self, answer_ids: torch.Tensor, answer_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode an answer into reasoning representation space (inference)."""
        return self.target_encoder(answer_ids, answer_mask)

    def similarity(self, z_question: torch.Tensor, z_answer: torch.Tensor) -> torch.Tensor:
        """Compute reasoning similarity between predicted and actual embeddings."""
        return F.cosine_similarity(z_question, z_answer, dim=-1)
