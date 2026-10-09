"""Lightweight probes for frozen ToyJEPA representations."""

import torch
import torch.nn as nn


class MeanPoolLinearProbe(nn.Module):
    """Mean-pool patch tokens and classify a frozen encoder representation.

    ToyJEPA appends register tokens after patch tokens. They are deliberately
    excluded from the mean so variable-length videos remain comparable.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        num_registers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if num_registers < 0:
            raise ValueError("num_registers must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(input_dim, num_classes)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one mean-pooled patch representation per sample."""
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape B,N,D, got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Expected token dimension {self.input_dim}, got {tokens.shape[-1]}")

        num_patch_tokens = tokens.shape[1] - self.num_registers
        if num_patch_tokens <= 0:
            raise ValueError(
                f"Expected at least one patch token before {self.num_registers} register tokens, "
                f"got sequence length {tokens.shape[1]}"
            )
        return tokens[:, :num_patch_tokens].mean(dim=1)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Classify already-pooled features, as used by the feature cache."""
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected pooled features with shape B,{self.input_dim}, got {tuple(features.shape)}")
        return self.head(self.dropout(features))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_features(self.pool_tokens(tokens))
