import torch
import torch.nn as nn
import random

import torch
import torch.nn as nn
import torch.nn.functional as F  # Still useful for default activation

class PFF(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_factor: int,
        activation: str,
        norm: callable,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()
        self.stochastic_depth = stochastic_depth
        self.do_stochastic_depth = stochastic_depth > 0.0
        hidden_dim = dim * expansion_factor

        self.ff_layer = nn.Sequential(
            norm(dim),
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.do_stochastic_depth:
            return x + self.ff_layer(x)

        if self.training:
            # if we are training, randomly drop the layer
            if random.random() < self.stochastic_depth:
                return x
            else:
                return x + self.ff_layer(x)

        else:
            # if we are in validation mode, scale the layer output by the retention probability
            return x + (1 - self.stochastic_depth) * self.ff_layer(x)