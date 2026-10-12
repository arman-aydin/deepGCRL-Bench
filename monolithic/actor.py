import torch
import torch.nn as nn


class GoalConditionedActor(nn.Module):
    """Monolithic Baseline: Feedforward Goal-Conditioned MLP Policy.
    
    Concatenates state [s] and target goal [g] vectors into a dense network
    with SiLU activations, LayerNorm, and bounded Tanh action outputs.
    """
    def __init__(self, state_dim: int = 69, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
