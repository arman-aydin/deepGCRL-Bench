import torch
import torch.nn as nn


class GoalConditionedActor(nn.Module):
    def __init__(self, state_dim=69, action_dim=8, hidden_dim=256):
        # Action dim for pointmaze is 2. For humanoid, check your dataset.actions.shape[-1]
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim), # Takes state + goal
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Motor actions in MuJoCo are usually bounded [-1, 1]
        )

    def forward(self, state, goal):
        # Concatenate state and goal together
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)