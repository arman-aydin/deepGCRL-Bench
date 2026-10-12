import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-Norm Residual Block from Wang et al. (arXiv:2503.14858)."""
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ln(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        return residual + out


class ResNetConfigDepth(nn.Module):
    """Deep Pre-Norm ResNet for Self-Supervised Contrastive Goal-Conditioned RL."""
    def __init__(self, depth: int = 32, state_dim: int = 29, hidden_dim: int = 256):
        super().__init__()
        self.depth = depth
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.layers = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.state_proj(x)
        for layer in self.layers:
            out = layer(out)
        return self.final_norm(out)


class WangContrastiveCritic(nn.Module):
    """Pre-Trained 32-Layer ResNet Contrastive Critic (Wang et al., 2025).
    
    Evaluates topological reachability and advantage values:
    V(s, g) = normalize(phi(s)) . normalize(psi(g))
    Advantage A(s, a, g) = V(s_next, g) - V(s, g)
    """
    def __init__(
        self,
        checkpoint_path: str = None,
        state_dim: int = 29,
        hidden_dim: int = 256,
        depth: int = 32,
        device: torch.device = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ResNetConfigDepth(depth=depth, state_dim=state_dim, hidden_dim=hidden_dim).to(self.device)
        self.goal_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.goal_proj.load_state_dict(ckpt['goal_proj_state_dict'])
            print(f"Loaded Wang et al. 32-layer ResNet Critic from: {checkpoint_path}")
        else:
            print("Initialized untrained ResNet Critic.")

        # Freeze critic weights
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def get_value(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity value metric between state and goal."""
        with torch.no_grad():
            z_s = F.normalize(self.model(state), p=2.0, dim=-1)
            z_g = F.normalize(self.goal_proj(self.model(goal)), p=2.0, dim=-1)
            return (z_s * z_g).sum(dim=-1)

    def compute_advantage(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        goal: torch.Tensor,
        alpha: float = 0.75
    ) -> torch.Tensor:
        """Compute normalized exponential advantage weights for policy training."""
        with torch.no_grad():
            v_t = self.get_value(state, goal)
            v_next = self.get_value(next_state, goal)
            raw_adv = v_next - v_t

            # Standardize across batch
            adv_norm = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
            weights = torch.exp(alpha * adv_norm).clamp(max=100.0)
            return weights
