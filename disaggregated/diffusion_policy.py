import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiTBlock(nn.Module):
    """Diffusion Transformer Block with Adaptive Layer Normalization (AdaLN-Zero)."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # AdaLN modulation generates: (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        # Initialize gate weights to zero (AdaLN-Zero initialization)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c shape: [B, dim]
        mod = self.adaLN_modulation(c) # [B, 6 * dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Modulate attention input
        norm_x = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Modulate MLP input
        norm_x2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class GoalConditionedDiffusionTransformer(nn.Module):
    """150M+ Parameter Goal-Conditioned Diffusion Transformer Policy (DiT).
    
    Conditioned on (state, goal, timestep), this policy models continuous
    actions using score-based diffusion. Designed specifically for
    FSDP2 distributed sharding across multi-GPU clusters.
    """
    def __init__(
        self,
        state_dim: int = 69,
        action_dim: int = 8,
        action_horizon: int = 4,
        dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_train_timesteps: int = 100
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_train_timesteps = num_train_timesteps

        # Action input projection
        self.action_in_proj = nn.Linear(action_dim, dim)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, action_horizon, dim))
        nn.init.normal_(self.action_pos_emb, std=0.02)

        # Diffusion timestep conditioning
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # Goal and State conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(state_dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # Transformer DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final Layer (AdaLN + Linear head to action_dim)
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        self.final_proj = nn.Linear(dim, action_dim)
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        # Register diffusion noise schedule
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        is_2d = (noisy_actions.ndim == 2)
        if is_2d:
            noisy_actions = noisy_actions.unsqueeze(1)

        # 1. Action tokens + position embedding
        x = self.action_in_proj(noisy_actions) + self.action_pos_emb[:, :noisy_actions.shape[1], :]

        # 2. Condition representation: timestep + state-goal condition
        t_emb = self.time_emb(timesteps)
        state_goal = torch.cat([state, goal], dim=-1)
        cond_emb = self.cond_proj(state_goal)
        c = t_emb + cond_emb

        # 3. Pass through DiT Blocks
        for block in self.blocks:
            x = block(x, c)

        # 4. Final AdaLN and projection
        mod = self.final_adaLN(c)
        shift, scale = mod.chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.final_proj(x)

        if is_2d:
            out = out.squeeze(1)
        return out

    def compute_loss(self, actions: torch.Tensor, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        device = actions.device
        batch_size = actions.shape[0]

        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(actions)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[timesteps].view(-1, *([1] * (actions.ndim - 1)))
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, *([1] * (actions.ndim - 1)))
        noisy_actions = sqrt_alpha_t * actions + sqrt_one_minus_alpha_t * noise

        pred_noise = self.forward(noisy_actions, timesteps, state, goal)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        num_inference_steps: int = 5,
        horizon: int = None
    ) -> torch.Tensor:
        """Accelerated DDIM sampling for fast batched GPU rollout inference."""
        device = state.device
        batch_size = state.shape[0]
        h = horizon or self.action_horizon

        shape = (batch_size, h, self.action_dim) if h > 1 else (batch_size, self.action_dim)
        x = torch.randn(shape, device=device)

        step_ratio = max(1, self.num_train_timesteps // num_inference_steps)
        time_steps = torch.arange(0, self.num_train_timesteps, step_ratio, device=device).flip(0)

        for i, t in enumerate(time_steps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.forward(x, t_batch, state, goal)

            alpha_t = self.alphas_cumprod[t]
            prev_t = time_steps[i + 1] if i + 1 < len(time_steps) else -1
            alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)

            pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            direction_xt = torch.sqrt(1.0 - alpha_prev) * pred_noise
            x = torch.sqrt(alpha_prev) * pred_x0 + direction_xt

        return torch.clamp(x, -1.0, 1.0)