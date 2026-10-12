import numpy as np
import torch
import torch.distributed as dist


class CentralizedReplayBuffer:
    """Centralized Replay Buffer with Blocking dist.broadcast Primitives.
    
    In naive distributed setups, Rank 0 maintains the sole physical replay buffer.
    To execute a training update across distributed workers, Rank 0 samples a batch
    and sequentially broadcasts 6 individual tensors over the distributed process group.
    """
    def __init__(self, max_size: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.obs = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.goal = np.zeros((max_size, state_dim), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray,
            reward: float, done: bool, goal: np.ndarray):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.goal[self.ptr] = goal

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def broadcast_sample(self, batch_size: int, rank: int, device: torch.device):
        """Rank 0 samples, then issues 6 blocking broadcast calls across ranks."""
        if rank == 0:
            ind = np.random.randint(0, max(self.size, 1), size=batch_size)
            s_t = torch.as_tensor(self.obs[ind], device=device, dtype=torch.float32)
            a_t = torch.as_tensor(self.action[ind], device=device, dtype=torch.float32)
            s_next = torch.as_tensor(self.next_obs[ind], device=device, dtype=torch.float32)
            r_t = torch.as_tensor(self.reward[ind], device=device, dtype=torch.float32)
            d_t = torch.as_tensor(self.done[ind], device=device, dtype=torch.float32)
            s_g = torch.as_tensor(self.goal[ind], device=device, dtype=torch.float32)
        else:
            s_t = torch.zeros((batch_size, self.state_dim), device=device, dtype=torch.float32)
            a_t = torch.zeros((batch_size, self.action_dim), device=device, dtype=torch.float32)
            s_next = torch.zeros((batch_size, self.state_dim), device=device, dtype=torch.float32)
            r_t = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
            d_t = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
            s_g = torch.zeros((batch_size, self.state_dim), device=device, dtype=torch.float32)

        if dist.is_initialized():
            dist.broadcast(s_t, src=0)
            dist.broadcast(a_t, src=0)
            dist.broadcast(s_next, src=0)
            dist.broadcast(r_t, src=0)
            dist.broadcast(d_t, src=0)
            dist.broadcast(s_g, src=0)

        return s_t, a_t, s_next, r_t, d_t, s_g

    def __len__(self) -> int:
        return self.size
