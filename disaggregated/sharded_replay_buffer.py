import threading
import numpy as np
import torch


class StreamingShardedReplayBuffer:
    """High-Throughput Streaming Replay Buffer for Disaggregated Multi-GPU RL.
    
    Designed to eliminate the centralized Rank-0 broadcast bottleneck.
    Supports O(1) vectorized batch ingestion directly from batched GPU rollout
    workers and concurrent lock-free sampling across distributed FSDP2 ranks.
    """
    def __init__(
        self,
        max_size: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 1,
        device: str = "cpu"
    ):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = torch.device(device)

        # Pre-allocate contiguous arrays for memory efficiency
        act_shape = (max_size, action_dim) if action_horizon == 1 else (max_size, action_horizon, action_dim)
        self.obs = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros(act_shape, dtype=np.float32)
        self.next_obs = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.goal = np.zeros((max_size, state_dim), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.total_inserted = 0
        self._lock = threading.Lock()

    def push_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        goal: np.ndarray
    ) -> int:
        """Vectorized insertion of a batch of experiences into the circular buffer.
        
        Args:
            obs: [B, state_dim]
            action: [B, action_dim] or [B, horizon, action_dim]
            next_obs: [B, state_dim]
            reward: [B] or [B, 1]
            done: [B] or [B, 1]
            goal: [B, state_dim]
        Returns:
            Current total buffer size
        """
        batch_size = obs.shape[0]
        if batch_size == 0:
            return self.size

        # Ensure 2D shapes for 1D arrays
        reward_2d = reward.reshape(-1, 1).astype(np.float32)
        done_2d = done.reshape(-1, 1).astype(np.float32)

        with self._lock:
            if self.ptr + batch_size <= self.max_size:
                # Contiguous slice insertion
                idx_slice = slice(self.ptr, self.ptr + batch_size)
                self.obs[idx_slice] = obs
                self.action[idx_slice] = action
                self.next_obs[idx_slice] = next_obs
                self.reward[idx_slice] = reward_2d
                self.done[idx_slice] = done_2d
                self.goal[idx_slice] = goal
            else:
                # Wrapped boundary insertion
                first_chunk = self.max_size - self.ptr
                second_chunk = batch_size - first_chunk

                # First chunk to the end
                self.obs[self.ptr:] = obs[:first_chunk]
                self.action[self.ptr:] = action[:first_chunk]
                self.next_obs[self.ptr:] = next_obs[:first_chunk]
                self.reward[self.ptr:] = reward_2d[:first_chunk]
                self.done[self.ptr:] = done_2d[:first_chunk]
                self.goal[self.ptr:] = goal[:first_chunk]

                # Second chunk wrapped to the start
                self.obs[:second_chunk] = obs[first_chunk:]
                self.action[:second_chunk] = action[first_chunk:]
                self.next_obs[:second_chunk] = next_obs[first_chunk:]
                self.reward[:second_chunk] = reward_2d[first_chunk:]
                self.done[:second_chunk] = done_2d[first_chunk:]
                self.goal[:second_chunk] = goal[first_chunk:]

            self.ptr = (self.ptr + batch_size) % self.max_size
            self.size = min(self.size + batch_size, self.max_size)
            self.total_inserted += batch_size
            return self.size

    def sample(self, batch_size: int, device: torch.device = None):
        """Asynchronously sample a batch as PyTorch tensors for distributed ranks."""
        curr_device = device or self.device
        current_size = self.size
        if current_size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {current_size}")

        ind = np.random.randint(0, current_size, size=batch_size)

        return (
            torch.as_tensor(self.obs[ind], device=curr_device, dtype=torch.float32),
            torch.as_tensor(self.action[ind], device=curr_device, dtype=torch.float32),
            torch.as_tensor(self.next_obs[ind], device=curr_device, dtype=torch.float32),
            torch.as_tensor(self.reward[ind], device=curr_device, dtype=torch.float32),
            torch.as_tensor(self.done[ind], device=curr_device, dtype=torch.float32),
            torch.as_tensor(self.goal[ind], device=curr_device, dtype=torch.float32),
        )

    def memory_footprint_mb(self) -> float:
        """Returns total memory consumed by pre-allocated buffer arrays in MB."""
        total_bytes = (
            self.obs.nbytes + self.action.nbytes + self.next_obs.nbytes +
            self.reward.nbytes + self.done.nbytes + self.goal.nbytes
        )
        return total_bytes / (1024 * 1024)

    def __len__(self) -> int:
        return self.size