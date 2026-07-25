import numpy as np
import torch

class GoalConditionedReplayBuffer:
    """A memory-efficient replay buffer for Goal-Conditioned Reinforcement Learning.
    
    Stores transitions (obs, action, next_obs, reward, done, goal) in pre-allocated
    numpy arrays, and returns them as PyTorch tensors upon sampling.
    """
    def __init__(self, max_size: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Pre-allocate numpy arrays for performance
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
        """Append a new transition to the buffer."""
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.goal[self.ptr] = goal
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """Sample a batch of transitions as PyTorch tensors on the target device."""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.as_tensor(self.obs[ind], device=self.device),
            torch.as_tensor(self.action[ind], device=self.device),
            torch.as_tensor(self.next_obs[ind], device=self.device),
            torch.as_tensor(self.reward[ind], device=self.device),
            torch.as_tensor(self.done[ind], device=self.device),
            torch.as_tensor(self.goal[ind], device=self.device)
        )

    def __len__(self) -> int:
        return self.size
