import time
import ray
import gymnasium as gym
import ogbench
import torch
import numpy as np

from .actor import GoalConditionedActor


@ray.remote
class NaiveRolloutWorker:
    """Ray Remote Worker that synchronizes weights through Plasma Object Store."""
    def __init__(
        self,
        env_name: str,
        state_dim: int,
        action_dim: int,
        mean: np.ndarray,
        std: np.ndarray,
        num_envs: int = 4
    ):
        self.device = torch.device("cpu")
        self.policy = GoalConditionedActor(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.policy.eval()

        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device)
        self.num_envs = num_envs

        self.envs = gym.vector.AsyncVectorEnv([
            lambda: ogbench.make_env_and_datasets(env_name)[0] for _ in range(num_envs)
        ])
        self.obs, _ = self.envs.reset()

    def sync_weights_plasma(self, weight_ref):
        """Fetches weights through Ray Plasma Object Store (CPU D2H / H2D roundtrip)."""
        t0 = time.perf_counter()
        weights = ray.get(weight_ref)
        self.policy.load_state_dict(weights)
        return (time.perf_counter() - t0) * 1000.0

    def collect_transitions(self, num_steps: int, goal_state: np.ndarray):
        """Step environments and return transition batch."""
        goal_tensor = torch.tensor(goal_state, dtype=torch.float32, device=self.device)
        goal_norm = (goal_tensor - self.mean) / self.std

        collected = []
        for _ in range(num_steps):
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
            obs_norm = (obs_tensor - self.mean) / self.std
            goals_batch = goal_norm.unsqueeze(0).repeat(self.num_envs, 1)

            with torch.no_grad():
                actions = self.policy(obs_norm, goals_batch).cpu().numpy()

            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

            for i in range(self.num_envs):
                done = terminated[i] or truncated[i]
                actual_next_obs = next_obs[i]
                if done and 'final_observation' in infos and infos['final_observation'][i] is not None:
                    actual_next_obs = infos['final_observation'][i]

                collected.append({
                    'obs': self.obs[i],
                    'action': actions[i],
                    'next_obs': actual_next_obs,
                    'reward': rewards[i],
                    'done': done,
                    'goal': goal_state
                })
            self.obs = next_obs

        return collected

    def close(self):
        self.envs.close()
