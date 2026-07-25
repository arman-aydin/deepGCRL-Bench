import ray
import gymnasium as gym
import ogbench
import torch
import numpy as np
from actor_modeling import GoalConditionedActor

@ray.remote
class RolloutWorker:
    """A remote Ray actor that manages vectorized environments and collects experiences."""
    def __init__(self, env_name: str, state_dim: int, action_dim: int, 
                 mean: np.ndarray, std: np.ndarray, num_envs: int = 4):
        self.device = torch.device("cpu") # Keep worker execution on CPU to reserve GPU for Trainer
        self.policy = GoalConditionedActor(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.policy.eval()
        
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device)
        
        self.num_envs = num_envs
        
        # Initialize parallel vectorized environments using processes
        self.envs = gym.vector.AsyncVectorEnv([
            lambda: ogbench.make_env_and_datasets(env_name)[0] for _ in range(num_envs)
        ])
        self.obs, _ = self.envs.reset()

    def sync_weights(self, weights: dict):
        """Update local actor parameters with fresh weights from the trainer."""
        self.policy.load_state_dict(weights)

    def collect_steps(self, num_steps: int, goal_state: np.ndarray, queue):
        """Collect transitions and push them directly to a shared Ray Queue."""
        goal_tensor = torch.tensor(goal_state, dtype=torch.float32, device=self.device)
        goal_norm = (goal_tensor - self.mean) / self.std
        
        for _ in range(num_steps):
            # Normalize current observations
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
            obs_norm = (obs_tensor - self.mean) / self.std
            
            # Batch the single goal state for policy input
            goals_batch = goal_norm.unsqueeze(0).repeat(self.num_envs, 1)
            
            # Compute actions (eval mode, no gradients needed)
            with torch.no_grad():
                actions = self.policy(obs_norm, goals_batch).cpu().numpy()
            
            # Step the vectorized environments
            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
            
            step_transitions = []
            for i in range(self.num_envs):
                done = terminated[i] or truncated[i]
                
                # Gymnasium Auto-Reset handling:
                actual_next_obs = next_obs[i]
                if done and 'final_observation' in infos:
                    final_obs = infos['final_observation'][i]
                    if final_obs is not None:
                        actual_next_obs = final_obs
                
                step_transitions.append({
                    'obs': self.obs[i],
                    'action': actions[i],
                    'next_obs': actual_next_obs,
                    'reward': rewards[i],
                    'done': done,
                    'goal': goal_state
                })
            
            # Push the batch to the Queue
            queue.put(step_transitions)
            self.obs = next_obs

    def close(self):
        """Shut down the sub-process environments."""
        self.envs.close()
