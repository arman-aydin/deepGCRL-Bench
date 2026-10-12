import time
import torch
import numpy as np
import gymnasium as gym
import ogbench
import ray

from .diffusion_policy import GoalConditionedDiffusionTransformer
from .ping_pong_weight_sync import PingPongWeightSync


@ray.remote
class PipelinedGPURolloutEngine:
    """Asynchronous Rollout Engine operating on dedicated CUDA inference streams.
    
    Generates batch t+1 continuous control trajectories while training rank
    optimizes batch t, overlapping simulation and DDIM denoising computation.
    """
    def __init__(
        self,
        env_name: str,
        state_dim: int,
        action_dim: int,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        num_envs: int = 8,
        obs_mean: np.ndarray = None,
        obs_std: np.ndarray = None,
        device_str: str = "cuda"
    ):
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.inference_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

        self.policy = GoalConditionedDiffusionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            dim=dim,
            depth=depth,
            num_heads=num_heads
        ).to(self.device)
        self.policy.eval()

        self.ping_pong = PingPongWeightSync(self.policy, self.device)

        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.obs_mean = torch.tensor(obs_mean if obs_mean is not None else np.zeros(state_dim), dtype=torch.float32, device=self.device)
        self.obs_std = torch.tensor(obs_std if obs_std is not None else np.ones(state_dim), dtype=torch.float32, device=self.device)
        self.obs_std[self.obs_std == 0] = 1.0

        self.envs = gym.vector.AsyncVectorEnv([
            lambda: ogbench.make_env_and_datasets(env_name)[0] for _ in range(num_envs)
        ])
        self.obs, _ = self.envs.reset()

    def generate_next_batch_async(self, num_steps_per_env: int, global_goal: np.ndarray, num_inference_steps: int = 5):
        """Asynchronously produces a rollout batch on dedicated inference stream."""
        t0 = time.perf_counter()

        # Check for incoming weight updates
        self.ping_pong.swap_active_slot_if_ready(self.policy, self.inference_stream)

        goal_t = torch.tensor(global_goal, dtype=torch.float32, device=self.device)
        goal_norm = (goal_t - self.obs_mean) / self.obs_std
        goals_batch = goal_norm.unsqueeze(0).repeat(self.num_envs, 1)

        batch_obs = []
        batch_act = []
        batch_next_obs = []
        batch_rew = []
        batch_done = []
        batch_goal = []

        context = torch.cuda.stream(self.inference_stream) if self.inference_stream is not None else torch.no_grad()

        with context:
            for _ in range(num_steps_per_env):
                obs_t = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
                obs_norm = (obs_t - self.obs_mean) / self.obs_std

                # Fast DDIM action generation
                actions_gpu = self.policy.sample_actions(
                    obs_norm, goals_batch, num_inference_steps=num_inference_steps, horizon=1
                )
                actions_np = actions_gpu.squeeze(1).cpu().numpy()

                next_obs, rewards, terminated, truncated, infos = self.envs.step(actions_np)

                for i in range(self.num_envs):
                    done = terminated[i] or truncated[i]
                    act_next = next_obs[i]
                    if done and 'final_observation' in infos and infos['final_observation'][i] is not None:
                        act_next = infos['final_observation'][i]

                    batch_obs.append(self.obs[i])
                    batch_act.append(actions_np[i])
                    batch_next_obs.append(act_next)
                    batch_rew.append(rewards[i])
                    batch_done.append(float(done))
                    batch_goal.append(global_goal)

                self.obs = next_obs

        if self.inference_stream is not None:
            self.inference_stream.synchronize()

        elapsed = time.perf_counter() - t0
        total_transitions = len(batch_obs)

        return {
            "obs": np.array(batch_obs, dtype=np.float32),
            "action": np.array(batch_act, dtype=np.float32),
            "next_obs": np.array(batch_next_obs, dtype=np.float32),
            "reward": np.array(batch_rew, dtype=np.float32),
            "done": np.array(batch_done, dtype=np.float32),
            "goal": np.array(batch_goal, dtype=np.float32),
            "elapsed_sec": elapsed,
            "fps": total_transitions / max(elapsed, 1e-5)
        }

    def close(self):
        self.envs.close()
