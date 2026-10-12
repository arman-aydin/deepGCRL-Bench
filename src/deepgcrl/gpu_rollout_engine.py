import time
import torch
import numpy as np
import gymnasium as gym
import ogbench
import ray

try:
    from .diffusion_policy import GoalConditionedDiffusionTransformer
    from .ipc_weight_sync import CUDAPeerWeightReceiver
except ImportError:
    from diffusion_policy import GoalConditionedDiffusionTransformer
    from ipc_weight_sync import CUDAPeerWeightReceiver


@ray.remote
class DisaggregatedGPURolloutEngine:
    """Disaggregated Batched GPU Rollout Inference Engine.
    
    A standalone Ray remote worker that dedicates GPU compute to high-throughput
    batched diffusion policy inference and vectorized environment stepping.
    Decoupled entirely from the distributed training backward pass.
    """
    def __init__(
        self,
        env_name: str,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 1,
        dim: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        num_envs: int = 8,
        obs_mean: np.ndarray = None,
        obs_std: np.ndarray = None,
        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device_str)
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Normalization statistics
        if obs_mean is not None:
            self.obs_mean = torch.tensor(obs_mean, dtype=torch.float32, device=self.device)
            self.obs_std = torch.tensor(obs_std, dtype=torch.float32, device=self.device)
            self.obs_std[self.obs_std == 0] = 1.0
        else:
            self.obs_mean = torch.zeros(state_dim, dtype=torch.float32, device=self.device)
            self.obs_std = torch.ones(state_dim, dtype=torch.float32, device=self.device)

        # 1. Initialize GPU Inference Policy (Eval Mode)
        self.policy = GoalConditionedDiffusionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dim=dim,
            depth=depth,
            num_heads=num_heads
        ).to(self.device)
        self.policy.eval()

        # 2. CUDA Inference Stream & Peer Receiver for Zero-Copy Weight Sync
        self.receiver = CUDAPeerWeightReceiver(self.policy, self.device)
        self.is_cuda = (self.device.type == "cuda" and torch.cuda.is_available())
        if self.is_cuda:
            self.inference_stream = torch.cuda.Stream(device=self.device)
        else:
            self.inference_stream = None

        # 3. Vectorized Simulation Pool
        self.envs = gym.vector.AsyncVectorEnv([
            lambda: ogbench.make_env_and_datasets(env_name)[0] for _ in range(num_envs)
        ])
        self.obs, _ = self.envs.reset()

        # Telemetry metrics
        self.total_steps_collected = 0
        self.total_inference_time_ms = 0.0

    def sync_weights(self, ipc_handle_or_tensor) -> float:
        """Receive updated weights directly via zero-copy GPU IPC."""
        latency_ms = self.receiver.unpack_weights_async(ipc_handle_or_tensor)
        return latency_ms

    def collect_rollout_batch(
        self,
        num_steps_per_env: int,
        global_goal: np.ndarray,
        num_inference_steps: int = 5
    ) -> dict:
        """Execute batched GPU rollout inference and step parallel environments.
        
        Args:
            num_steps_per_env: Steps to advance each environment
            global_goal: Target goal coordinate / state
            num_inference_steps: Fast DDIM denoising steps (default: 5)
        Returns:
            Dictionary containing batch transitions and rollout metrics
        """
        batch_size = self.num_envs
        goal_tensor = torch.tensor(global_goal, dtype=torch.float32, device=self.device)
        goal_norm = (goal_tensor - self.obs_mean) / self.obs_std
        goals_batch = goal_norm.unsqueeze(0).repeat(batch_size, 1)

        collected_obs = []
        collected_actions = []
        collected_next_obs = []
        collected_rewards = []
        collected_dones = []
        collected_goals = []

        t_start = time.perf_counter()

        for _ in range(num_steps_per_env):
            # Batch normalize observations on GPU
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
            obs_norm = (obs_tensor - self.obs_mean) / self.obs_std

            # Fast batched DDIM inference on dedicated CUDA stream
            with torch.no_grad():
                if self.is_cuda:
                    with torch.cuda.stream(self.inference_stream):
                        actions_gpu = self.policy.sample_actions(
                            obs_norm,
                            goals_batch,
                            num_inference_steps=num_inference_steps,
                            horizon=self.action_horizon
                        )
                    self.inference_stream.synchronize()
                else:
                    actions_gpu = self.policy.sample_actions(
                        obs_norm,
                        goals_batch,
                        num_inference_steps=num_inference_steps,
                        horizon=self.action_horizon
                    )

            if self.action_horizon > 1:
                # Take first action in chunk for env step
                actions_cpu = actions_gpu[:, 0, :].cpu().numpy()
                actions_stored = actions_gpu.cpu().numpy()
            else:
                actions_cpu = actions_gpu.cpu().numpy()
                actions_stored = actions_cpu

            # Step parallel environments
            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions_cpu)
            dones = terminated | truncated

            # Gymnasium auto-reset handle
            actual_next_obs = next_obs.copy()
            if 'final_observation' in infos:
                for i in range(batch_size):
                    if dones[i] and infos['final_observation'][i] is not None:
                        actual_next_obs[i] = infos['final_observation'][i]

            collected_obs.append(self.obs)
            collected_actions.append(actions_stored)
            collected_next_obs.append(actual_next_obs)
            collected_rewards.append(rewards)
            collected_dones.append(dones)
            collected_goals.append(np.tile(global_goal, (batch_size, 1)))

            self.obs = next_obs

        elapsed = time.perf_counter() - t_start
        steps_collected = num_steps_per_env * batch_size
        self.total_steps_collected += steps_collected
        sps = steps_collected / max(1e-6, elapsed)

        return {
            "obs": np.concatenate(collected_obs, axis=0),
            "action": np.concatenate(collected_actions, axis=0),
            "next_obs": np.concatenate(collected_next_obs, axis=0),
            "reward": np.concatenate(collected_rewards, axis=0),
            "done": np.concatenate(collected_dones, axis=0),
            "goal": np.concatenate(collected_goals, axis=0),
            "steps": steps_collected,
            "elapsed_sec": elapsed,
            "sps": sps
        }

    def close(self):
        self.envs.close()