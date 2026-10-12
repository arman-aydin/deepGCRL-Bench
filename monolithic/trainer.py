import time
import numpy as np
import torch
import torch.nn.functional as F
import ogbench

from .actor import GoalConditionedActor
from .replay_buffer import GoalConditionedReplayBuffer


class MonolithicTrainer:
    """Synchronous single-process baseline runner.
    
    Executes an alternating lockstep loop:
    1. CPU Environment physics simulation step
    2. Synchronous GPU action inference
    3. Replay buffer insertion
    4. Mini-batch gradient descent update
    """
    def __init__(
        self,
        env_name: str = "antmaze-large-navigate-v0",
        hidden_dim: int = 256,
        buffer_size: int = 20000,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.env_name = env_name
        self.batch_size = batch_size

        # Create environment & dataset
        self.env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
        obs_raw = torch.from_numpy(train_dataset['observations'])
        self.obs_mean = obs_raw.mean(dim=0).to(self.device)
        self.obs_std = obs_raw.std(dim=0).to(self.device)
        self.obs_std[self.obs_std == 0] = 1.0

        self.state_dim = obs_raw.shape[-1]
        self.action_dim = train_dataset['actions'].shape[-1]

        # Map target coordinates
        target_xy = np.array([16.0, 16.0])
        all_obs_flat = obs_raw.numpy()
        all_xy = all_obs_flat[:, :2]
        dist_to_all = np.linalg.norm(all_xy - target_xy, axis=1)
        best_match_idx = dist_to_all.argmin()
        self.global_goal = all_obs_flat[best_match_idx]

        # Policy, optimizer, and buffer
        self.actor = GoalConditionedActor(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.replay_buffer = GoalConditionedReplayBuffer(buffer_size, self.state_dim, self.action_dim, device="cpu")

    def warmup(self, num_steps: int = 200):
        """Pre-fill replay buffer with random exploration transitions."""
        obs, _ = self.env.reset()
        for _ in range(num_steps):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, action, next_obs, reward, done, self.global_goal)
            obs = next_obs if not done else self.env.reset()[0]
        self.current_obs = obs

    def run_benchmark(self, num_steps: int = 100) -> dict:
        """Run synchronous lockstep benchmark and measure phase latencies."""
        if len(self.replay_buffer) < self.batch_size:
            self.warmup(max(self.batch_size * 2, 200))

        env_step_times = []
        inference_times = []
        sgd_times = []

        goal_tensor = torch.tensor(self.global_goal, dtype=torch.float32, device=self.device)
        goal_norm = (goal_tensor - self.obs_mean) / self.obs_std

        total_start = time.perf_counter()

        for _ in range(num_steps):
            # 1. Action inference (GPU)
            t0 = time.perf_counter()
            obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32, device=self.device)
            obs_norm = (obs_tensor - self.obs_mean) / self.obs_std
            with torch.no_grad():
                action = self.actor(obs_norm.unsqueeze(0), goal_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            inference_times.append(time.perf_counter() - t0)

            # 2. Physics Simulation Step (CPU - GPU idles)
            t1 = time.perf_counter()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(self.current_obs, action, next_obs, reward, done, self.global_goal)
            self.current_obs = next_obs if not done else self.env.reset()[0]
            env_step_times.append(time.perf_counter() - t1)

            # 3. Synchronous SGD Step (GPU - CPU idles)
            t2 = time.perf_counter()
            s_t, a_t, s_next, r_t, d_t, s_g = self.replay_buffer.sample(self.batch_size, device=self.device)
            s_t_norm = (s_t - self.obs_mean) / self.obs_std
            s_g_norm = (s_g - self.obs_mean) / self.obs_std

            self.optimizer.zero_grad()
            pred_action = self.actor(s_t_norm, s_g_norm)
            loss = F.mse_loss(pred_action, a_t)
            loss.backward()
            self.optimizer.step()
            sgd_times.append(time.perf_counter() - t2)

        total_elapsed = time.perf_counter() - total_start

        return {
            "total_elapsed_sec": total_elapsed,
            "steps": num_steps,
            "throughput_steps_per_sec": num_steps / total_elapsed,
            "avg_env_step_ms": np.mean(env_step_times) * 1000.0,
            "avg_inference_ms": np.mean(inference_times) * 1000.0,
            "avg_sgd_ms": np.mean(sgd_times) * 1000.0,
            "gpu_idle_percentage": (sum(env_step_times) / total_elapsed) * 100.0,
            "model_parameters": self.actor.count_parameters()
        }
