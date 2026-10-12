import time
import numpy as np
import torch
import ray
import ogbench

from pipelined_async.diffusion_policy import GoalConditionedDiffusionTransformer
from pipelined_async.ping_pong_weight_sync import PingPongWeightSync
from pipelined_async.sharded_replay_buffer import StreamingShardedReplayBuffer
from pipelined_async.async_rollout_engine import PipelinedGPURolloutEngine


def benchmark_pipelined_async():
    print("=" * 70)
    print("PIPELINED ASYNCHRONOUS ENGINE (Double-Buffering & Overlap)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_name = "antmaze-large-navigate-v0"

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 1. Inspect Environment & Dataset
    _, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
    obs_raw = torch.from_numpy(train_dataset['observations'])
    obs_mean = obs_raw.mean(dim=0).numpy()
    obs_std = obs_raw.std(dim=0).numpy()
    obs_std[obs_std == 0] = 1.0
    state_dim = obs_raw.shape[-1]
    action_dim = train_dataset['actions'].shape[-1]

    # 232M Model Architecture
    model = GoalConditionedDiffusionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        dim=1024,
        depth=12,
        num_heads=16
    ).to(device)

    total_params = model.count_parameters()
    print(f"Policy Parameter Count:         {total_params:,} ({total_params/1e6:.1f}M)")

    # 2. Ping-Pong Double-Buffered Weight Synchronization
    ping_pong = PingPongWeightSync(model, device)
    push_latencies = []
    swap_latencies = []

    for _ in range(10):
        t_push = ping_pong.push_weights_async(model)
        push_latencies.append(t_push)

        t0 = time.perf_counter()
        ping_pong.swap_active_slot_if_ready(model)
        swap_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_push_ms = np.mean(push_latencies)
    avg_swap_ms = np.mean(swap_latencies)
    print(f"Async Weight Staging Latency:   {avg_push_ms:.2f} ms (Background Stream)")
    print(f"Active Slot Pointer Swap:       {avg_swap_ms:.4f} ms (0-Bubble Handoff)")

    # 3. Continuous Pipelined Rollout Engine
    _, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
    obs_raw = torch.from_numpy(train_dataset['observations'])
    obs_mean = obs_raw.mean(dim=0).numpy()
    obs_std = obs_raw.std(dim=0).numpy()
    obs_std[obs_std == 0] = 1.0

    print("Launching Pipelined GPU Rollout Engine...")
    rollout_engine = PipelinedGPURolloutEngine.remote(
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        dim=256,
        depth=4,
        num_heads=8,
        num_envs=8,
        obs_mean=obs_mean,
        obs_std=obs_std,
        device_str=device.type
    )

    # Double-buffered benchmark: overlap step t training with step t+1 rollout
    buffer = StreamingShardedReplayBuffer(50000, state_dim, action_dim, action_horizon=1, device="cpu")
    num_steps = 5

    # Launch prefetch
    t0 = time.perf_counter()
    future = rollout_engine.generate_next_batch_async.remote(
        num_steps_per_env=10,
        global_goal=obs_raw[0].numpy(),
        num_inference_steps=5
    )

    step_times = []
    for _ in range(num_steps):
        t_iter = time.perf_counter()
        # Concurrently launch next batch rollout while processing current
        next_future = rollout_engine.generate_next_batch_async.remote(
            num_steps_per_env=10,
            global_goal=obs_raw[0].numpy(),
            num_inference_steps=5
        )

        # Simulate compute work on current batch
        batch = ray.get(future)
        buffer.push_batch(batch["obs"], batch["action"], batch["next_obs"], batch["reward"], batch["done"], batch["goal"])
        future = next_future
        step_times.append(time.perf_counter() - t_iter)

    ray.get(future)
    avg_pipelined_step = np.mean(step_times) * 1000.0
    print(f"Double-Buffered Step Latency:   {avg_pipelined_step:.2f} ms")
    print("=" * 70)

    return {
        "parameters": total_params,
        "staging_push_ms": avg_push_ms,
        "pointer_swap_ms": avg_swap_ms,
        "pipelined_step_ms": avg_pipelined_step
    }


if __name__ == '__main__':
    benchmark_pipelined_async()
