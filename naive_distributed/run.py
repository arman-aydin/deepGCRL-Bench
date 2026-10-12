import time
import numpy as np
import torch
import ray
import ogbench

from naive_distributed.actor import GoalConditionedActor
from naive_distributed.centralized_buffer import CentralizedReplayBuffer
from naive_distributed.ray_workers import NaiveRolloutWorker


def benchmark_naive_distributed():
    print("=" * 70)
    print("NAIVE DISTRIBUTED BENCHMARK (Ray Plasma + Centralized Broadcast)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_name = "antmaze-large-navigate-v0"

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    _, dataset, _ = ogbench.make_env_and_datasets(env_name)
    obs_raw = torch.from_numpy(dataset['observations'])
    obs_mean = obs_raw.mean(dim=0).numpy()
    obs_std = obs_raw.std(dim=0).numpy()
    obs_std[obs_std == 0] = 1.0

    state_dim = obs_raw.shape[-1]
    action_dim = dataset['actions'].shape[-1]

    # 1. Benchmark Ray Plasma Object Store Serialization
    print("Testing Ray Plasma Object Store Weight Broadcast...")
    policy = GoalConditionedActor(state_dim, action_dim).to(device)
    
    plasma_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        # D2H + Serialization
        cpu_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        ref = ray.put(cpu_dict)
        fetched = ray.get(ref)
        policy.load_state_dict(fetched)
        plasma_times.append((time.perf_counter() - t0) * 1000.0)

    avg_plasma_ms = np.mean(plasma_times)
    print(f"    - Ray Plasma Weight Broadcast Latency: {avg_plasma_ms:.2f} ms")

    # 2. Benchmark Centralized Buffer Sequential Broadcast
    print("Testing Centralized Replay Buffer Broadcast Primitives...")
    buf = CentralizedReplayBuffer(10000, state_dim, action_dim)
    for i in range(100):
        buf.add(
            obs_raw[i].numpy(),
            dataset['actions'][i],
            dataset['next_observations'][i],
            1.0,
            False,
            obs_raw[i].numpy()
        )

    t0 = time.perf_counter()
    num_samples = 100
    for _ in range(num_samples):
        s_t, a_t, s_next, r_t, d_t, s_g = buf.broadcast_sample(batch_size=32, rank=0, device=device)
    broadcast_ms = ((time.perf_counter() - t0) / num_samples) * 1000.0

    print(f"    - Centralized Buffer Sample & Ingestion: {broadcast_ms:.2f} ms/step")
    print("=" * 70)

    return {
        "plasma_sync_latency_ms": avg_plasma_ms,
        "sample_step_ms": broadcast_ms,
        "parameters": policy.count_parameters()
    }


if __name__ == '__main__':
    benchmark_naive_distributed()
