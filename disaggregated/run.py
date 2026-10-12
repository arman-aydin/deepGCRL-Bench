import os
import sys
import time
import numpy as np
import torch
import ray
import ogbench

from disaggregated import (
    GoalConditionedDiffusionTransformer,
    CUDAPeerWeightBroadcaster,
    CUDAPeerWeightReceiver,
    StreamingShardedReplayBuffer,
    DisaggregatedGPURolloutEngine
)


def benchmark_disaggregated():
    print("=" * 70)
    print("DISAGGREGATED ENGINE BENCHMARK (FSDP2 + CUDA IPC + Batched DDIM)")
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
    print(f"Policy Parameter Count:       {total_params:,} ({total_params/1e6:.1f}M)")

    # 2. CUDA Peer Zero-Copy IPC Benchmark
    broadcaster = CUDAPeerWeightBroadcaster(model, device)
    receiver = CUDAPeerWeightReceiver(model, device)

    # Warmup
    handle, _ = broadcaster.pack_and_broadcast_async(model)
    receiver.unpack_weights_async(handle)

    trials = 10
    ipc_latencies = []
    for _ in range(trials):
        handle, b_ms = broadcaster.pack_and_broadcast_async(model)
        r_ms = receiver.unpack_weights_async(handle)
        ipc_latencies.append(b_ms + r_ms)

    avg_ipc_ms = float(np.mean(ipc_latencies))
    print(f"CUDA IPC Weight Broadcast:    {avg_ipc_ms:.2f} ms")

    # 3. Streaming Sharded Buffer Throughput
    buffer = StreamingShardedReplayBuffer(50000, state_dim, action_dim, action_horizon=1, device="cpu")
    dummy_obs = np.random.randn(1000, state_dim).astype(np.float32)
    dummy_act = np.random.randn(1000, action_dim).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(10):
        buffer.push_batch(dummy_obs, dummy_act, dummy_obs, np.ones(1000), np.zeros(1000), dummy_obs)
    ingest_time = time.perf_counter() - t0
    ingest_rate = 10000 / ingest_time
    print(f"Decentralized Buffer Ingest:  {ingest_rate:,.0f} transitions/sec")

    # 4. Batched GPU Rollout Engine
    _, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
    obs_raw = torch.from_numpy(train_dataset['observations'])
    obs_mean = obs_raw.mean(dim=0).numpy()
    obs_std = obs_raw.std(dim=0).numpy()
    obs_std[obs_std == 0] = 1.0

    print("Launching Batched GPU Rollout Engine...")
    rollout_engine = DisaggregatedGPURolloutEngine.remote(
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

    t0 = time.perf_counter()
    batch_res = ray.get(rollout_engine.collect_rollout_batch.remote(
        num_steps_per_env=10,
        global_goal=obs_raw[0].numpy(),
        num_inference_steps=5
    ))
    rollout_time = time.perf_counter() - t0
    rollout_fps = (8 * 10) / rollout_time
    print(f"Batched DDIM Rollout FPS:     {rollout_fps:.1f} transitions/sec")
    print("=" * 70)

    return {
        "parameters": total_params,
        "ipc_sync_latency_ms": avg_ipc_ms,
        "buffer_ingest_rate": ingest_rate,
        "rollout_fps": rollout_fps
    }


if __name__ == '__main__':
    benchmark_disaggregated()
