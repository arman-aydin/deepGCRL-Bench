import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import ogbench

try:
    from .diffusion_policy import GoalConditionedDiffusionTransformer
    from .sharded_replay_buffer import StreamingShardedReplayBuffer
    from .ipc_weight_sync import CUDAPeerWeightBroadcaster
except ImportError:
    from diffusion_policy import GoalConditionedDiffusionTransformer
    from sharded_replay_buffer import StreamingShardedReplayBuffer
    from ipc_weight_sync import CUDAPeerWeightBroadcaster

# Check PyTorch FSDP2 availability
try:
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    HAS_FSDP2 = True
except ImportError:
    HAS_FSDP2 = False


def setup_distributed():
    """Initialize distributed process group and select device."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        backend = "gloo" # Multi-rank sharing single GPU
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return rank, local_rank, world_size, device, backend


def build_sharded_policy(
    state_dim: int,
    action_dim: int,
    dim: int = 1024,
    depth: int = 12,
    num_heads: int = 16,
    device: torch.device = None,
    backend: str = "nccl",
    world_size: int = 1
):
    """Instantiate and shard the 150M+ parameter DiT policy using FSDP2."""
    raw_model = GoalConditionedDiffusionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=1,
        dim=dim,
        depth=depth,
        num_heads=num_heads
    ).to(device)

    total_params = raw_model.count_parameters()

    if dist.is_initialized() and world_size > 1:
        if backend == "nccl" and HAS_FSDP2:
            # Native PyTorch FSDP2 with NCCL multi-GPU mesh
            mesh = init_device_mesh("cuda", (world_size,))
            for block in raw_model.blocks:
                fully_shard(block, mesh=mesh)
            model = fully_shard(raw_model, mesh=mesh)
            sharding_type = "PyTorch Composable FSDP2 (NCCL Mesh)"
        else:
            # Standard FSDP1 fallback for single-GPU or Gloo emulation
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            model = FSDP(raw_model, device_id=device if device.type == "cuda" else None)
            sharding_type = "PyTorch FSDP (FullyShardedDataParallel)"
    else:
        if HAS_FSDP2 and device.type == "cuda":
            mesh = init_device_mesh("cuda", (1,))
            for block in raw_model.blocks:
                fully_shard(block, mesh=mesh)
            model = fully_shard(raw_model, mesh=mesh)
            sharding_type = "PyTorch Composable FSDP2 (Single GPU)"
        else:
            model = raw_model
            sharding_type = "Unsharded Local"

    return model, raw_model, total_params, sharding_type


def main():
    parser = argparse.ArgumentParser(description="Disaggregated FSDP2 Training Engine")
    parser.add_argument("--num_updates", type=int, default=50, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per rank")
    parser.add_argument("--dim", type=int, default=1024, help="DiT hidden dimension (1024 -> ~150M+ params)")
    parser.add_argument("--depth", type=int, default=12, help="Number of DiT blocks")
    parser.add_argument("--heads", type=int, default=16, help="Attention heads")
    parser.add_argument("--env_name", type=str, default="antmaze-large-navigate-v0")
    args = parser.parse_args()

    rank, local_rank, world_size, device, backend = setup_distributed()

    if rank == 0:
        print(f"\n========================================================")
        print(f"Initializing Disaggregated FSDP2 Training Engine")
        print(f"   World Size: {world_size} | Device: {device} | Backend: {backend}")
        print(f"========================================================\n")

    # 1. Load dataset & parameters
    _, train_dataset, _ = ogbench.make_env_and_datasets(args.env_name)
    obs_raw = torch.from_numpy(train_dataset['observations'])
    obs_mean = obs_raw.mean(dim=0).numpy()
    obs_std = obs_raw.std(dim=0).numpy()
    obs_std[obs_std == 0] = 1.0

    state_dim = obs_raw.shape[-1]
    action_dim = train_dataset['actions'].shape[-1]

    # Target goal coordinates
    target_xy = np.array([16.0, 16.0])
    all_xy = obs_raw.numpy()[:, :2]
    best_match_idx = np.linalg.norm(all_xy - target_xy, axis=1).argmin()
    global_goal = obs_raw.numpy()[best_match_idx]

    # 2. Build and Shard 150M+ Parameter Diffusion Policy
    model, raw_model, total_params, sharding_type = build_sharded_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        device=device,
        backend=backend,
        world_size=world_size
    )

    if rank == 0:
        print(f"Model Policy Instantiated: {total_params:,} parameters ({total_params/1e6:.1f}M)")
        print(f"Sharding Engine: {sharding_type}")
        sharded_per_rank = total_params // world_size
        print(f"Parameter Memory per Rank: ~{sharded_per_rank * 4 / (1024**2):.1f} MB (vs ~{total_params * 4 / (1024**2):.1f} MB unsharded)")

    # 3. Sharded AdamW Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # 4. Peer-to-Peer Weight Broadcaster (Rank 0)
    broadcaster = CUDAPeerWeightBroadcaster(raw_model, device) if rank == 0 else None

    # 5. Populate Initial Streaming Sharded Buffer from offline dataset
    buffer = StreamingShardedReplayBuffer(
        max_size=50000,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=1,
        device="cpu"
    )

    # Preload initial offline data
    n_pre = min(10000, len(train_dataset['observations']) - 1)
    rewards = train_dataset['rewards'][:n_pre] if 'rewards' in train_dataset else np.zeros((n_pre, 1), dtype=np.float32)

    buffer.push_batch(
        obs=train_dataset['observations'][:n_pre],
        action=train_dataset['actions'][:n_pre],
        next_obs=train_dataset['next_observations'][:n_pre],
        reward=rewards,
        done=train_dataset['terminals'][:n_pre],
        goal=np.tile(global_goal, (n_pre, 1))
    )

    dist.barrier()

    # Pre-extract diffusion constants for training step
    num_train_timesteps = raw_model.num_train_timesteps
    sqrt_alphas_cumprod = raw_model.sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = raw_model.sqrt_one_minus_alphas_cumprod.to(device)

    # 6. Training Loop with Overlapped FSDP Communications
    start_time = time.perf_counter()
    sync_latencies = []

    for step in range(args.num_updates):
        t_step = time.perf_counter()

        # Decentralized Sampling: each rank samples independently from sharded buffer
        s_t, a_t, _, _, _, s_goal = buffer.sample(args.batch_size, device=device)

        # Forward pass under AMP autocast via model.__call__
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=torch.float16):
            # Sample diffusion noise and timesteps
            batch_sz = a_t.shape[0]
            timesteps = torch.randint(0, num_train_timesteps, (batch_sz,), device=device, dtype=torch.long)
            noise = torch.randn_like(a_t)

            sqrt_alpha_t = sqrt_alphas_cumprod[timesteps].view(-1, *([1] * (a_t.ndim - 1)))
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[timesteps].view(-1, *([1] * (a_t.ndim - 1)))
            noisy_actions = sqrt_alpha_t * a_t + sqrt_one_minus_alpha_t * noise

            # Model forward call triggers FSDP hooks
            pred_noise = model(noisy_actions, timesteps, s_t, s_goal)
            loss = F.mse_loss(pred_noise, noise)

        # Backward pass (FSDP overlaps reduce-scatter of gradients)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Rank 0 periodically broadcasts updated weights via non-blocking CUDA stream / IPC
        sync_lat_ms = 0.0
        if rank == 0 and step % 10 == 0:
            payload, sync_lat_ms = broadcaster.pack_and_broadcast_async(raw_model)
            sync_latencies.append(sync_lat_ms)

        if rank == 0 and step % 10 == 0:
            print(f"[FSDP Rank 0] Step {step:02d}/{args.num_updates} | Loss: {loss.item():.4f} | Sync Latency: {sync_lat_ms:.2f} ms")

    dist.barrier()
    total_elapsed = time.perf_counter() - start_time

    if rank == 0:
        updates_per_sec = args.num_updates / total_elapsed
        avg_sync_lat = np.mean(sync_latencies) if sync_latencies else 0.0

        print(f"\n========================================================")
        print(f"FSDP Training Engine Performance Telemetry")
        print(f"========================================================")
        print(f"Total Updates:          {args.num_updates}")
        print(f"Total Elapsed Time:     {total_elapsed:.2f} seconds")
        print(f"Training Velocity:      {updates_per_sec:.2f} updates/second")
        print(f"Policy Parameters:      {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Avg Weight Sync Latency: {avg_sync_lat:.3f} ms (Zero-Copy GPU IPC)")
        print(f"========================================================\n")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()