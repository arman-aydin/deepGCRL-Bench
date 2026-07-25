import os
import time
import threading
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import ogbench
import ray
from ray.util.queue import Queue

from actor_modeling import GoalConditionedActor
from ray_workers import RolloutWorker
from replay_buffer import GoalConditionedReplayBuffer

# 1. Coordinate Stop Signals across Ray Processes
@ray.remote
class StopEventActor:
    def __init__(self):
        self.stop = False
    def set_stop(self):
        self.stop = True
    def is_set(self):
        return self.stop

# 2. Ray Remote collection loop
@ray.remote
def collection_loop(worker, goal_state, queue, stop_event_actor):
    while not ray.get(stop_event_actor.is_set.remote()):
        ray.get(worker.collect_steps.remote(num_steps=50, goal_state=goal_state, queue=queue))
        time.sleep(0.01)

def main():
    # 3. Initialize PyTorch DDP
    # Get environment variables set by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Determine backend: Use NCCL for multi-GPU, Gloo for CPU/local single-GPU simulation
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        # Fallback to Gloo for single GPU or CPU machines
        backend = "gloo"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Initialized DDP process group with backend '{backend}' on device '{device}'.")

    # 4. Extract environment parameters (shared constants across ranks)
    env_name = 'antmaze-large-navigate-v0'
    env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
    obs_raw = torch.from_numpy(train_dataset['observations'])
    obs_mean = obs_raw.mean(dim=0).numpy()
    obs_std = obs_raw.std(dim=0).numpy()
    obs_std[obs_std == 0] = 1.0
    state_dim = obs_raw.shape[-1]
    action_dim = train_dataset['actions'].shape[-1]

    # Map target goal coordinates [16.0, 16.0]
    target_xy = np.array([16.0, 16.0])
    all_obs_flat = obs_raw.numpy()
    all_xy = all_obs_flat[:, :2]
    dist_to_all = np.linalg.norm(all_xy - target_xy, axis=1)
    best_match_idx = dist_to_all.argmin()
    global_goal_raw = all_obs_flat[best_match_idx]

    # 5. Initialize the Actor Model and wrap with DDP
    raw_actor = GoalConditionedActor(state_dim=state_dim, action_dim=action_dim).to(device)
    actor = DDP(raw_actor, device_ids=[local_rank] if backend == "nccl" else None)
    optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    actor.train()

    # 6. Rank 0 coordinates Ray Cluster and Replay Buffer populator thread
    if rank == 0:
        print("[Rank 0] Starting Ray session and worker actors...")
        ray.init()

        experience_queue = Queue(maxsize=10000)
        stop_event_actor = StopEventActor.remote()

        num_workers = 2
        num_envs_per_worker = 4
        workers = [
            RolloutWorker.remote(
                env_name=env_name,
                state_dim=state_dim,
                action_dim=action_dim,
                mean=obs_mean,
                std=obs_std,
                num_envs=num_envs_per_worker
            )
            for _ in range(num_workers)
        ]

        # Broadcast initial weights to Ray workers
        # DDP model weights are wrapped in actor.module
        cpu_state_dict = {k: v.cpu() for k, v in actor.module.state_dict().items()}
        weights_ref = ray.put(cpu_state_dict)
        ray.get([w.sync_weights.remote(weights_ref) for w in workers])

        # Launch Ray worker simulation loops
        loop_futures = [
            collection_loop.remote(worker, global_goal_raw, experience_queue, stop_event_actor)
            for worker in workers
        ]

        # Start trainer populator thread
        local_replay_buffer = GoalConditionedReplayBuffer(
            max_size=20000, state_dim=state_dim, action_dim=action_dim, device=device
        )
        
        exit_flag = threading.Event()
        
        def populator_loop():
            while not exit_flag.is_set():
                try:
                    transitions = experience_queue.get(timeout=0.5)
                    for tx in transitions:
                        local_replay_buffer.add(
                            tx['obs'], tx['action'], tx['next_obs'],
                            tx['reward'], tx['done'], tx['goal']
                        )
                except Exception:
                    pass

        populator_thread = threading.Thread(target=populator_loop, daemon=True)
        populator_thread.start()

        # Wait for buffer to collect initial steps
        print("[Rank 0] Waiting for buffer to collect initial steps...")
        while len(local_replay_buffer) < 500:
            time.sleep(0.1)
        print(f"[Rank 0] Replay buffer populated with {len(local_replay_buffer)} transitions.")
    else:
        # Other ranks do not initialize Ray or buffers
        local_replay_buffer = None
        workers = None
        loop_futures = None

    # Sync all DDP processes before starting the training loop
    dist.barrier()

    # 7. Decoupled DDP Training Loop
    num_updates = 50
    batch_size = 32 # size per rank
    
    # Pre-allocate tensor buffers for broadcast data matching total world_size batch
    s_t = torch.zeros((batch_size * world_size, state_dim), dtype=torch.float32, device=device)
    a_t = torch.zeros((batch_size * world_size, action_dim), dtype=torch.float32, device=device)
    s_next = torch.zeros((batch_size * world_size, state_dim), dtype=torch.float32, device=device)
    r_t = torch.zeros((batch_size * world_size, 1), dtype=torch.float32, device=device)
    d_t = torch.zeros((batch_size * world_size, 1), dtype=torch.float32, device=device)
    s_goal = torch.zeros((batch_size * world_size, state_dim), dtype=torch.float32, device=device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start_time = time.time()

    for step in range(num_updates):
        # Rank 0 samples batch and writes into pre-allocated tensors
        if rank == 0:
            sampled_s_t, sampled_a_t, sampled_s_next, sampled_r_t, sampled_d_t, sampled_s_goal = local_replay_buffer.sample(batch_size * world_size)
            s_t.copy_(sampled_s_t)
            a_t.copy_(sampled_a_t)
            s_next.copy_(sampled_s_next)
            r_t.copy_(sampled_r_t)
            d_t.copy_(sampled_d_t)
            s_goal.copy_(sampled_s_goal)

        # Broadcast the batch from Rank 0 to all other ranks
        dist.broadcast(s_t, src=0)
        dist.broadcast(a_t, src=0)
        dist.broadcast(s_next, src=0)
        dist.broadcast(r_t, src=0)
        dist.broadcast(d_t, src=0)
        dist.broadcast(s_goal, src=0)

        # Each rank slices its unique segment from the broadcast batch
        slice_start = rank * batch_size
        slice_end = (rank + 1) * batch_size

        s_t_rank = s_t[slice_start:slice_end]
        a_t_rank = a_t[slice_start:slice_end]
        s_goal_rank = s_goal[slice_start:slice_end]

        # Execute training forward pass under AMP autocast
        optim.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred_actions = actor(s_t_rank, s_goal_rank)
            loss = torch.mean((pred_actions - a_t_rank) ** 2)
        
        # Scale loss and perform backward pass and step with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        # Rank 0 periodically broadcasts updated actor weights to remote Ray workers
        if rank == 0 and step > 0 and step % 10 == 0:
            cpu_state_dict = {k: v.cpu() for k, v in actor.module.state_dict().items()}
            weights_ref = ray.put(cpu_state_dict)
            for w in workers:
                w.sync_weights.remote(weights_ref)

        if rank == 0 and step % 10 == 0:
            print(f"[Trainer] Update {step:02d}/{num_updates} | Loss: {loss.item():.4f} | Replay Buffer Size: {len(local_replay_buffer)}")

    # Sync ranks at the end of training
    dist.barrier()
    elapsed_time = time.time() - start_time

    if rank == 0:
        updates_sps = num_updates / elapsed_time
        print("\n--- DDP Telemetry Results ---")
        print(f"Total training updates completed: {num_updates}")
        print(f"Final Replay Buffer Size: {len(local_replay_buffer)}")
        print(f"Trainer Update Velocity: {updates_sps:.2f} updates/second (world_size={world_size})")
        print("-----------------------------\n")

        # Cleanup Ray
        print("[Rank 0] Shutting down background collection loops and Ray actors...")
        ray.get(stop_event_actor.set_stop.remote())
        exit_flag.set()
        pop_thread_done = populator_thread.join(timeout=5)
        
        ray.get([w.close.remote() for w in workers])
        ray.shutdown()
        print("[Rank 0] Ray cluster shutdown complete.")

    # Cleanup DDP Process Group
    dist.destroy_process_group()
    print(f"[Rank {rank}] Destroyed process group. Process exiting successfully.")

if __name__ == '__main__':
    main()
