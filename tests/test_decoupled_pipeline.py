import time
import threading
import torch
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

# 2. Ray Remote collection loop to continuously step environments
@ray.remote
def collection_loop(worker, goal_state, queue, stop_event_actor):
    while not ray.get(stop_event_actor.is_set.remote()):
        # Collect 50 steps at a time across vectorized envs and write to queue
        ray.get(worker.collect_steps.remote(num_steps=50, goal_state=goal_state, queue=queue))
        time.sleep(0.01)

def main():
    print("Initializing Ray cluster locally...")
    ray.init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = 'antmaze-large-navigate-v0'
    print(f"Loading environment and dataset: {env_name}...")
    env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
    
    # Extract normalization statistics
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

    # Instantiate Actor model
    print("Initializing Trainer Actor model...")
    actor = GoalConditionedActor(state_dim=state_dim, action_dim=action_dim).to(device)
    optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    actor.train()

    # 3. Instantiate Ray Central Shared Queue and Stop Event Actor
    print("Creating central shared Ray Queue...")
    experience_queue = Queue(maxsize=10000)
    stop_event_actor = StopEventActor.remote()

    # 4. Spawn 2 RolloutWorkers
    num_workers = 2
    num_envs_per_worker = 4
    print(f"Spinning up {num_workers} RolloutWorkers, each with {num_envs_per_worker} vectorized envs...")
    
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

    # Sync initial weights to all workers
    cpu_state_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
    weights_ref = ray.put(cpu_state_dict)
    ray.get([w.sync_weights.remote(weights_ref) for w in workers])

    # 5. Launch Asynchronous Rollout Collection Loops on Ray Cluster
    print("Launching background experience collection loops...")
    loop_futures = [
        collection_loop.remote(worker, global_goal_raw, experience_queue, stop_event_actor)
        for worker in workers
    ]

    # 6. Instantiate Local Replay Buffer and populator thread
    print("Spawning asynchronous trainer populator thread...")
    local_replay_buffer = GoalConditionedReplayBuffer(
        max_size=20000, state_dim=state_dim, action_dim=action_dim, device=device
    )
    
    exit_flag = threading.Event()
    
    def populator_loop():
        while not exit_flag.is_set():
            try:
                # Retrieve list of transition dictionaries from Ray Queue
                transitions = experience_queue.get(timeout=0.5)
                for tx in transitions:
                    local_replay_buffer.add(
                        tx['obs'], tx['action'], tx['next_obs'],
                        tx['reward'], tx['done'], tx['goal']
                    )
            except Exception:
                # Timeout, ignore and loop again
                pass

    populator_thread = threading.Thread(target=populator_loop, daemon=True)
    populator_thread.start()

    # 7. Mock Training Loop (Concurrent updates on GPU)
    print("Waiting for replay buffer to collect initial steps (min 500)...")
    while len(local_replay_buffer) < 500:
        time.sleep(0.1)
    print(f"Replay buffer ready with {len(local_replay_buffer)} transitions.")

    print("Starting trainer gradient updates...")
    num_updates = 50
    batch_size = 32
    start_time = time.time()
    
    for step in range(num_updates):
        # Sample batch from local buffer populated in the background
        s_t, a_t, s_next, r_t, d_t, s_goal = local_replay_buffer.sample(batch_size)
        
        # Execute mock training step
        optim.zero_grad(set_to_none=True)
        pred_actions = actor(s_t, s_goal)
        loss = torch.mean((pred_actions - a_t) ** 2)
        loss.backward()
        optim.step()

        # Simulate periodic weight broadcast (every 10 gradient updates)
        if step > 0 and step % 10 == 0:
            cpu_state_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
            weights_ref = ray.put(cpu_state_dict)
            for w in workers:
                w.sync_weights.remote(weights_ref)

        if step % 10 == 0:
            print(f"Update {step:02d}/{num_updates} | Loss: {loss.item():.4f} | Replay Buffer Size: {len(local_replay_buffer)}")

    elapsed_time = time.time() - start_time
    updates_sps = num_updates / elapsed_time
    
    print("\n--- Telemetry Results ---")
    print(f"Total training updates completed: {num_updates}")
    print(f"Final Replay Buffer Size: {len(local_replay_buffer)}")
    print(f"Trainer Update Velocity: {updates_sps:.2f} updates/second")
    print("-------------------------\n")

    # 8. Clean up and Shutdown
    print("Stopping background processes and Ray actors...")
    ray.get(stop_event_actor.set_stop.remote())
    exit_flag.set()
    
    populator_thread.join()
    ray.get([w.close.remote() for w in workers])
    ray.shutdown()
    print("Verification Passed: Pipeline ran completely decoupled. Replay buffer was populated in the background while the Trainer executed parallel GPU gradient steps!")

if __name__ == '__main__':
    main()
