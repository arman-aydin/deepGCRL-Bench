import time
import torch
import numpy as np
import ogbench
import ray
from actor_modeling import GoalConditionedActor
from ray_workers import RolloutWorker

def main():
    # 1. Initialize Ray cluster locally
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

    # Map target coordinates [16.0, 16.0] to a dataset goal state
    target_xy = np.array([16.0, 16.0])
    all_obs_flat = obs_raw.numpy()
    all_xy = all_obs_flat[:, :2]
    dist_to_all = np.linalg.norm(all_xy - target_xy, axis=1)
    best_match_idx = dist_to_all.argmin()
    global_goal_raw = all_obs_flat[best_match_idx]

    # Load policy model
    print("Loading pretrained policy model...")
    actor = GoalConditionedActor(state_dim=state_dim, action_dim=action_dim).to(device)
    checkpoint_path = f"{env_name}-actor_checkpoint.pt"
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    actor.eval()

    # 2. Instantiate Remote Rollout Workers (2 workers, each with 4 vectorized envs)
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

    # 3. Synchronize Actor Weights using Plasma Shared Object Store
    print("Putting actor weights into Ray Object Store and synchronizing workers...")
    # Move actor weights to CPU dict first for local worker compatibility
    cpu_state_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
    weights_ref = ray.put(cpu_state_dict)
    
    # Sync weights to all workers and wait for completion
    sync_futures = [worker.sync_weights.remote(weights_ref) for worker in workers]
    ray.get(sync_futures)
    print("Synchronization complete.")

    # 4. Trigger Asynchronous Rollout Collection
    steps_to_collect = 100
    print(f"Triggering asynchronous step collection ({steps_to_collect} steps per environment)...")
    
    start_time = time.time()
    rollout_futures = [
        worker.collect_steps.remote(num_steps=steps_to_collect, goal_state=global_goal_raw)
        for worker in workers
    ]
    
    # Wait for all rollout workers to return transitions
    results = ray.get(rollout_futures)
    elapsed_time = time.time() - start_time
    
    # Flattens experience list
    all_transitions = []
    for worker_transitions in results:
        all_transitions.extend(worker_transitions)
        
    total_steps_collected = len(all_transitions)
    sps = total_steps_collected / elapsed_time
    
    print("\n--- Telemetry Results ---")
    print(f"Total steps collected: {total_steps_collected}")
    print(f"Expected steps: {num_workers * num_envs_per_worker * steps_to_collect}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Throughput: {sps:.2f} Steps Per Second (SPS)")
    print("-------------------------\n")

    # 5. Assertions and Verifications
    expected_total_steps = num_workers * num_envs_per_worker * steps_to_collect
    assert total_steps_collected == expected_total_steps, \
        f"Steps mismatch! Expected {expected_total_steps}, but collected {total_steps_collected}"
        
    # Verify transition contents
    sample_tx = all_transitions[0]
    assert sample_tx['obs'].shape == (state_dim,), f"Obs shape mismatch: {sample_tx['obs'].shape}"
    assert sample_tx['action'].shape == (action_dim,), f"Action shape mismatch: {sample_tx['action'].shape}"
    assert sample_tx['next_obs'].shape == (state_dim,), f"Next obs shape mismatch: {sample_tx['next_obs'].shape}"
    assert isinstance(sample_tx['reward'], (float, np.float32, np.float64)), f"Reward type mismatch: {type(sample_tx['reward'])}"
    assert isinstance(sample_tx['done'], (bool, np.bool_)), f"Done type mismatch: {type(sample_tx['done'])}"
    assert sample_tx['goal'].shape == (state_dim,), f"Goal shape mismatch: {sample_tx['goal'].shape}"

    # Close workers
    print("Closing environment pools...")
    close_futures = [worker.close.remote() for worker in workers]
    ray.get(close_futures)
    
    # Shutdown Ray
    print("Shutting down Ray...")
    ray.shutdown()
    
    print("Verification Passed: Ray Rollout Workers successfully orchestrated, synchronized, and returned expected transitions!")

if __name__ == '__main__':
    main()
