import torch
import numpy as np
import ogbench
from actor_modeling import GoalConditionedActor
from replay_buffer import GoalConditionedReplayBuffer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset and Environment
    env_name = 'antmaze-large-navigate-v0'
    print(f"Loading environment and dataset: {env_name}...")
    env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
    
    # Extract normalization statistics
    obs_raw = torch.from_numpy(train_dataset['observations'])
    obs_mean = obs_raw.mean(dim=0).to(device)
    obs_std = obs_raw.std(dim=0).to(device)
    obs_std[obs_std == 0] = 1.0
    
    state_dim = obs_raw.shape[-1]
    action_dim = train_dataset['actions'].shape[-1]
    print(f"Loaded successfully. State Dim: {state_dim}, Action Dim: {action_dim}")

    # 2. Instantiate and Load Pretrained Actor
    actor = GoalConditionedActor(state_dim=state_dim, action_dim=action_dim).to(device)
    checkpoint_path = f"{env_name}-actor_checkpoint.pt"
    print(f"Loading actor checkpoint from {checkpoint_path}...")
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    actor.eval()

    # 3. Create Goal-Conditioned Replay Buffer
    buffer_size = 10000
    replay_buffer = GoalConditionedReplayBuffer(
        max_size=buffer_size, state_dim=state_dim, action_dim=action_dim, device=device
    )

    # 4. Find Goal Observation State matching Target Goal coordinates
    # Check what goal attributes are present on the unwrapped env
    if hasattr(env.unwrapped, 'goal'):
        target_xy = env.unwrapped.goal
    elif hasattr(env.unwrapped, '_goal'):
        target_xy = env.unwrapped._goal
    else:
        target_xy = np.array([16.0, 16.0])
    
    print(f"Target logical goal coordinates: {target_xy}")
    
    # Map target coordinates to the closest dataset state to retrieve a full 29D goal state
    all_obs_flat = obs_raw.to(device)
    all_xy = all_obs_flat[:, :2]
    target_xy_tensor = torch.tensor(target_xy, dtype=torch.float32, device=device)
    dist_to_all = torch.norm(all_xy - target_xy_tensor, dim=1)
    best_match_idx = dist_to_all.argmin()
    
    global_goal_raw = all_obs_flat[best_match_idx].cpu().numpy()
    
    # 5. Online Rollout Loop
    print("Starting online rollouts...")
    num_episodes = 5
    max_steps = 100
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Normalize observation before feeding to policy
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            obs_norm = (obs_tensor - obs_mean) / obs_std
            
            # Goal is also normalized
            goal_tensor = torch.tensor(global_goal_raw, dtype=torch.float32, device=device)
            goal_norm = (goal_tensor - obs_mean) / obs_std
            
            # Forward pass
            with torch.no_grad():
                action = actor(obs_norm.unsqueeze(0), goal_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            
            # Step simulator
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store raw transition in the replay buffer
            replay_buffer.add(obs, action, next_obs, reward, done, global_goal_raw)
            
            obs = next_obs
            if done:
                break
                
        print(f"Episode {ep + 1}/{num_episodes} finished | Steps: {step + 1} | Reward: {episode_reward:.2f}")

    print(f"Replay Buffer Size: {len(replay_buffer)}")

    # 6. Verify Sampling and Shape correctness
    batch_size = 32
    print(f"Sampling batch of size {batch_size}...")
    batch = replay_buffer.sample(batch_size)
    
    s_t, a_t, s_next, r_t, d_t, s_goal = batch
    
    # Assertions to verify correctness
    assert s_t.shape == (batch_size, state_dim), f"Obs shape mismatch: {s_t.shape}"
    assert a_t.shape == (batch_size, action_dim), f"Action shape mismatch: {a_t.shape}"
    assert s_next.shape == (batch_size, state_dim), f"Next obs shape mismatch: {s_next.shape}"
    assert r_t.shape == (batch_size, 1), f"Reward shape mismatch: {r_t.shape}"
    assert d_t.shape == (batch_size, 1), f"Done shape mismatch: {d_t.shape}"
    assert s_goal.shape == (batch_size, state_dim), f"Goal shape mismatch: {s_goal.shape}"
    
    assert s_t.device.type == device.type, f"Device mismatch: {s_t.device} vs {device}"
    
    print("Verification Passed: Replay buffer successfully stored and sampled all tensors with correct shapes and devices!")

if __name__ == '__main__':
    main()
