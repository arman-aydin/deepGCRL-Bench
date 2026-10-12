import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import ogbench

from pipelined_async.diffusion_policy import GoalConditionedDiffusionTransformer
from pipelined_async.contrastive_critic import WangContrastiveCritic


def train_contrastive_diffusion(
    num_updates: int = 500,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: str = "cuda"
):
    print("=" * 75)
    print("TRAINING CONTRASTIVE-GUIDED DIFFUSION POLICY (Wang et al. Critic + DiT)")
    print("=" * 75)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    env_name = "antmaze-large-navigate-v0"

    # 1. Load Environment & Dataset
    env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
    obs_raw = torch.from_numpy(train_dataset['observations'])
    obs_mean = obs_raw.mean(dim=0).to(dev)
    obs_std = obs_raw.std(dim=0).to(dev)
    obs_std[obs_std == 0] = 1.0

    state_dim = obs_raw.shape[-1]
    action_dim = train_dataset['actions'].shape[-1]
    dataset_size = obs_raw.shape[0]

    # 2. Load Pre-trained Wang et al. 32-layer ResNet Critic
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(repo_root, "checkpoints/antmaze-large-navigate-v0-32-layers-contrastive_checkpoint.pt")
    critic = WangContrastiveCritic(checkpoint_path=ckpt_path, state_dim=state_dim, device=dev)

    # 3. Instantiate Diffusion Transformer Policy
    policy = GoalConditionedDiffusionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        dim=256,
        depth=4,
        num_heads=8
    ).to(dev)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    # Precompute diffusion schedule
    num_timesteps = policy.num_train_timesteps
    sqrt_alphas_cumprod = policy.sqrt_alphas_cumprod.to(dev)
    sqrt_one_minus_alphas_cumprod = policy.sqrt_one_minus_alphas_cumprod.to(dev)

    print(f"Policy Parameters: {policy.count_parameters():,}")
    print(f"Beginning {num_updates} Advantage-Weighted Diffusion Updates...")

    losses = []
    t0 = time.perf_counter()

    for step in range(1, num_updates + 1):
        # Sample random mini-batch from offline dataset
        idx = np.random.randint(0, dataset_size, size=batch_size)
        s_t = torch.as_tensor(train_dataset['observations'][idx], device=dev, dtype=torch.float32)
        a_t = torch.as_tensor(train_dataset['actions'][idx], device=dev, dtype=torch.float32)
        s_next = torch.as_tensor(train_dataset['next_observations'][idx], device=dev, dtype=torch.float32)
        
        # Sample goals from future trajectories
        goal_idx = np.clip(idx + np.random.randint(10, 50, size=batch_size), 0, dataset_size - 1)
        s_goal = torch.as_tensor(train_dataset['observations'][goal_idx], device=dev, dtype=torch.float32)

        # 1. Evaluate Advantage via Wang et al. ResNet
        with torch.no_grad():
            weights = critic.compute_advantage(s_t, s_next, s_goal, alpha=0.75)

        # 2. Normalize inputs for Diffusion Actor
        s_t_norm = (s_t - obs_mean) / obs_std
        s_goal_norm = (s_goal - obs_mean) / obs_std

        # 3. Diffusion Denoising Step
        timesteps = torch.randint(0, num_timesteps, (batch_size,), device=dev, dtype=torch.long)
        noise = torch.randn_like(a_t)

        sqrt_alpha_t = sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        noisy_actions = sqrt_alpha_t * a_t + sqrt_one_minus_alpha * noise

        optimizer.zero_grad(set_to_none=True)
        pred_noise = policy(noisy_actions, timesteps, s_t_norm, s_goal_norm)

        # Weighted MSE loss: high advantage transitions get prioritized
        loss_per_sample = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=-1)
        loss = (weights * loss_per_sample).mean()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0 or step == num_updates:
            print(f"Step [{step:04d}/{num_updates:04d}] | Loss: {np.mean(losses[-50:]):.4f} | Avg Adv Weight: {weights.mean().item():.2f}")

    train_elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {train_elapsed:.2f} s ({num_updates / train_elapsed:.1f} updates/s)")
    return policy, critic, obs_mean, obs_std


def evaluate_in_simulation(
    policy,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    num_episodes: int = 5,
    max_steps: int = 300,
    save_gif: bool = True,
    device: str = "cuda"
):
    print("=" * 75)
    print("EVALUATING POLICY IN ANTMZE SIMULATION (Goal Reachability & Trajectories)")
    print("=" * 75)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    policy.eval()

    env, _, _ = ogbench.make_env_and_datasets("antmaze-large-navigate-v0")

    successes = []
    final_distances = []
    recorded_frames = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        goal = info['goal']
        goal_xy = goal[:2]
        
        goal_tensor = torch.tensor(goal, dtype=torch.float32, device=dev)
        goal_norm = (goal_tensor - obs_mean) / obs_std

        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=dev)
            obs_norm = (obs_tensor - obs_mean) / obs_std

            # Fast 5-step DDIM inference
            with torch.no_grad():
                action_gpu = policy.sample_actions(
                    obs_norm.unsqueeze(0),
                    goal_norm.unsqueeze(0),
                    num_inference_steps=5,
                    horizon=1
                )
                action = action_gpu.squeeze(0).squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            # Record frames for the first episode
            if save_gif and ep == 1 and step % 2 == 0:
                frame = env.render()
                if frame is not None:
                    recorded_frames.append(frame)

            dist_to_goal = np.linalg.norm(next_obs[:2] - goal_xy)
            if dist_to_goal < 1.5:
                success = True
                break

            obs = next_obs
            if terminated or truncated:
                break

        final_dist = np.linalg.norm(obs[:2] - goal_xy)
        successes.append(float(success))
        final_distances.append(final_dist)

        print(f"Episode {ep:02d}/{num_episodes:02d} | Success: {'Reached' if success else 'Incomplete'} | Steps: {step+1:03d} | Final Dist: {final_dist:.2f}")

    success_rate = np.mean(successes) * 100.0
    avg_dist = np.mean(final_distances)

    print("\n" + "=" * 75)
    print("SIMULATION EVALUATION SUMMARY")
    print(f"Goal Reached Success Rate: {success_rate:.1f}%")
    print(f"Average Final Distance:     {avg_dist:.2f} units")
    print("=" * 75)

    if save_gif and recorded_frames:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gif_path = os.path.join(repo_root, "assets/antmaze_diffusion_rollout.gif")
        imageio.mimsave(gif_path, recorded_frames, fps=20, loop=0)
        print(f"Rollout animation saved to: {gif_path}")

    return {
        "success_rate": success_rate,
        "avg_final_distance": avg_dist
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=300, help="Number of training steps")
    parser.add_argument("--episodes", type=int, default=3, help="Simulation evaluation episodes")
    parser.add_argument("--no-gif", action="store_true", help="Skip saving gif")
    args = parser.parse_args()

    policy, critic, obs_mean, obs_std = train_contrastive_diffusion(num_updates=args.updates)
    evaluate_in_simulation(policy, obs_mean, obs_std, num_episodes=args.episodes, save_gif=not args.no_gif)


if __name__ == '__main__':
    main()
