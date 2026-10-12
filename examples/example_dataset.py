"""Example demonstrating trajectory parsing and goal-conditioned sampling from OGBench datasets."""

import numpy as np
import ogbench


def extract_trajectories(observations: np.ndarray, terminals: np.ndarray):
    """Split flat dataset transitions into distinct episodic trajectory arrays."""
    end_indices = np.where(terminals)[0] + 1
    if len(end_indices) > 0 and end_indices[-1] == len(observations):
        end_indices = end_indices[:-1]
    return np.split(observations, end_indices)


def sample_goal_conditioned_pairs(trajectories: list, batch_size: int = 128):
    """Sample state-goal transition pairs with temporal ordering."""
    rng = np.random.default_rng()
    traj_indices = rng.integers(low=0, high=len(trajectories), size=batch_size)

    anchor_states = []
    goal_states = []

    for idx in traj_indices:
        traj = trajectories[idx]
        t_len = len(traj)
        if t_len < 2:
            continue
        t_anchor = rng.integers(low=0, high=t_len - 1)
        t_goal = rng.integers(low=t_anchor + 1, high=t_len)
        anchor_states.append(traj[t_anchor])
        goal_states.append(traj[t_goal])

    return np.array(anchor_states), np.array(goal_states)


if __name__ == '__main__':
    dataset_name = 'pointmaze-large-navigate-v0'
    env, train_dataset, _ = ogbench.make_env_and_datasets(dataset_name)

    trajectories = extract_trajectories(
        train_dataset['observations'],
        train_dataset['terminals']
    )
    print(f"Extracted {len(trajectories)} trajectories from {dataset_name}.")

    anchors, goals = sample_goal_conditioned_pairs(trajectories, batch_size=128)
    print(f"Sampled {len(anchors)} state-goal transition pairs with shape {anchors.shape}.")
