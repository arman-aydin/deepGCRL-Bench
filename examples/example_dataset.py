"""
Before you can sample temporarlly, you must parse these flat arrays
into distinct, isolated trajectories. You must use the dataset's terminals
or timeouts arrays to find where one episode ends and the next begins.
"""

import numpy as np
import ogbench

# 1. Load the flat dataset from OGBench
# This pulls down the massive flat arrays into your environment
dataset_name = 'pointmaze-large-navigate-v0'
# dataset_name = 'humanoidmaze-large-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

# This will print out exactly what arrays are available 
# (e.g., 'observations', 'actions', 'rewards', 'terminals', 'masks')
print(train_dataset.keys())

# Extract the relevant flat arrays
obs = train_dataset['observations']
print(f'obs[:5]: {obs[:5]}')
actions = train_dataset['actions']
print(f'actions[:5]: {actions[:5]}')
terminals = train_dataset['terminals']
print(f'terminals[:5]: {terminals[:5]}')
next_observations = train_dataset['next_observations']
print(f'next_observations[:5]: {next_observations[:5]}')

# print(f'obs[0] + actions[0] ?= next_observations[0]: {obs[0]} + {actions[0]} = {obs[0] + actions[0]} ?= {next_observations[0]}')

# np.where returns the exact array indices where the condition is True.
# We add + 1 because np.split cuts *before* the index, but we want the 
# terminal state included in the current trajectory, not the next one.
end_indices = np.where(terminals)[0] + 1

# Safety check: if the very last transition is a terminal, np.split will
# create an empty array at the end. We strip it off if necessary
if len(end_indices) > 0 and end_indices[-1] == len(obs):
    end_indices = end_indices[:-1]

# 3. Chunk the flat arrays
# np.split efficiently partitions the massive array at the specified indices
trajectories = np.array(np.split(obs, end_indices)[:100])
print(f'Total trajectories extracted: {len(trajectories)}')
trajectory_actions = np.array(np.split(actions, end_indices)[:100])
print(f'Total trajectory_actions extracted: {len(trajectory_actions)}')
print(f"Shape of the first trajectory_actions: {trajectory_actions[0].shape}")
trajectory_next_observations = np.array(np.split(next_observations, end_indices)[:100])
print(f'Total trajectory_next_observations extracted: {len(trajectory_next_observations)}')
# print(f'trajectories[0][0]: {trajectories[0][0]}, trajectories[0][1]: {trajectories[0][1]}, trajectory_next_observations[0][0]: {trajectory_next_observations[0][0]}')

# # Verification
# print(f"Total trajectories extracted: {len(trajectories)}")
# print(f"Shape of the first trajectory: {trajectories[0].shape}")
# print(f'First trajectory: {trajectories[0]}')
# print(f"Shape of the second trajectory: {trajectories[2].shape}")
# print(f'Second trajectory: {trajectories[2]}')
rng = np.random.default_rng()   # Initialize the new standard generator
batch=128
horizon = trajectories[0].shape[0]
# Sample a random integer t uniformly from the range [0, L-2]. This is your anchor index.
anchor_indices = rng.integers(low=0, high=horizon-1, size=(batch))
example_indices = rng.integers(low=0, high=trajectories.shape[0], size=(batch))
print(f'anchor_indices.shape: {anchor_indices.shape}')
anchor_indices = np.stack((example_indices, anchor_indices), axis=0)
print(f'anchor_indices.shape: {anchor_indices.shape}')
print(f'anchor_indices: {anchor_indices}')

print(f'trajectories.shape: {trajectories.shape}')
# anchor_states = trajectories[example_indices, anchor_indices]
anchor_states = trajectories[tuple(anchor_indices)]
print(f'anchor_states.shape: {anchor_states.shape}')
print(f'anchor_states: {anchor_states}')
future_goal_indices = rng.integers(low=anchor_indices+1, high=horizon, size=(batch))
future_goal_indices = np.stack((example_indices, future_goal_indices), axis=0)
future_goal_states = trajectories[tuple(future_goal_indices)] 


# print(f'trajectories[0].shape: {trajectories[0].shape}')
# you can't do this since action dim != observation dim (only state and next_state)
tuples = np.stack((anchor_states, future_goal_states), axis=1)
print(f'tuples.shape: {tuples.shape}')


print(f'anchor_states[0]: {anchor_states[0]}')
print(f'future_goal_states[0]: {future_goal_states[0]}')
print(f'tuples[0]: {tuples[0]}')
