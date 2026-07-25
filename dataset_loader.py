import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable
import ogbench

class OGBenchCustomBatchedDataset(Dataset):
    """Load OGBench datasets as (s0, a0, s1) tuples"""

    def __init__(self, dataset_name: str, batch_size: int = 128,
                 transform: Callable | None = None):
        self.dataset_name = dataset_name
        self.transform = transform
        self.batch_size = batch_size
        self.horizon = 1000  # Known fixed horizon for OGBench pointmaze/humanoidmaze
        
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)
        
        obs = torch.from_numpy(train_dataset['observations'])
        actions = torch.from_numpy(train_dataset['actions'])

        self.actions = actions.view(-1, self.horizon, actions.shape[-1])

        # Normalize observations to have zero mean and unit variance. This is crucial
        # for stable neural network training.
        self.obs_mean = obs.mean(dim=0)
        self.obs_std = obs.std(dim=0)
        self.obs_std[self.obs_std == 0] = 1.0 # Avoid division by zero for constant features
        
        # Reshape flat array (Total_Steps, Dim) into (Num_Trajectories, Horizon, Dim)
        self.observations = obs.view(-1, self.horizon, obs.shape[-1])
        self.n_trajectories = self.observations.shape[0]
        
        print(f'Total trajectories extracted: {self.n_trajectories}')

    def __len__(self):
        # Because __getitem__ returns a full batch randomly, __len__ dictates 
        # how many batches constitute one "epoch" in your training loop.
        return (self.n_trajectories * self.horizon) // self.batch_size

    def __getitem__(self, idx):
        # 1. Sample random trajectories
        traj_indices = torch.randint(low=0, high=self.n_trajectories, size=(self.batch_size,))
        
        # 2. Sample random anchor indices: [0, horizon - 2]
        anchor_indices = torch.randint(low=0, high=self.horizon - 1, size=(self.batch_size,))
        
        # 3. Vectorized sampling for future goals: local temporal window
        # Cap the future sampling to a local temporal window (e.g., max 40 steps)
        max_step_window = 40
        
        # We generate random floats (0 to 1), scale them by the remaining valid steps, and add to the anchor
        valid_ranges = torch.clamp(self.horizon - anchor_indices - 1, max=max_step_window)
        random_offsets = (torch.rand(self.batch_size) * valid_ranges).long()
        future_goal_indices = anchor_indices + 1 + random_offsets

        # 4. Advanced indexing to grab the states
        anchor_states_raw = self.observations[traj_indices, anchor_indices]
        future_goal_states_raw = self.observations[traj_indices, future_goal_indices]

        anchor_actions = self.actions[traj_indices, anchor_indices]
        next_states_raw = self.observations[traj_indices, anchor_indices + 1]

        # --- Debugging Prints ---
        # We'll check the first batch of an epoch to see what's happening.
        if idx == 0:
            print("\n--- Dataloader __getitem__ Debug (for first batch of epoch) ---")
            print(f"Sampled anchor_indices (first 5): {anchor_indices[:5].tolist()}")
            print(f"Sampled future_goal_indices (first 5): {future_goal_indices[:5].tolist()}")

            # Verification check
            assert torch.all(future_goal_indices > anchor_indices), "Error: Not all future goals are after anchors!"
            print("Assertion Passed: All future_goal_indices > anchor_indices.")

            print(f"\nRaw anchor_states (first 2): \n{anchor_states_raw[:2]}")
            print(f"Raw future_goal_states (first 2): \n{future_goal_states_raw[:2]}")

        # Apply normalization before returning the batch
        # We move the tensors to the main dataset device (CPU) before normalization
        anchor_states_normalized = (anchor_states_raw - self.obs_mean) / self.obs_std
        future_goal_states_normalized = (future_goal_states_raw - self.obs_mean) / self.obs_std

        next_states_normalized = (next_states_raw - self.obs_mean) / self.obs_std

        if idx == 0:
            print(f"\nNormalized anchor_states (first 2): \n{anchor_states_normalized[:2]}")
            print(f"Normalized future_goal_states (first 2): \n{future_goal_states_normalized[:2]}")
            print("--- End Dataloader Debug ---\n")

        # Returns a tuple of (128, D) tensors
        return anchor_states_normalized, anchor_actions, next_states_normalized, future_goal_states_normalized


class OGBenchDataset(Dataset):
    def __init__(self, dataset_name: str, transform: Callable | None = None):
        self.dataset_name = dataset_name
        self.transform = transform


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

        

        
    

if __name__ == '__main__':
    pass
