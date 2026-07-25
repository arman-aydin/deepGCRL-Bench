import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Import your model classes and dataset
from resnet_modelling import ResNetConfigDepth
from dataset_loader import OGBenchCustomBatchedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 0. Configuration
# ==========================================
# Select the environment to visualize
# dataset_name = 'humanoidmaze-large-navigate-v0'
dataset_name = 'antmaze-large-navigate-v0'

VIS_CONFIGS = {
    'antmaze-large-navigate-v0': {
        'x_min': -5.0, 'x_max': 30.0,
        'y_min': -5.0, 'y_max': 25.0,
        'target_coordinate': np.array([[0.0, 16.0]]),
    },
    'humanoidmaze-large-navigate-v0': {
        'x_min': -5.0, 'x_max': 40.0,
        'y_min': -5.0, 'y_max': 30.0,
        'target_coordinate': np.array([[12.5, 16.0]]),
    }
}
vis_config = VIS_CONFIGS[dataset_name]
checkpoint_path = f"{dataset_name}-contrastive_checkpoint.pt"

# ==========================================
# 1. Load Dataset for Normalization & Layout
# ==========================================
print("Loading Dataset to extract Normalization Constants and Maze Layout...")
dataset = OGBenchCustomBatchedDataset(dataset_name=dataset_name, batch_size=128)

# Move mean and std to the correct device
obs_mean = dataset.obs_mean.to(device)
obs_std = dataset.obs_std.to(device)
state_dim = dataset.observations.shape[-1]

# ==========================================
# 2. Re-initialize & Load Models
# ==========================================
print(f"Loading checkpoint from {checkpoint_path}...")
model = ResNetConfigDepth(depth=32, state_dim=state_dim, hidden_dim=256).to(device)

goal_proj = nn.Sequential(
    nn.Linear(256, 256),
    nn.LayerNorm(256),
    nn.SiLU(), 
    nn.Linear(256, 256)
).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
goal_proj.load_state_dict(checkpoint['goal_proj_state_dict'])

model.eval()
goal_proj.eval()

# ==========================================
# 3. Build & Normalize the Grid
# ==========================================
# Define maze bounds
x_min, x_max = vis_config['x_min'], vis_config['x_max']
y_min, y_max = vis_config['y_min'], vis_config['y_max']
grid_resolution = 1000

# Create the X-Y grid
x_coords = np.linspace(x_min, x_max, grid_resolution)
y_coords = np.linspace(y_min, y_max, grid_resolution)
xx, yy = np.meshgrid(x_coords, y_coords)

# Flatten the grid (Shape: [10000, 2])
grid_points = np.c_[xx.ravel(), yy.ravel()] 

# THE FIX: Extract the actual mean for the 67 joint dimensions
mean_joints_raw = obs_mean[2:].cpu().numpy()

# Tile this mean vector for every single point in the 10,000 point grid
neutral_joints = np.tile(mean_joints_raw, (grid_points.shape[0], 1))

# Concatenate the X,Y coordinates with the average joint values
grid_states_raw = np.concatenate([grid_points, neutral_joints], axis=1)

# Convert to tensor and NORMALIZE
grid_tensor_raw = torch.tensor(grid_states_raw, dtype=torch.float32).to(device)
grid_tensor_normalized = (grid_tensor_raw - obs_mean) / obs_std

# ==========================================
# 4. Define & Normalize the Target Goal
# ==========================================
target_coordinate = vis_config['target_coordinate']

# THE FIX: Pad the target goal with the mean joints as well
target_state_raw = np.concatenate([target_coordinate, mean_joints_raw.reshape(1, -1)], axis=1)

target_tensor_raw = torch.tensor(target_state_raw, dtype=torch.float32).to(device)
target_tensor_normalized = (target_tensor_raw - obs_mean) / obs_std

# ==========================================
# 5. Compute Latent Reachability
# ==========================================
print("Computing latent reachability...")
with torch.no_grad():
    # Embed the Goal (Using the target projection head)
    z_goal = goal_proj(model(target_tensor_normalized))
    z_goal = F.normalize(z_goal, p=2.0, dim=-1)
    
    # Embed the massive grid of Anchor States
    z_grid = model(grid_tensor_normalized)
    z_grid = F.normalize(z_grid, p=2.0, dim=-1)
    
    # Compute reachability (Cosine Similarity)
    similarities = (z_grid @ z_goal.T).squeeze() 

# ==========================================
# 6. Extract Visited Corridors for Wall Overlay
# ==========================================
print("Extracting visited coordinates to map the walls...")
# Flatten the [Num_Trajectories, Horizon, Dim] into [Total_Steps, Dim]
flat_observations = dataset.observations.view(-1, state_dim)

# We want the RAW (unnormalized) coordinates for plotting on our physical grid
visited_x = flat_observations[:, 0].numpy()
visited_y = flat_observations[:, 1].numpy()

# ==========================================
# 7. Render & Save the Plot
# ==========================================
heatmap = similarities.cpu().numpy().reshape(grid_resolution, grid_resolution)

plt.figure(figsize=(10, 8))

# 7a. Plot the Model's Representation Heatmap
plt.contourf(xx, yy, heatmap, levels=50, cmap='viridis')
plt.colorbar(label='Cosine Similarity (Reachability)')

# 7b. Plot the Physical Maze Walls
# The alpha is very low so the massive density of points forms a soft shadow, leaving the walls completely blank
# plt.scatter(visited_x, visited_y, color='black', s=1, alpha=0.01, zorder=1, label='Corridors (Dataset)')

# 7c. Plot the Target Goal
plt.scatter(target_state_raw[0, 0], target_state_raw[0, 1], 
            color='red', marker='X', s=200, zorder=2, label='Target Goal')

plt.title(f"{state_dim}D {dataset_name.split('-')[0].capitalize()}Maze: Latent Reachability")
plt.legend(loc="upper left")

# Save the plot
output_filename = f"{dataset_name.split('-')[0]}_reachability.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Heatmap successfully saved to {output_filename}")