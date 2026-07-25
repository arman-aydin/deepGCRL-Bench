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
        'x_min': -5.0, 'x_max': 40.0,
        'y_min': -5.0, 'y_max': 30.0,
        'target_coordinate': np.array([[20.0, 12.5]]),
    },
    'humanoidmaze-large-navigate-v0': {
        'x_min': -5.0, 'x_max': 40.0,
        'y_min': -5.0, 'y_max': 30.0,
        'target_coordinate': np.array([[12.5, 16.0]]),
    }
}
vis_config = VIS_CONFIGS[dataset_name]
checkpoint_path = f"{dataset_name}-32-layers-contrastive_checkpoint.pt"

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
# 3. Extract Real Dataset States (For the Blueprint Overlay)
# ==========================================
print("Extracting physical dataset states for corridor overlay...")
all_obs_raw = dataset.observations.view(-1, state_dim).to(device)
all_obs_normalized = (all_obs_raw - obs_mean) / obs_std

# Extract true X, Y coordinates to paint the corridors
true_x = all_obs_raw[:, 0].cpu().numpy()
true_y = all_obs_raw[:, 1].cpu().numpy()

# ==========================================
# 4. Define Target Goal
# ==========================================
target_coordinate = vis_config['target_coordinate']
target_xy_tensor = torch.tensor(target_coordinate, dtype=torch.float32, device=device)

# Find the closest real state to act as our anchor goal
distances_to_target = torch.norm(all_obs_raw[:, :2] - target_xy_tensor, dim=1)
best_target_idx = distances_to_target.argmin()

target_state_raw = all_obs_raw[best_target_idx]
target_state_normalized = all_obs_normalized[best_target_idx].unsqueeze(0)
print(f"Mapped visual goal to real dataset state! Error: {distances_to_target[best_target_idx]:.2f}m")

# ==========================================
# 5. Build the Dense Continuous Grid
# ==========================================
print("Building continuous 2D evaluation grid...")
x_min, x_max = vis_config['x_min'], vis_config['x_max']
y_min, y_max = vis_config['y_min'], vis_config['y_max']
grid_resolution = 500 # 500x500 = 250,000 points

x_coords = np.linspace(x_min, x_max, grid_resolution)
y_coords = np.linspace(y_min, y_max, grid_resolution)
xx, yy = np.meshgrid(x_coords, y_coords)

grid_points = np.c_[xx.ravel(), yy.ravel()] 
mean_joints_raw = obs_mean[2:].cpu().numpy()
neutral_joints = np.tile(mean_joints_raw, (grid_points.shape[0], 1))

grid_states_raw = np.concatenate([grid_points, neutral_joints], axis=1)
grid_tensor_raw = torch.tensor(grid_states_raw, dtype=torch.float32).to(device)
grid_tensor_normalized = (grid_tensor_raw - obs_mean) / obs_std

# ==========================================
# 6. Chunked Latent Reachability (Protecting VRAM)
# ==========================================
print("Computing latent reachability across the entire map...")
batch_size_inf = 20000 
z_grid_list = []

with torch.no_grad():
    z_goal = goal_proj(model(target_state_normalized))
    z_goal = F.normalize(z_goal, p=2.0, dim=-1)
    
    num_grid_states = grid_tensor_normalized.shape[0]
    for i in range(0, num_grid_states, batch_size_inf):
        chunk = grid_tensor_normalized[i : i + batch_size_inf]
        z_chunk = model(chunk)
        z_chunk = F.normalize(z_chunk, p=2.0, dim=-1)
        z_grid_list.append(z_chunk)

    z_grid = torch.cat(z_grid_list, dim=0)
    similarities = (z_grid @ z_goal.T).squeeze().cpu().numpy()

# ==========================================
# 7. Render & Save the Plot
# ==========================================
print("Rendering the Geodesic Blueprint...")
heatmap = similarities.reshape(grid_resolution, grid_resolution)

plt.figure(figsize=(12, 10))

# LAYER 1: The Continuous Heatmap (The Model's Beliefs)
plt.contourf(xx, yy, heatmap, levels=50, cmap='viridis', zorder=1)
plt.colorbar(label='Cosine Similarity (Reachability)')

# LAYER 2: The Physical Corridors (The Ground Truth Reality)
# By plotting the dataset points in black with low alpha, they form solid black corridors.
# The spaces between the black corridors are the physical walls.
# plt.scatter(true_x, true_y, color='black', s=1, alpha=0.05, zorder=2, label='Valid Corridors')

# LAYER 3: The Target Goal
plt.scatter(target_state_raw[0].item(), target_state_raw[1].item(), 
            color='red', marker='X', s=300, edgecolors='white', linewidth=2, zorder=3, label='Target Goal')

plt.title(f"{state_dim}D {dataset_name.split('-')[0].capitalize()}Maze: Geodesic vs Euclidean Flow", fontsize=16)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend(loc="upper left")

plt.gca().set_aspect('equal', adjustable='box')

output_filename = f"{dataset_name.split('-')[0]}_blueprint_reachability.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Heatmap successfully saved to {output_filename}")

# # ==========================================
# # 3. Extract and Normalize Dataset States
# # ==========================================
# print("Extracting actual physical dataset states...")
# # Shape: [Total_Steps, 29]
# all_obs_raw = dataset.observations.view(-1, state_dim).to(device)
# all_obs_normalized = (all_obs_raw - obs_mean) / obs_std

# # Extract true X, Y coordinates for plotting
# true_x = all_obs_raw[:, 0].cpu().numpy()
# true_y = all_obs_raw[:, 1].cpu().numpy()

# # ==========================================
# # 4. Find the Real Target Goal
# # ==========================================
# target_coordinate = vis_config['target_coordinate']
# target_xy_tensor = torch.tensor(target_coordinate, dtype=torch.float32, device=device)

# print(f"Finding closest real state to target {target_coordinate}...")
# # Calculate distances purely in the 2D plane
# distances_to_target = torch.norm(all_obs_raw[:, :2] - target_xy_tensor, dim=1)
# best_target_idx = distances_to_target.argmin()

# # Select the real dataset state
# target_state_raw = all_obs_raw[best_target_idx]
# target_state_normalized = all_obs_normalized[best_target_idx].unsqueeze(0)

# print(f"Mapped visual goal to real dataset state! Error: {distances_to_target[best_target_idx]:.2f}m")

# # ==========================================
# # 5. Chunked Latent Reachability (Protecting VRAM)
# # ==========================================
# print("Computing latent reachability (chunked to prevent CUDA OOM)...")
# batch_size_inf = 20000 # Process 20k states at a time
# z_grid_list = []

# with torch.no_grad():
#     # Embed the Goal (Using the asymmetric projection head)
#     z_goal = goal_proj(model(target_state_normalized))
#     z_goal = F.normalize(z_goal, p=2.0, dim=-1)
    
#     # Chunk through the 1M dataset points
#     num_states = all_obs_normalized.shape[0]
#     for i in range(0, num_states, batch_size_inf):
#         chunk = all_obs_normalized[i : i + batch_size_inf]
#         z_chunk = model(chunk)
#         z_chunk = F.normalize(z_chunk, p=2.0, dim=-1)
#         z_grid_list.append(z_chunk)
        
#         if (i // batch_size_inf) % 10 == 0:
#             print(f"Processed {min(i + batch_size_inf, num_states)} / {num_states} states...")

#     # Concatenate all embeddings
#     z_grid = torch.cat(z_grid_list, dim=0)
    
#     # Compute dot product (Reachability) for all 1M states instantly
#     similarities = (z_grid @ z_goal.T).squeeze().cpu().numpy()

# # ==========================================
# # 6. Render & Save the Plot
# # ==========================================
# print("Rendering the highly-dense scatter heatmap...")
# plt.figure(figsize=(12, 10))

# # Plot the dense dataset points, colored by their latent similarity to the goal!
# scatter = plt.scatter(true_x, true_y, c=similarities, cmap='viridis', s=2, alpha=0.8)
# plt.colorbar(scatter, label='Cosine Similarity (Reachability)')

# # Plot the Target Goal as a giant red X
# plt.scatter(target_state_raw[0].item(), target_state_raw[1].item(), 
#             color='red', marker='X', s=300, edgecolors='black', linewidth=2, zorder=5, label='Target Goal')

# plt.title(f"{state_dim}D {dataset_name.split('-')[0].capitalize()}Maze: Latent Geodesic Reachability", fontsize=16)
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.legend(loc="upper left")

# # Keep the aspect ratio square so the maze doesn't look stretched
# plt.gca().set_aspect('equal', adjustable='box')

# output_filename = f"{dataset_name.split('-')[0]}_reachability_crossroads.png"
# plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"Heatmap successfully saved to {output_filename}")