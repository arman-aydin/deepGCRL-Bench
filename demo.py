import torch
import torch.nn as nn
import numpy as np
import imageio
import ogbench
import torch.nn.functional as F

import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

# Import your custom architectures
from resnet_modelling import ResNetConfigDepth
from actor_modeling import GoalConditionedActor
from dataset_loader import OGBenchCustomBatchedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Dataset to get Normalization Stats (CRITICAL)
print("Loading dataset stats...")
dataset_name = 'antmaze-large-navigate-v0'
# dataset_name = 'humanoidmaze-large-navigate-v0'
batch_size = 256
dataset = OGBenchCustomBatchedDataset(dataset_name=dataset_name, batch_size=batch_size)
obs_mean = dataset.obs_mean.to(device)
obs_std = dataset.obs_std.to(device)
action_dim = dataset.actions.shape[-1]

# 2. Initialize Models
actor = GoalConditionedActor(state_dim=29, action_dim=action_dim).to(device)
actor.load_state_dict(torch.load(f"{dataset_name}-actor_checkpoint.pt", map_location=device))
actor.eval()

checkpoint_path = f"{dataset_name}-contrastive_checkpoint.pt"

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)

model = ResNetConfigDepth(depth=32, state_dim=29, hidden_dim=256).to(device)

goal_proj = nn.Sequential(
    nn.Linear(256, 256),
    nn.LayerNorm(256),
    nn.SiLU(), 
    nn.Linear(256, 256)
).to(device)

# Load architecture weights
model.load_state_dict(checkpoint['model_state_dict'])
goal_proj.load_state_dict(checkpoint['goal_proj_state_dict'])
    

# ========================================================
# 3 & 4. Setup Live Environment, Waypoints & REAL Goal
# ========================================================
# ========================================================
# 3. Setup Live Environment & CUSTOM GOAL OVERRIDE
# ========================================================
env_name = 'antmaze-large-navigate-v0'
env, _, _ = ogbench.make_env_and_datasets(env_name)

obs, info = env.reset()

true_goal_xy = [16.0, 16.0] 
print(f"OVERRIDING TARGET! Custom Red Dot logically set to: X={true_goal_xy[0]:.2f}, Y={true_goal_xy[1]:.2f}")

# Overwrite the environment's internal logic pointers
if hasattr(env.unwrapped, 'goal'): env.unwrapped.goal = np.array(true_goal_xy)
if hasattr(env.unwrapped, 'target_goal'): env.unwrapped.target_goal = np.array(true_goal_xy)

# The MuJoCo Cache Flush
if hasattr(env.unwrapped, 'model') and hasattr(env.unwrapped, 'data'):
    try:
        import mujoco
        site_id = None
        # OGBench/D4RL usually names the goal site 'target' or 'goal'
        for name in ['target', 'goal']:
            try:
                site_id = env.unwrapped.model.site(name).id
                break
            except:
                try:
                    site_id = env.unwrapped.model.site_name2id(name)
                    break
                except:
                    pass
        
        if site_id is not None:
            # 1. Update the static base model
            env.unwrapped.model.site_pos[site_id][:2] = true_goal_xy
            
            # 2. THE FIX: Flush the kinematics cache so the renderer sees it!
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            print("MuJoCo kinematics cache flushed! Visual red dot successfully moved.")
    except Exception as e:
        print(f"Warning: Could not flush MuJoCo visualizer: {e}")

target_xy_tensor = torch.tensor(true_goal_xy[:2], dtype=torch.float32, device=device)

print("Searching entire dataset for exact goal match...")
all_obs_raw = dataset.observations.view(-1, 29).to(device)
all_xy = all_obs_raw[:, :2]

# Search all 1,000,000 states to find the perfect physical match
dist_to_all = torch.norm(all_xy - target_xy_tensor, dim=1)
best_overall_idx = dist_to_all.argmin()

global_goal_raw = all_obs_raw[best_overall_idx]
global_goal_normalized = ((global_goal_raw - obs_mean) / obs_std).unsqueeze(0)
print(f"Mapped Red Dot to REAL dataset state! Euclidean error: {dist_to_all[best_overall_idx]:.2f}m")

exact_target_raw = np.zeros(29)
exact_target_raw[0] = true_goal_xy[0]
exact_target_raw[1] = true_goal_xy[1]
exact_target_tensor = torch.tensor(exact_target_raw, dtype=torch.float32, device=device).unsqueeze(0)
exact_target_normalized = (exact_target_tensor - obs_mean) / obs_std

# ========================================================
# 4. Build Latent Waypoint Bank
# ========================================================
print("Sampling 8000 waypoints for Latent Graph...")
num_nodes = 10000
waypoint_indices = torch.randint(0, all_obs_raw.shape[0], (num_nodes,))
waypoints_raw = all_obs_raw[waypoint_indices]
waypoints_normalized = (waypoints_raw - obs_mean) / obs_std

# ========================================================
# 5. Live Physical Rollout (With Kinematics & Terminal Handoff)
# ========================================================
frames = []
max_steps = 1000
success = False

active_subgoal = None
subgoal_step_count = 0

print("Starting live physical rollout with SORB...")

# Pre-find the visual site ID so we don't slow down the loop
target_site_id = None
if hasattr(env.unwrapped, 'model'):
    for name in ['target', 'goal']:
        try:
            target_site_id = env.unwrapped.model.site(name).id
            break
        except:
            try:
                target_site_id = env.unwrapped.model.site_name2id(name)
                break
            except:
                pass

with torch.no_grad():
    for step in range(max_steps):
        
        # ---------------------------------------------------------
        # THE VISUAL HIJACK: Force the Red Dot to our coordinates EVERY frame
        # ---------------------------------------------------------
        if target_site_id is not None:
            env.unwrapped.model.site_pos[target_site_id][:2] = true_goal_xy
            # If using older mujoco-py, you might need: env.unwrapped.sim.model.site_pos...
        
        # Now take the picture!
        frames.append(env.render()) 
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        obs_normalized = (obs_tensor - obs_mean) / obs_std
        
        # Calculate real physical distance to the true Red Dot
        ant_xy = obs[:2]
        dist_to_real_goal = np.linalg.norm(ant_xy - true_goal_xy[:2])
        
        # ---------------------------------------------------------
        # THE TERMINAL HANDOFF (2.5 meters)
        # ---------------------------------------------------------
        if dist_to_real_goal < 2.5:
            active_subgoal = exact_target_normalized
            if step % 20 == 0: 
                print(f"TERMINAL HANDOFF! Direct line of sight. Distance: {dist_to_real_goal:.2f}m")
        else:
            need_new_waypoint = False
            
            if active_subgoal is None:
                need_new_waypoint = True
            else:
                z_current = F.normalize(model(obs_normalized), p=2.0, dim=-1)
                z_active = F.normalize(model(active_subgoal), p=2.0, dim=-1)
                if (z_current @ z_active.T).squeeze() > 0.60: 
                    need_new_waypoint = True
                elif subgoal_step_count > 30:
                    need_new_waypoint = True

            # --- KINEMATIC GRAPH SEARCH ROUTING ---
            if need_new_waypoint:
                all_nodes = torch.cat([obs_normalized, waypoints_normalized, global_goal_normalized], dim=0)
                
                # Calculate True Physical Distances
                nodes_xy = torch.cat([
                    torch.tensor(obs[:2], dtype=torch.float32, device=device).unsqueeze(0),
                    waypoints_raw[:, :2],
                    global_goal_raw[:2].unsqueeze(0)
                ], dim=0)
                
                phys_dist_matrix = torch.cdist(nodes_xy, nodes_xy).cpu().numpy()
                
                Z_source = F.normalize(model(all_nodes), p=2.0, dim=-1)
                Z_target = F.normalize(goal_proj(model(all_nodes)), p=2.0, dim=-1)
                sim_matrix = (Z_source @ Z_target.T).cpu().numpy()
                
                adj_matrix = np.zeros_like(sim_matrix)
                valid_edges = np.zeros_like(sim_matrix, dtype=bool)
                
                # Connect nodes using Latent Similarity, but ONLY if physically safe!
                for i in range(len(all_nodes)):
                    # Must be physically reachable (less than 3.0 meters) to avoid Ghost Edges
                    reachable_mask = (phys_dist_matrix[i] < 3.0) & (phys_dist_matrix[i] > 0.01)
                    reachable_indices = np.where(reachable_mask)[0]
                    
                    if len(reachable_indices) > 0:
                        reachable_sims = sim_matrix[i, reachable_indices]
                        k = min(15, len(reachable_indices))
                        best_local_indices = reachable_indices[np.argsort(reachable_sims)[-k:]]
                        valid_edges[i, best_local_indices] = True
                
                # Guarantee goal can be reached safely
                goal_idx = len(all_nodes) - 1
                reachable_to_goal = np.where((phys_dist_matrix[:, goal_idx] < 3.0) & (phys_dist_matrix[:, goal_idx] > 0.01))[0]
                if len(reachable_to_goal) > 0:
                    sims_to_goal = sim_matrix[reachable_to_goal, goal_idx]
                    k = min(15, len(reachable_to_goal))
                    best_to_goal = reachable_to_goal[np.argsort(sims_to_goal)[-k:]]
                    valid_edges[best_to_goal, goal_idx] = True
                
                adj_matrix[valid_edges] = np.clip(1.0 - sim_matrix[valid_edges], a_min=1e-5, a_max=None)
                graph = sp.csr_matrix(adj_matrix)
                
                distances, predecessors = csgraph.shortest_path(graph, directed=True, indices=0, return_predecessors=True)
                
                if predecessors[goal_idx] != -9999:
                    curr = goal_idx
                    path = []
                    while curr != 0:
                        path.append(curr)
                        curr = predecessors[curr]
                    
                    # THE FIX: Reel in the carrot! Target the IMMEDIATE next hop (Index 1) 
                    # so the Actor stays strictly within its 40-step training distribution!
                    lookahead_idx = min(1, len(path))
                    next_waypoint_idx = path[-lookahead_idx] 
                    
                    active_subgoal = all_nodes[next_waypoint_idx].unsqueeze(0)
                    print(f"Path found! Target is {len(path)} hops away.")
                else:
                    print("Graph disconnected. Taking a predictive forward hop.")
                    best_neighbor = np.argmax(sim_matrix[0][1:-1]) + 1
                    active_subgoal = all_nodes[best_neighbor].unsqueeze(0)
                
                subgoal_step_count = 0
            
        subgoal_step_count += 1
        
        # Pass the local breadcrumb to the Actor
        action_tensor = actor(obs_normalized, active_subgoal)
        action = action_tensor.squeeze(0).cpu().numpy()
        
        action = np.nan_to_num(action, nan=0.0)
        action = action + np.random.normal(loc=0.0, scale=0.05, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminal, truncated, info = env.step(action)
        
        # If the Ant gets within 1.0 meter of our custom goal, consider it a success!
        if dist_to_real_goal < 1.0 or terminal:
            success = True
            print(f"Goal successfully reached at step {step}!")
            for _ in range(30): frames.append(env.render()) 
            break
            
        if truncated:
            print(f"Episode timed out at {step} steps.")
            break

output_filename = "antmaze_sorb_rollout_custom_goal.mp4"
print(f"Saving video to {output_filename}...")
import imageio
imageio.mimsave(output_filename, frames, fps=30)