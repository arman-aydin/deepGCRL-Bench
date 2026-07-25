import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader

from dataset_loader import OGBenchCustomBatchedDataset
from resnet_modelling import ResNetConfigDepth
from actor_modeling import GoalConditionedActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 256
# dataset_name = 'humanoidmaze-large-navigate-v0'
dataset_name = 'antmaze-large-navigate-v0'
dataset = OGBenchCustomBatchedDataset(dataset_name=dataset_name, batch_size=batch_size)
dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

action_dim = dataset.actions.shape[-1]
print(f"Detected Action Dimension: {action_dim}")

# Then pass it to your Actor

model = ResNetConfigDepth(depth=32, state_dim=29, hidden_dim=256).to(device)

goal_proj = nn.Sequential(
    nn.Linear(256, 256),
    nn.LayerNorm(256),
    nn.SiLU(), 
    nn.Linear(256, 256)
).to(device)

# Load architecture weights
checkpoint_path = f"{dataset_name}-contrastive_checkpoint.pt"

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
goal_proj.load_state_dict(checkpoint['goal_proj_state_dict'])
    

# 1. Freeze the ResNet Representations
model.eval()
goal_proj.eval()
for param in model.parameters(): param.requires_grad = False
for param in goal_proj.parameters(): param.requires_grad = False

# 2. Initialize Actor
actor = GoalConditionedActor(state_dim=29, action_dim=action_dim).to(device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)

# Alpha controls how strongly we weight the Advantage
alpha = 0.75 

for epoch in range(10):
    for i, (s_t, a_t, s_next, s_goal) in enumerate(dataloader):
        s_t, a_t, s_next, s_goal = s_t.to(device), a_t.to(device), s_next.to(device), s_goal.to(device)
        
        actor_optim.zero_grad(set_to_none=True)
        
        # 1. Evaluate Advantage using Frozen ResNet
        with torch.no_grad():
            z_t = F.normalize(model(s_t), p=2.0, dim=-1)
            z_next = F.normalize(model(s_next), p=2.0, dim=-1)
            z_goal = F.normalize(goal_proj(model(s_goal)), p=2.0, dim=-1)
            
            v_t = (z_t * z_goal).sum(dim=-1) 
            v_next = (z_next * z_goal).sum(dim=-1) 
            
            raw_advantage = v_next - v_t # Save the raw math for our telemetry
            
            # Normalize for training
            advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)
            weights = torch.exp(alpha * advantage).clamp(max=100.0)

        # 2. Train the Actor
        pred_actions = actor(s_t, s_goal)
        bc_loss = F.mse_loss(pred_actions, a_t, reduction='none').mean(dim=-1)
        
        loss = (weights * bc_loss).mean()
        loss.backward()
        actor_optim.step()

        # --- TELEMETRY ---
        if i % 500 == 0:
            # Print the RAW advantage mean so you can see the ResNet is still alive
            print(f"Epoch {epoch} | Batch {i}/{len(dataloader)} | AWAC Loss: {loss.item():.4f} | Raw Adv Mean: {raw_advantage.mean().item():.5f} | W-Max: {weights.max().item():.2f}")

    print(f"=== End of Epoch {epoch} ===")

# Save the trained Actor for Milestone 5
actor_checkpoint_path = f"{dataset_name}-actor_checkpoint.pt"
torch.save(actor.state_dict(), actor_checkpoint_path)
print(f"Successfully saved Actor to {actor_checkpoint_path}")