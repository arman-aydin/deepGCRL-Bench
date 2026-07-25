import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader

from dataset_loader import OGBenchCustomBatchedDataset
from resnet_modelling import ResNetConfigDepth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "contrastive_checkpoint.pt"

# ==========================================
# 1. Dataset Initialization
# ==========================================
batch_size = 128
dataset = OGBenchCustomBatchedDataset(dataset_name='humanoidmaze-large-navigate-v0', batch_size=batch_size)
dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

# ==========================================
# 2. Architecture Initialization
# ==========================================
model = ResNetConfigDepth(depth=32, state_dim=69, hidden_dim=256).to(device)

goal_proj = nn.Sequential(
    nn.Linear(256, 256),
    nn.LayerNorm(256),
    nn.SiLU(), 
    nn.Linear(256, 256)
).to(device)

# Initialize log_tau parameter
log_tau = torch.nn.Parameter(torch.tensor(math.log(0.07), device=device, requires_grad=True))

# ==========================================
# 3. Safe Checkpoint Loading
# ==========================================
# We define the optimizer AFTER the model architectures are defined, 
# but we only load its state dict AFTER linking it to the parameters.
optim = torch.optim.Adam(list(model.parameters()) + list(goal_proj.parameters()) + [log_tau], lr=3e-4)

start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load architecture weights
    model.load_state_dict(checkpoint['model_state_dict'])
    goal_proj.load_state_dict(checkpoint['goal_proj_state_dict'])
    
    # Carefully overwrite the leaf parameter data for the temperature
    with torch.no_grad():
        log_tau.copy_(torch.tensor(checkpoint['log_tau'], device=device))
    
    # Load optimizer momentum buffers and state
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Successfully loaded checkpoint. Resuming training with Tau: {math.exp(log_tau.item()):.4f}")
    # Optional: If you want to keep track of total epochs visually, you can start from 8.
    start_epoch = 8 
else:
    print(f"No checkpoint found at {checkpoint_path}. Starting from scratch...")

# ==========================================
# 4. Resumed Training Loop
# ==========================================
additional_epochs = 10  # How many MORE epochs you want to train

for epoch in range(start_epoch, start_epoch + additional_epochs):
    # Ensure models are in training mode
    model.train()
    goal_proj.train()
    
    for i, sample in enumerate(dataloader):
        anchor_states, future_goal_states = sample
        anchor_states = anchor_states.to(device)
        future_goal_states = future_goal_states.to(device)

        optim.zero_grad(set_to_none=True) # Slightly faster and saves memory

        # 1. Generate Latent Embeddings
        z_anchors = model(anchor_states)
        z_futures = goal_proj(model(future_goal_states))

        # 2. Normalize the embeddings to unit length (Cosine Similarity)
        z_anchors = F.normalize(z_anchors, p=2.0, dim=-1)
        z_futures = F.normalize(z_futures, p=2.0, dim=-1)

        # 3. Compute the InfoNCE Logit Matrix and apply Temperature
        logits = z_anchors @ z_futures.T
        tau = torch.exp(log_tau)
        values = logits / tau
        targets = torch.arange(start=0, end=batch_size, device=device)

        loss = F.cross_entropy(values, targets)
        loss.backward()
        
        optim.step()

        # Monitor progress within the epoch
        if i % 500 == 0:
            print(f"Epoch {epoch} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f} | Tau: {tau.item():.4f}")

    print(f"=== End of Epoch {epoch} | Final Loss: {loss.item():.4f} | Tau: {tau.item():.4f} ===")

    # Safely overwrite the checkpoint at the end of each epoch
    torch.save({
        'epoch': epoch, # Pro-tip: Save the current epoch number for future tracking
        'model_state_dict': model.state_dict(),
        'goal_proj_state_dict': goal_proj.state_dict(),
        'log_tau': log_tau.item(),
        'optimizer_state_dict': optim.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint updated at {checkpoint_path}")