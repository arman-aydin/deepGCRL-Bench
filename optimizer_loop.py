import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader

from dataset_loader import OGBenchCustomBatchedDataset
from resnet_modelling import ResNetConfigDepth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
dataset_name = 'antmaze-large-navigate-v0'
dataset = OGBenchCustomBatchedDataset(dataset_name=dataset_name, batch_size=batch_size)
dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

# We need the dataset stats to un-normalize coordinates for the Kinematic Mask!
obs_mean = dataset.obs_mean.to(device)
obs_std = dataset.obs_std.to(device)

# FIX 1: Downscale to 32 layers! 64 layers will overfit the 29D Ant state and shatter the graph.
model = ResNetConfigDepth(depth=32, state_dim=29, hidden_dim=256).to(device)

# We need a separate projection for the goals to avoid the "Symmetric Encoder Trap"
goal_proj = nn.Sequential(
    nn.Linear(256, 256),
    nn.LayerNorm(256),
    nn.SiLU(), # Non-linearity is crucial here
    nn.Linear(256, 256)
).to(device)

# Standard initialization for contrastive temperature is ~0.07 (math.log(0.07) is approx -2.65)
log_tau = torch.nn.Parameter(torch.tensor(math.log(0.07), device=device, requires_grad=True))

optim = torch.optim.Adam(list(model.parameters()) + list(goal_proj.parameters()) + [log_tau], lr=3e-4)

for epoch in range(8):
    for i, sample in enumerate(dataloader):
        anchor_states, _, _, future_goal_states = sample
        anchor_states = anchor_states.to(device)
        future_goal_states = future_goal_states.to(device)
        
        current_batch_size = anchor_states.size(0)

        optim.zero_grad()

        # 1. Generate Latent Embeddings
        z_anchors = model(anchor_states)
        z_futures = model(future_goal_states)

        # Asymmetric projection so model(anchor) doesn't have to identically equal model(future)
        z_futures = goal_proj(z_futures)

        # 2. Normalize the embeddings to unit length (Cosine Similarity)
        z_anchors = F.normalize(z_anchors, p=2.0, dim=-1)
        z_futures = F.normalize(z_futures, p=2.0, dim=-1)

        # 3. Compute the InfoNCE Logit Matrix and apply Temperature
        logits = z_anchors @ z_futures.T
        tau = torch.exp(log_tau)
        values = logits / tau
        
        # ====================================================================
        # FIX 2: KINEMATIC MASKING (False Negative Rejection)
        # ====================================================================
        # Un-normalize the first two dimensions to get absolute physical (X, Y) coordinates
        true_xy = anchor_states[:, :2] * obs_std[:2] + obs_mean[:2]
        
        # Calculate physical distances between all states in the batch
        phys_dist_matrix = torch.cdist(true_xy, true_xy)
        
        # Identify False Negatives: States physically closer than 1.5 meters
        false_negative_mask = phys_dist_matrix < 1.5
        
        # Remove the diagonal (True Positives) from the mask, because we DO want them!
        mask_diagonal = torch.eye(current_batch_size, dtype=torch.bool, device=device)
        false_negative_mask = false_negative_mask & ~mask_diagonal
        
        # Apply the mask: Set False Negatives to -1e9 so CrossEntropy ignores them
        values[false_negative_mask] = -1e9
        # ====================================================================

        targets = torch.arange(start=0, end=current_batch_size).to(device)

        loss = F.cross_entropy(values, targets)
        loss.backward()
        
        optim.step()

        # Monitor progress within the epoch
        if i % 500 == 0:
            print(f"Epoch {epoch} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f} | Tau: {tau.item():.4f}")

    # Print loss at the end of each epoch for a cleaner log
    print(f"=== End of Epoch {epoch} | Final Loss: {loss.item():.4f} | Tau: {tau.item():.4f} ===")

# Save the trained model, projection head, temperature, and optimizer state
checkpoint_path = f"{dataset_name}-32-layers-contrastive_checkpoint.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'goal_proj_state_dict': goal_proj.state_dict(),
    'log_tau': log_tau.item(),
    'optimizer_state_dict': optim.state_dict(),
}, checkpoint_path)
print(f"Successfully saved checkpoint to {checkpoint_path}")