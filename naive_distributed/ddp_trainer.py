import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .actor import GoalConditionedActor
from .centralized_buffer import CentralizedReplayBuffer


class NaiveDDPTrainer:
    """DDP Trainer with Centralized Replay Buffer & Sequential dist.broadcast."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = None,
        lr: float = 1e-3
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor wrapped in standard DDP
        raw_actor = GoalConditionedActor(state_dim=state_dim, action_dim=action_dim).to(self.device)
        if dist.is_initialized():
            self.actor = DDP(raw_actor, device_ids=[self.device.index] if self.device.type == "cuda" else None)
        else:
            self.actor = raw_actor

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.buffer = CentralizedReplayBuffer(50000, state_dim, action_dim, device="cpu") if rank == 0 else None

    def run_training_step(self, batch_size: int = 32):
        """Executes a single step measuring dist.broadcast and gradient sync latency."""
        t0 = time.perf_counter()
        
        # Centralized broadcast chokepoint
        if self.buffer is not None:
            s_t, a_t, _, _, _, s_g = self.buffer.broadcast_sample(batch_size, self.rank, self.device)
        else:
            s_t, a_t, _, _, _, s_g = None, None, None, None, None, None
            if dist.is_initialized():
                dummy_buffer = CentralizedReplayBuffer(1, s_t_shape=69, a_t_shape=8)
                s_t, a_t, _, _, _, s_g = dummy_buffer.broadcast_sample(batch_size, self.rank, self.device)

        broadcast_lat_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        self.optimizer.zero_grad()
        pred = self.actor(s_t, s_g)
        loss = F.mse_loss(pred, a_t)
        loss.backward()
        self.optimizer.step()
        sgd_lat_ms = (time.perf_counter() - t1) * 1000.0

        return {
            "broadcast_latency_ms": broadcast_lat_ms,
            "sgd_latency_ms": sgd_lat_ms,
            "loss": loss.item()
        }
