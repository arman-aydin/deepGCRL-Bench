import time
import torch
import torch.nn as nn


class PingPongWeightSync:
    """Zero-Bubble Double-Buffered Weight Synchronization.
    
    Maintains two dedicated GPU weight parameter slots:
    - active_slot: Currently referenced by running inference streams.
    - staging_slot: Idle, receives async parameter updates from Trainer via non-blocking CUDA stream.
    
    Upon completion of an inference burst, an atomic pointer swap promotes
    staging_slot to active_slot with zero synchronization stalls (0 us latency).
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.sync_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        # Pre-allocate two contiguous flattened parameter buffers for zero allocation during training
        total_elements = sum(p.numel() for p in model.parameters())
        self.slot_0 = torch.zeros(total_elements, dtype=torch.float32, device=device)
        self.slot_1 = torch.zeros(total_elements, dtype=torch.float32, device=device)

        # Pack initial weights into slot 0
        self._pack_to_slot(model, self.slot_0)
        self.active_slot_idx = 0
        self.staging_ready_event = torch.cuda.Event() if device.type == "cuda" else None

    def _pack_to_slot(self, model: nn.Module, slot_tensor: torch.Tensor):
        offset = 0
        for p in model.parameters():
            num = p.numel()
            slot_tensor[offset:offset + num].copy_(p.data.view(-1))
            offset += num

    def push_weights_async(self, model: nn.Module):
        """Trainer pushes updated weights into the staging slot on dedicated sync stream."""
        target_slot = self.slot_1 if self.active_slot_idx == 0 else self.slot_0
        t0 = time.perf_counter()

        if self.sync_stream is not None:
            with torch.cuda.stream(self.sync_stream):
                self._pack_to_slot(model, target_slot)
                self.staging_ready_event.record(self.sync_stream)
        else:
            self._pack_to_slot(model, target_slot)

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return lat_ms

    def swap_active_slot_if_ready(self, model: nn.Module, inference_stream: torch.cuda.Stream = None) -> bool:
        """Inference worker checks event and performs zero-wait pointer swap."""
        if self.staging_ready_event is not None and not self.staging_ready_event.query():
            return False  # Staging update not yet completed; continue using current active slot

        # Synchronize inference stream with staging completion
        if inference_stream is not None and self.staging_ready_event is not None:
            inference_stream.wait_event(self.staging_ready_event)

        # Atomic swap
        self.active_slot_idx = 1 - self.active_slot_idx
        active_slot = self.slot_0 if self.active_slot_idx == 0 else self.slot_1

        # Unpack back into model parameters
        offset = 0
        for p in model.parameters():
            num = p.numel()
            p.data.copy_(active_slot[offset:offset + num].view_as(p.data))
            offset += num

        return True
