import os
import time
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class CUDABroadcastPayload:
    """Wrapper carrying either a direct GPU tensor (same process) or a CUDA IPC handle (cross-process)."""
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.pid = os.getpid()
        if tensor.is_cuda:
            try:
                self.ipc_handle = tensor.untyped_storage()._share_cuda_()
            except Exception:
                self.ipc_handle = None
        else:
            self.ipc_handle = None


class CUDAPeerWeightBroadcaster:
    """Zero-Copy Peer-to-Peer GPU Weight Broadcaster.
    
    Resides on the distributed training master rank. Synchronizes full actor
    parameters directly to remote GPU rollout inference actors without host
    CPU roundtrips, utilizing non-blocking CUDA streams and CUDA IPC handles.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.is_cuda = (device.type == "cuda" and torch.cuda.is_available())

        # Calculate total flat parameter elements
        self.param_specs = []
        total_elements = 0
        for name, param in model.named_parameters():
            numel = param.numel()
            self.param_specs.append((name, param.shape, param.dtype, total_elements, numel))
            total_elements += numel

        self.total_elements = total_elements
        self.total_bytes = total_elements * 4 # FP32 default

        # Allocate contiguous shared GPU buffer for broadcast
        if self.is_cuda:
            self.broadcast_stream = torch.cuda.Stream(device=self.device)
            self.buffer = torch.zeros(total_elements, dtype=torch.float32, device=self.device)
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
        else:
            self.broadcast_stream = None
            self.buffer = torch.zeros(total_elements, dtype=torch.float32, device=self.device)
            self.start_event = None
            self.stop_event = None

    def pack_and_broadcast_async(self, model: nn.Module) -> Tuple[CUDABroadcastPayload, float]:
        """Flatten model weights into contiguous GPU buffer on a non-blocking stream.
        
        Returns:
            payload: CUDABroadcastPayload with tensor / IPC handle
            sync_latency_ms: time taken for non-blocking transfer in ms
        """
        t0 = time.perf_counter()
        named_params = dict(model.named_parameters())

        if self.is_cuda:
            with torch.cuda.stream(self.broadcast_stream):
                self.start_event.record(self.broadcast_stream)

                for name, shape, dtype, offset, numel in self.param_specs:
                    param = named_params[name]
                    flat_data = param.data.view(-1).to(self.device, dtype=torch.float32)
                    self.buffer[offset:offset + numel].copy_(flat_data, non_blocking=True)

                self.stop_event.record(self.broadcast_stream)
                self.broadcast_stream.synchronize()
                elapsed_ms = self.start_event.elapsed_time(self.stop_event)
        else:
            for name, shape, dtype, offset, numel in self.param_specs:
                param = named_params[name]
                flat_data = param.data.view(-1).to(self.device, dtype=torch.float32)
                self.buffer[offset:offset + numel].copy_(flat_data)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        payload = CUDABroadcastPayload(self.buffer)
        return payload, elapsed_ms


class CUDAPeerWeightReceiver:
    """Zero-Copy Peer-to-Peer GPU Weight Receiver.
    
    Resides on the disaggregated GPU rollout inference engine. Unpacks weights
    directly into the inference policy on GPU with zero host-memory copying.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.is_cuda = (device.type == "cuda" and torch.cuda.is_available())
        self.model = model

        # Parameter unpacking specifications
        self.param_specs = []
        total_elements = 0
        for name, param in model.named_parameters():
            numel = param.numel()
            self.param_specs.append((name, param.shape, param.dtype, total_elements, numel))
            total_elements += numel

        self.total_elements = total_elements

        if self.is_cuda:
            self.inference_stream = torch.cuda.Stream(device=self.device)
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
        else:
            self.inference_stream = None

    def unpack_weights_async(self, payload) -> float:
        """Unpack contiguous GPU buffer directly into target model parameters."""
        t0 = time.perf_counter()

        # Resolve flat buffer
        if isinstance(payload, CUDABroadcastPayload):
            if os.getpid() == payload.pid or payload.ipc_handle is None or not self.is_cuda:
                flat_buffer = payload.tensor
            else:
                storage = torch.UntypedStorage._new_shared_cuda(*payload.ipc_handle)
                flat_buffer = torch.tensor(storage, dtype=torch.float32, device=self.device)
        elif isinstance(payload, torch.Tensor):
            flat_buffer = payload
        elif isinstance(payload, (tuple, list)):
            storage = torch.UntypedStorage._new_shared_cuda(*payload)
            flat_buffer = torch.tensor(storage, dtype=torch.float32, device=self.device)
        else:
            raise TypeError(f"Unsupported payload type: {type(payload)}")

        named_params = dict(self.model.named_parameters())

        if self.is_cuda:
            with torch.cuda.stream(self.inference_stream):
                self.start_event.record(self.inference_stream)
                for name, shape, dtype, offset, numel in self.param_specs:
                    param = named_params[name]
                    unflat = flat_buffer[offset:offset + numel].view(shape).to(dtype)
                    param.data.copy_(unflat, non_blocking=True)
                self.stop_event.record(self.inference_stream)
                self.inference_stream.synchronize()
                elapsed_ms = self.start_event.elapsed_time(self.stop_event)
        else:
            for name, shape, dtype, offset, numel in self.param_specs:
                param = named_params[name]
                unflat = flat_buffer[offset:offset + numel].view(shape).to(self.device, dtype=param.dtype)
                param.data.copy_(unflat)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return elapsed_ms