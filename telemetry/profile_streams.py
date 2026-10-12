import os
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from disaggregated.diffusion_policy import GoalConditionedDiffusionTransformer
from disaggregated.ipc_weight_sync import CUDAPeerWeightBroadcaster


def profile_dual_stream_concurrency():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA device required for hardware stream profiling.")
        return

    print("Profiling Dual CUDA Streams (Compute vs Async Sync)...")
    model = GoalConditionedDiffusionTransformer(69, 8, dim=1024, depth=12, num_heads=16).to(device)
    broadcaster = CUDAPeerWeightBroadcaster(model, device)

    compute_stream = torch.cuda.current_stream()
    sync_stream = torch.cuda.Stream()

    batch_sz = 8
    s = torch.randn(batch_sz, 69, device=device)
    g = torch.randn(batch_sz, 69, device=device)
    a = torch.randn(batch_sz, 8, device=device)
    t = torch.randint(0, 100, (batch_sz,), device=device)

    trace_path = os.path.join(ROOT, "telemetry/results/stream_concurrency_trace.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("compute_step"):
            pred = model(a, t, s, g)
            loss = pred.sum()
            loss.backward()

        with record_function("async_weight_sync"):
            with torch.cuda.stream(sync_stream):
                broadcaster.pack_and_broadcast_async(model)

        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"Exported PyTorch Profiler Chrome trace to: {trace_path}")


if __name__ == '__main__':
    profile_dual_stream_concurrency()
