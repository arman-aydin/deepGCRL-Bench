import os
import sys
import time
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepgcrl import (
    GoalConditionedDiffusionTransformer,
    CUDAPeerWeightBroadcaster
)


def profile_stream_overlap():
    print("=" * 70)
    print("NSIGHT STREAM CONCURRENCY & OVERLAP PROFILER")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA device required for stream concurrency profiling.")
        return

    # 1. Instantiate 232M Model & Broadcaster
    model = GoalConditionedDiffusionTransformer(
        state_dim=69,
        action_dim=8,
        dim=1024,
        depth=12,
        num_heads=16
    ).to(device)

    broadcaster = CUDAPeerWeightBroadcaster(model, device)

    # Prepare dummy training batch
    batch_size = 8
    s = torch.randn(batch_size, 69, device=device)
    g = torch.randn(batch_size, 69, device=device)
    a = torch.randn(batch_size, 8, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)

    # 2. CUDA Streams
    compute_stream = torch.cuda.current_stream()
    transfer_stream = torch.cuda.Stream()

    # Warmup
    for _ in range(3):
        out = model(a, t, s, g)
        out.sum().backward()
        model.zero_grad(set_to_none=True)
        broadcaster.pack_and_broadcast_async(model)
    torch.cuda.synchronize()

    # -------------------------------------------------------------
    # Experiment 1: Measure Compute alone (Forward + Backward)
    # -------------------------------------------------------------
    start_compute = torch.cuda.Event(enable_timing=True)
    stop_compute = torch.cuda.Event(enable_timing=True)

    start_compute.record(compute_stream)
    out = model(a, t, s, g)
    loss = out.sum()
    loss.backward()
    stop_compute.record(compute_stream)
    compute_stream.synchronize()
    t_compute_alone_ms = start_compute.elapsed_time(stop_compute)

    # -------------------------------------------------------------
    # Experiment 2: Measure Memory Transfer alone (232M Params)
    # -------------------------------------------------------------
    start_transfer = torch.cuda.Event(enable_timing=True)
    stop_transfer = torch.cuda.Event(enable_timing=True)

    start_transfer.record(transfer_stream)
    with torch.cuda.stream(transfer_stream):
        broadcaster.pack_and_broadcast_async(model)
    stop_transfer.record(transfer_stream)
    transfer_stream.synchronize()
    t_transfer_alone_ms = start_transfer.elapsed_time(stop_transfer)

    # -------------------------------------------------------------
    # Experiment 3: Overlapped Execution (Concurrent Dual Streams)
    # -------------------------------------------------------------
    model.zero_grad(set_to_none=True)
    start_all = torch.cuda.Event(enable_timing=True)
    stop_all = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push("Overlapped_Iteration")
    start_all.record(compute_stream)

    # 1. Non-blocking launch: memory transfer on dedicated stream
    with torch.cuda.stream(transfer_stream):
        torch.cuda.nvtx.range_push("Async_Memory_Transfer_Stream")
        broadcaster.pack_and_broadcast_async(model)
        torch.cuda.nvtx.range_pop()

    # 2. Concurrently launch compute on compute_stream
    torch.cuda.nvtx.range_push("Compute_Forward_Backward_Stream")
    out = model(a, t, s, g)
    loss = out.sum()
    loss.backward()
    torch.cuda.nvtx.range_pop()

    # 3. Synchronize both streams
    transfer_stream.synchronize()
    stop_all.record(compute_stream)
    compute_stream.synchronize()

    torch.cuda.nvtx.range_pop()

    t_overlapped_ms = start_all.elapsed_time(stop_all)
    t_serial_sum_ms = t_compute_alone_ms + t_transfer_alone_ms
    critical_path_delta_ms = max(0.0, t_overlapped_ms - t_compute_alone_ms)
    overlap_efficiency_pct = max(0.0, min(100.0, (1.0 - (critical_path_delta_ms / max(1e-4, t_transfer_alone_ms))) * 100.0))

    print(f"\nEMPIRICAL OVERLAP TELEMETRY:")
    print(f"    - Compute Time Alone (Forward + Backward): {t_compute_alone_ms:.2f} ms")
    print(f"    - Memory Transfer Alone (232M Parameters): {t_transfer_alone_ms:.2f} ms")
    print(f"    - Theoretical Serial Sum (No Overlap):     {t_serial_sum_ms:.2f} ms")
    print(f"    - Overlapped Execution Time:               {t_overlapped_ms:.2f} ms")
    print(f"    - Critical Path Overhead Added by Transfer: {critical_path_delta_ms:.2f} ms")
    print(f"    - Memory Transfer Hiding Efficiency:       {overlap_efficiency_pct:.1f}% HIDDEN (Off Critical Path)\n")

    # -------------------------------------------------------------
    # Experiment 4: Capture PyTorch Profiler / Chrome Trace JSON
    # -------------------------------------------------------------
    print("Capturing PyTorch Profiler CUDA timeline trace...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    trace_path = os.path.join(results_dir, "stream_overlap_trace.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        with record_function("Concurrent_Execution"):
            with torch.cuda.stream(transfer_stream):
                broadcaster.pack_and_broadcast_async(model)

            out = model(a, t, s, g)
            out.sum().backward()

            transfer_stream.synchronize()
            compute_stream.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"Chrome Perfetto Trace saved to: {trace_path}")

    return {
        "compute_alone_ms": round(t_compute_alone_ms, 2),
        "transfer_alone_ms": round(t_transfer_alone_ms, 2),
        "serial_sum_ms": round(t_serial_sum_ms, 2),
        "overlapped_time_ms": round(t_overlapped_ms, 2),
        "critical_path_delta_ms": round(critical_path_delta_ms, 2),
        "transfer_hidden_percent": round(overlap_efficiency_pct, 1)
    }


if __name__ == '__main__':
    profile_stream_overlap()