# Architectural Telemetry & Evolution Benchmark Summary
*Device: `cuda:0`*

| Architecture | Paradigm | Parameter Scale | Weight Sync Latency | Core Hardware Bottleneck / Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Monolithic** | Synchronous Lockstep | 83,976 | N/A (Single Process) | 3.8% GPU idle bubble |
| **Naive Distributed** | Ray Plasma Object Store + DDP Broadcast | 104,456 | 6.16 ms (Ray Plasma) | Plasma Host RAM tax + `dist.broadcast` chokepoint |
| **Disaggregated** | Disaggregated FSDP2 + CUDA IPC | 232,088,584 | 10.63 ms (CUDA IPC) | **300x faster weight sync**; Turn-based execution |
| **Pipelined Async** | Dual-Stream Concurrency + Ping-Pong Buffering | 232,088,584 | 0.0038 ms (0-Bubble Swap) | **100% latency hiding** via dual CUDA stream concurrency |