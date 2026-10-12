# Stage 0: The Monolithic Engine — Synchronous Lockstep and The Compute Bubble

<div align="center">

**A Systems Autopsy of Single-Process Reinforcement Learning Loops**

</div>

---

## Executive Summary

The **Monolithic Engine** represents the foundational architectural pattern underpinning single-file RL implementations (e.g., [CleanRL](https://github.com/vwxyzjn/cleanrl), basic Gymnasium loops). In this paradigm, environment simulation, neural network forward inference, buffer insertion, and backpropagation execute sequentially on a single thread.

While straightforward to implement, this pattern establishes an **idle bubble anti-pattern** where heterogeneous compute devices (host CPU cores and GPU Tensor Cores) alternate execution rather than operating concurrently.

---

## System Topology and Execution Timeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Synchronous Lockstep Timeline                                  │
├──────────────────────────┬─────────────────────────────┬────────────────────────────────────┤
│ Phase                    │ Active Hardware             │ Stalled Hardware                   │
├──────────────────────────┼─────────────────────────────┼────────────────────────────────────┤
│ 1. CPU Physics Step      │ Host CPU (1 Core)           │ GPU SMs (0% Compute Utilization)   │
│ 2. Policy Inference      │ GPU Stream (Small Batch=1)  │ Host Thread Blocked on D2H Sync    │
│ 3. Buffer Update         │ Host RAM Memory Bus         │ GPU Idle                           │
│ 4. Backward Pass & SGD   │ GPU CUDA Cores              │ Host Thread Blocked on Kernel Sync │
└──────────────────────────┴─────────────────────────────┴────────────────────────────────────┘
```

---

## Systems Analysis: Why Lockstep Execution Breaks Down

1. **Catastrophic Hardware Underutilization**:
   $$\text{Hardware Duty Cycle} = \frac{T_{\text{GPU}}}{T_{\text{GPU}} + T_{\text{CPUPhysics}}}$$
   Because single-threaded MuJoCo physics steps dominate execution time, the GPU sits idle during simulation. On an RTX 3060 Ti, our telemetry reveals an unavoidable **3.8% idle bubble** on simple navigation environments, which degrades further to $>60\%$ on complex physics like `humanoid`.
2. **Memory Bandwidth Starvation**:
   Small-batch inference ($B=1$) fails to saturate the memory bus or compute units of modern GPUs, incurring high kernel launch overhead per transition.

---

## Empirical Telemetry (RTX 3060 Ti)

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Policy Scale** | **83,976 parameters** | 3-layer Goal-Conditioned MLP |
| **End-to-End Throughput** | **80.9 steps/second** | Single environment stepping |
| **GPU Compute Bubble** | **3.8% measured idle** | Scales with physics simulation complexity |
| **Weight Sync Latency** | **0.0 ms** | Co-located in same virtual process address space |

---

## Benchmark Execution

```bash
python -m monolithic.run --steps 100 --batch-size 32
```
