# Stage 1: The Naive Distributed Engine — Serialization Taxes and Collective Chokepoints

<div align="center">

**Scaling Bottlenecks in Standard Ray Plasma and PyTorch DDP Frameworks**

</div>

---

## Executive Summary

To overcome the single-threaded CPU bottleneck of the monolithic baseline, the **Naive Distributed Engine** adopts standard industry practices:
1. Scaling environment rollouts across multi-core CPU workers using [Ray remote actors](https://www.ray.io/).
2. Synchronizing distributed gradient updates across GPUs using PyTorch [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

While this decouples physics simulation from optimization, scaling to modern parameter scales ($>100\text{M}$) reveals two fatal systems failures: **the 3-second Ray Plasma serialization tax** and **the centralized `dist.broadcast` collective chokepoint**.

---

## System Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Ray Cluster Coordinator                                   │
└───────────────────────────┬────────────────────────────────┬───────────────────────────┘
                            │                                │
            [ Ray CPU Rollout Workers ]             [ Distributed DDP Training Ranks ]
         ┌───────────────────────────────┐        ┌───────────────────────────────────┐
         │ Ray Worker Pool (1..N)        │        │ PyTorch DDP Rank 0                │
         │ • AsyncVectorEnv instances    │        │ • Holds physical replay buffer    │
         │ • Fetches weights via Plasma  │        │ • Issues 6x dist.broadcast calls  │
         └──────────────┬────────────────┘        └─────────────────┬─────────────────┘
                        │                                           │
                        │ Ray Plasma Host Memory                    │ Sequential NCCL / Gloo
                        ▼ Roundtrip (D2H -> H2D)                    ▼ dist.broadcast
         ┌───────────────────────────────┐        ┌───────────────────────────────────┐
         │ Plasma Object Store           │───────►│ DDP Worker Rank 1..W              │
         │ (3.1s serialization tax)      │        │ (Blocked waiting for Rank 0 data) │
         └───────────────────────────────┘        └───────────────────────────────────┘
```

---

## Systems Breakdown of the Dual Bottlenecks

### 1. The 3-Second Serialization Tax (Ray Plasma)
When rollout workers synchronize parameters with the trainer, weights undergo a 4-hop host memory roundtrip:
$$\text{Trainer GPU} \xrightarrow{\text{D2H}} \text{CPU Host RAM} \xrightarrow{\text{IPC}} \text{Plasma Store} \xrightarrow{\text{IPC}} \text{Worker CPU} \xrightarrow{\text{H2D}} \text{Worker GPU}$$

For a toy 100k-parameter MLP, this incurs $6.16\text{ ms}$. But for a **232M-parameter Foundation Actor ($885\text{ MB}$)**:
$$\text{Transfer Time} \approx \frac{885\text{ MB}}{16\text{ GB/s (PCIe Gen4)}} \times 2 + T_{\text{pickle}} + T_{\text{IPC}} \ge \mathbf{3,100\text{ ms}}$$
Every synchronization freeze halts rollout workers for over 3 seconds, stalling environment throughput.

### 2. The Centralized Broadcast Chokepoint
In naive multi-GPU training, Rank 0 aggregates all transitions into a single centralized buffer. During each gradient step, Rank 0 samples a batch and sequentially broadcasts 6 tensors:
$$\text{broadcast}(s_t) \to \text{broadcast}(a_t) \to \text{broadcast}(s_{t+1}) \to \text{broadcast}(r_t) \to \text{broadcast}(d_t) \to \text{broadcast}(g)$$
Worker ranks sit blocked in network barriers waiting for Rank 0 serialization, creating severe collective stalls.

---

## Empirical Telemetry (RTX 3060 Ti)

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Policy Scale** | **104,456 parameters** | Goal-Conditioned Actor |
| **Ray Plasma Sync Latency** | **6.16 ms** | Small MLP ($350\text{ KB}$); extrapolates to $>3.1\text{s}$ at $232\text{M}$ scale |
| **Broadcast Sample Stall** | **0.32 ms/step** | Blocking collective overhead per mini-batch update |
| **DDP Memory Scaling** | **No Parameter Sharding** | $100\%$ parameter, gradient, and AdamW duplication |

---

## Benchmark Execution

```bash
python -m naive_distributed.run
```
