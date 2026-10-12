# Telemetry: Empirical Benchmarking Methodology and Profiling Suite

<div align="center">

**A Reproducible Measurement Framework for Distributed RL Post-Training Systems**

</div>

---

## Purpose and Scope

The **Telemetry** package provides an automated, statistically rigorous benchmarking harness designed to capture precise hardware timing, memory consumption, and kernel concurrency profiles across all four architectural stages under identical seeds and environment configurations.

---

## Metrics Captured and Instrumentation Methodology

1. **Parameter Synchronization Latency ($T_{\text{sync}}$)**:
   * **Ray Plasma**: High-resolution timers bracket the `ray.put(state_dict)` and `ray.get()` serialization roundtrip over CPU host RAM.
   * **CUDA IPC**: CUDA Events (`torch.cuda.Event(enable_timing=True)`) measure direct GPU peer memory handle acquisition and unpack kernel execution.
   * **Ping-Pong Swap**: Nanosecond-resolution CPU timers measure pointer reassignment and event query overhead.
2. **Replay Buffer Ingestion and Sampling Throughput**:
   * Evaluates vectorized slice assignment rate versus sequential `dist.broadcast` blocking collective calls across world size $W$.
3. **GPU Compute Bubble and Concurrency Overlap**:
   * Measures percentage of idle GPU streaming multiprocessor cycles during environment stepping.
   * Exports PyTorch Profiler traces to visualize dual CUDA stream concurrency.

---

## Running the Telemetry Suite

### 1. Run Full Automated Multi-Architecture Benchmark
```bash
python -m telemetry.benchmark_suite
```
*Outputs structured JSON metrics to `telemetry/results/benchmark_summary.json`.*

### 2. Export Markdown Summary Report
```bash
python -m telemetry.export_metrics
```
*Generates publication-ready comparative tables in `telemetry/results/telemetry_summary.md`.*

### 3. Capture Chrome Concurrency Traces (Perfetto)
```bash
python -m telemetry.profile_streams
```
*Outputs `telemetry/results/stream_concurrency_trace.json`. Drop into [ui.perfetto.dev](https://ui.perfetto.dev) to inspect kernel concurrency timelines.*

### 4. Capture Native NVIDIA Nsight Systems Traces
Generate a native hardware trace capturing SM kernel execution, Tensor Core activity, and CUDA stream concurrency:
```bash
nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true -o telemetry/results/nsight_report python benchmarks/profile_overlap.py
```
Generated artifacts:
* `telemetry/results/nsight_report.nsys-rep`: Open directly in the desktop **NVIDIA Nsight Systems GUI** to inspect multi-stream kernel concurrency.
* `telemetry/results/nsight_report.sqlite`: Queryable SQLite database containing fine-grained CUDA runtime events and memory transfer stats.
