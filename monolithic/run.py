import argparse
from monolithic.trainer import MonolithicTrainer


def main():
    parser = argparse.ArgumentParser(description="Monolithic Synchronous Baseline Benchmark")
    parser.add_argument("--steps", type=int, default=100, help="Number of benchmark steps")
    parser.add_argument("--env", type=str, default="antmaze-large-navigate-v0", help="OGBench environment")
    parser.add_argument("--batch-size", type=int, default=32, help="SGD batch size")
    args = parser.parse_args()

    print("=" * 70)
    print("MONOLITHIC BASELINE: Synchronous Lockstep Benchmark")
    print("=" * 70)

    trainer = MonolithicTrainer(env_name=args.env, batch_size=args.batch_size)
    metrics = trainer.run_benchmark(num_steps=args.steps)

    print(f"Total Steps:              {metrics['steps']}")
    print(f"Total Elapsed Time:       {metrics['total_elapsed_sec']:.2f} s")
    print(f"Throughput:               {metrics['throughput_steps_per_sec']:.2f} steps/sec")
    print(f"Avg Env Physics Time:     {metrics['avg_env_step_ms']:.2f} ms")
    print(f"Avg GPU Inference Time:   {metrics['avg_inference_ms']:.2f} ms")
    print(f"Avg GPU SGD Time:         {metrics['avg_sgd_ms']:.2f} ms")
    print(f"Calculated GPU Idle Time: {metrics['gpu_idle_percentage']:.1f}%")
    print(f"Model Parameters:         {metrics['model_parameters']:,}")
    print("=" * 70)


if __name__ == '__main__':
    main()
