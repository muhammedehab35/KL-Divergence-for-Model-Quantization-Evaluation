"""
Quick Start Example: KL Divergence for Quantization Evaluation

This script demonstrates the basic usage of the KL divergence toolkit
for evaluating model quantization quality.
"""

import sys
sys.path.append('../src')

import numpy as np
from kl_divergence import KLDivergence, calculate_kl_divergence
from quantization import QuantizationType, ModelQuantizer, compare_quantization_types
from benchmarks import QuantizationBenchmark, print_benchmark_results
from visualization import BenchmarkVisualizer


def example_1_basic_kl_divergence():
    """Example 1: Calculate basic KL divergence between two distributions."""
    print("=" * 80)
    print("Example 1: Basic KL Divergence Calculation")
    print("=" * 80)

    # Create two probability distributions
    original = np.array([0.5, 0.3, 0.15, 0.05])
    quantized = np.array([0.48, 0.32, 0.14, 0.06])

    # Calculate KL divergence
    kl_div = calculate_kl_divergence(original, quantized)

    print(f"\nOriginal distribution:  {original}")
    print(f"Quantized distribution: {quantized}")
    print(f"\nKL Divergence: {kl_div:.6f}")
    print("\nInterpretation: Lower KL divergence = better fidelity to original")
    print("=" * 80 + "\n")


def example_2_batch_calculation():
    """Example 2: Batch KL divergence calculation (simulating model outputs)."""
    print("=" * 80)
    print("Example 2: Batch KL Divergence (Simulating LLM Outputs)")
    print("=" * 80)

    # Simulate 1000 output distributions from a language model
    vocab_size = 50000
    num_outputs = 1000

    # Original model outputs
    np.random.seed(42)
    original_probs = np.random.dirichlet(np.ones(vocab_size), size=num_outputs)

    # Quantized model outputs (with added noise)
    noise = np.random.normal(0, 0.001, original_probs.shape)
    quantized_probs = original_probs + noise
    quantized_probs = np.abs(quantized_probs)
    quantized_probs = quantized_probs / quantized_probs.sum(axis=1, keepdims=True)

    # Calculate statistics
    stats = calculate_kl_divergence(original_probs, quantized_probs, return_statistics=True)

    print(f"\nSimulated {num_outputs} outputs with vocabulary size {vocab_size}")
    print("\nKL Divergence Statistics:")
    print(f"  Mean KL:    {stats['mean_kld']:.6f}")
    print(f"  Median KL:  {stats['median_kld']:.6f}")
    print(f"  99.9% KL:   {stats['kld_99_9']:.6f}")
    print(f"  Max KL:     {stats['max_kld']:.6f}")
    print(f"  Std Dev:    {stats['std_kld']:.6f}")

    print("\nThese are the metrics Unsloth uses to evaluate quantization quality!")
    print("=" * 80 + "\n")


def example_3_compare_quantizations():
    """Example 3: Compare different quantization types."""
    print("=" * 80)
    print("Example 3: Comparing Different Quantization Types")
    print("=" * 80)

    # Show available quantization types
    print("\nAvailable Quantization Types:")
    compare_quantization_types()

    # Simulate model outputs
    vocab_size = 10000
    num_outputs = 500

    np.random.seed(42)
    quantizer = ModelQuantizer(seed=42)

    # Generate realistic model outputs
    logits = np.random.randn(num_outputs, vocab_size)
    # Make distributions peaky (like real LLMs)
    for i in range(num_outputs):
        top_indices = np.random.choice(vocab_size, 100, replace=False)
        logits[i, top_indices] += np.random.uniform(2, 5, 100)

    original_probs = quantizer.logits_to_probs(logits)

    # Test different quantization types
    test_types = [
        QuantizationType.Q2_K,
        QuantizationType.Q4_K_M,
        QuantizationType.Q6_K,
        QuantizationType.Q8_0,
    ]

    print("\nQuantization Impact Analysis:")
    print("-" * 80)
    print(f"{'Quant Type':<15} {'Size':<15} {'Mean KLD':<15} {'99.9% KLD':<15}")
    print("-" * 80)

    from quantization import WeightQuantizer

    for quant_type in test_types:
        # Simulate quantization
        quantized_probs = WeightQuantizer.add_quantization_noise(
            original_probs,
            quant_type,
            seed=42
        )

        # Calculate KL divergence
        stats = KLDivergence.calculate_statistics(original_probs, quantized_probs)

        # Get size info
        size_ratio = quantizer.get_size_reduction(quant_type)

        print(f"{quant_type.value:<15} {size_ratio:<15.1%} "
              f"{stats['mean_kld']:<15.6f} {stats['kld_99_9']:<15.6f}")

    print("-" * 80)
    print("\nKey Insight: Lower bit quantizations (Q2_K) save more space but have")
    print("higher KL divergence, indicating lower fidelity to the original model.")
    print("=" * 80 + "\n")


def example_4_full_benchmark():
    """Example 4: Run a complete benchmark suite like Unsloth."""
    print("=" * 80)
    print("Example 4: Full Benchmark Suite (Unsloth-style)")
    print("=" * 80)

    # Initialize benchmark
    benchmark = QuantizationBenchmark(
        vocab_size=10000,  # Smaller for demo
        num_samples=300,   # Fewer samples for speed
        seed=42
    )

    # Run benchmark
    print("\nRunning comprehensive benchmark...")
    results = benchmark.run_benchmark_suite(
        base_model_size_gb=35.0,  # Simulating Qwen3.6-35B
        show_progress=True
    )

    # Display results
    print_benchmark_results(results)

    # Save results
    output_path = "../results/example_benchmark.json"
    benchmark.save_results(results, output_path)

    print(f"\nResults saved to {output_path}")
    print("=" * 80 + "\n")

    return results


def example_5_visualization(results):
    """Example 5: Create visualizations of benchmark results."""
    print("=" * 80)
    print("Example 5: Visualizing Benchmark Results")
    print("=" * 80)

    # Initialize visualizer
    viz = BenchmarkVisualizer(figsize=(12, 7))

    print("\nCreating visualizations...")

    # 1. KLD vs Size plot (Unsloth's signature visualization)
    print("  - KL Divergence vs Model Size")
    viz.plot_kld_vs_size(
        results,
        metric='mean_kld',
        title='GGUF Performance Benchmarks\nMean KL Divergence vs Model Size',
        save_path='../results/kld_vs_size.png'
    )

    # 2. Pareto frontier
    print("  - Pareto Frontier")
    viz.plot_pareto_frontier(
        results,
        metric='kld_99_9',
        save_path='../results/pareto_frontier.png'
    )

    # 3. Bits vs KLD
    print("  - Quantization Level vs Quality")
    viz.plot_bits_vs_kld(
        results,
        save_path='../results/bits_vs_kld.png'
    )

    print("\nAll visualizations saved to ../results/")
    print("=" * 80 + "\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" KL Divergence for Model Quantization - Quick Start Examples")
    print("=" * 80 + "\n")

    # Run examples
    example_1_basic_kl_divergence()
    example_2_batch_calculation()
    example_3_compare_quantizations()
    results = example_4_full_benchmark()
    example_5_visualization(results)

    print("=" * 80)
    print(" All examples completed successfully!")
    print(" Check the ../results/ directory for saved outputs.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
