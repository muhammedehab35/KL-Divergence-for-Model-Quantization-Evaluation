"""
Benchmarking Framework for Quantization Evaluation

Implements the benchmarking methodology used by Unsloth to evaluate
quantization quality across different model sizes and quantization types.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from tqdm import tqdm

from .kl_divergence import KLDivergence
from .quantization import QuantizationType, ModelQuantizer, QUANT_CONFIGS


@dataclass
class BenchmarkResult:
    """Results from a single quantization benchmark."""
    quant_type: str
    model_size_gb: float
    mean_kld: float
    kld_99_9: float
    max_kld: float
    median_kld: float
    std_kld: float
    bits_per_weight: float
    compression_ratio: float
    provider: str = "custom"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(**data)


class QuantizationBenchmark:
    """
    Benchmark quantization methods following Unsloth's methodology.

    This class simulates the benchmarking process Unsloth uses to evaluate
    their GGUF quantizations, measuring KL divergence across different
    quantization types and model sizes.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        num_samples: int = 1000,
        seed: int = 42
    ):
        """
        Initialize benchmark.

        Args:
            vocab_size: Size of model vocabulary
            num_samples: Number of output distributions to test
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seed = seed
        self.quantizer = ModelQuantizer(seed=seed)

        np.random.seed(seed)

    def generate_model_outputs(
        self,
        temperature: float = 1.0,
        use_realistic_distribution: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simulated model outputs (logits and probabilities).

        Args:
            temperature: Softmax temperature
            use_realistic_distribution: If True, use more realistic
                (peaky) distributions similar to real LLMs

        Returns:
            Tuple of (logits, probabilities)
        """
        if use_realistic_distribution:
            # Real LLMs often have peaky distributions
            # Use Zipf-like distribution for more realism
            logits = np.random.randn(self.num_samples, self.vocab_size)

            # Make distributions more peaky (like real LLMs)
            for i in range(self.num_samples):
                # Randomly boost some logits
                top_k = 100
                top_indices = np.random.choice(self.vocab_size, top_k, replace=False)
                logits[i, top_indices] += np.random.uniform(2, 5, top_k)
        else:
            # Simple random logits
            logits = np.random.randn(self.num_samples, self.vocab_size)

        # Convert to probabilities
        probs = self.quantizer.logits_to_probs(logits, temperature)

        return logits, probs

    def benchmark_quantization(
        self,
        quant_type: QuantizationType,
        original_probs: np.ndarray,
        base_model_size_gb: float = 16.0,
        provider: str = "custom"
    ) -> BenchmarkResult:
        """
        Benchmark a single quantization type.

        Args:
            quant_type: Type of quantization to test
            original_probs: Original model output probabilities
            base_model_size_gb: Size of full-precision model in GB
            provider: Name of quantization provider

        Returns:
            BenchmarkResult with statistics
        """
        # Simulate quantized model outputs
        from quantization import WeightQuantizer
        quantized_probs = WeightQuantizer.add_quantization_noise(
            original_probs,
            quant_type,
            seed=self.seed
        )

        # Calculate KL divergence statistics
        kl_stats = KLDivergence.calculate_statistics(original_probs, quantized_probs)

        # Calculate quantized model size
        config = QUANT_CONFIGS[quant_type]
        quantized_size_gb = base_model_size_gb * config.compression_ratio

        return BenchmarkResult(
            quant_type=quant_type.value,
            model_size_gb=quantized_size_gb,
            mean_kld=kl_stats['mean_kld'],
            kld_99_9=kl_stats['kld_99_9'],
            max_kld=kl_stats['max_kld'],
            median_kld=kl_stats['median_kld'],
            std_kld=kl_stats['std_kld'],
            bits_per_weight=config.bits,
            compression_ratio=config.compression_ratio,
            provider=provider
        )

    def run_benchmark_suite(
        self,
        quant_types: Optional[List[QuantizationType]] = None,
        base_model_size_gb: float = 35.0,
        temperature: float = 1.0,
        show_progress: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks across multiple quantization types.

        This replicates Unsloth's benchmark process for comparing
        quantization methods.

        Args:
            quant_types: List of quantization types to test
            base_model_size_gb: Base model size (e.g., 35GB for Qwen3.6-35B)
            temperature: Sampling temperature
            show_progress: Show progress bar

        Returns:
            List of BenchmarkResults
        """
        if quant_types is None:
            # Default: test common quantization types
            quant_types = [
                QuantizationType.IQ2_XXS,
                QuantizationType.Q2_K,
                QuantizationType.Q2_K_XL,
                QuantizationType.IQ3_XXS,
                QuantizationType.Q3_K_M,
                QuantizationType.Q3_K_XL,
                QuantizationType.Q4_K_M,
                QuantizationType.Q4_K_XL,
                QuantizationType.Q5_K_M,
                QuantizationType.Q5_K_XL,
                QuantizationType.Q6_K,
                QuantizationType.Q6_K_XL,
                QuantizationType.Q8_0,
            ]

        # Generate reference outputs
        print(f"Generating {self.num_samples} model outputs...")
        _, original_probs = self.generate_model_outputs(temperature=temperature)

        # Run benchmarks
        results = []
        iterator = tqdm(quant_types) if show_progress else quant_types

        for quant_type in iterator:
            if show_progress:
                iterator.set_description(f"Benchmarking {quant_type.value}")

            result = self.benchmark_quantization(
                quant_type,
                original_probs,
                base_model_size_gb
            )
            results.append(result)

        return results

    def save_results(
        self,
        results: List[BenchmarkResult],
        output_path: str
    ) -> None:
        """
        Save benchmark results to JSON file.

        Args:
            results: List of benchmark results
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': {
                'vocab_size': self.vocab_size,
                'num_samples': self.num_samples,
                'seed': self.seed
            },
            'results': [r.to_dict() for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {output_path}")

    @staticmethod
    def load_results(input_path: str) -> Tuple[dict, List[BenchmarkResult]]:
        """
        Load benchmark results from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Tuple of (metadata, results)
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        metadata = data['metadata']
        results = [BenchmarkResult.from_dict(r) for r in data['results']]

        return metadata, results


class ComparativeBenchmark:
    """
    Compare quantizations from different providers.

    Replicates Unsloth's comparative analysis showing how their
    quantizations compare to other providers (bartowski, AesSedai, etc.)
    """

    def __init__(self, vocab_size: int = 50000, num_samples: int = 1000):
        """Initialize comparative benchmark."""
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def compare_providers(
        self,
        quant_type: QuantizationType,
        providers: Dict[str, float],  # provider_name -> noise_scale
        original_probs: np.ndarray,
        base_size_gb: float = 35.0
    ) -> List[BenchmarkResult]:
        """
        Compare same quantization type from different providers.

        Args:
            quant_type: Quantization type to compare
            providers: Dict mapping provider name to their noise scale
            original_probs: Original model outputs
            base_size_gb: Base model size

        Returns:
            List of results for each provider
        """
        results = []

        for provider_name, noise_scale in providers.items():
            # Simulate each provider's quantization quality
            # (better providers have lower noise)
            from quantization import WeightQuantizer

            # Add provider-specific noise
            config = QUANT_CONFIGS[quant_type]
            base_noise = 0.02 / config.bits
            adjusted_noise = base_noise * noise_scale

            noise = np.random.normal(0, adjusted_noise, original_probs.shape)
            quantized_probs = original_probs + noise
            quantized_probs = np.abs(quantized_probs)
            quantized_probs = quantized_probs / quantized_probs.sum(axis=-1, keepdims=True)

            # Calculate KL stats
            kl_stats = KLDivergence.calculate_statistics(original_probs, quantized_probs)

            # Create result
            quantized_size = base_size_gb * config.compression_ratio

            result = BenchmarkResult(
                quant_type=quant_type.value,
                model_size_gb=quantized_size,
                mean_kld=kl_stats['mean_kld'],
                kld_99_9=kl_stats['kld_99_9'],
                max_kld=kl_stats['max_kld'],
                median_kld=kl_stats['median_kld'],
                std_kld=kl_stats['std_kld'],
                bits_per_weight=config.bits,
                compression_ratio=config.compression_ratio,
                provider=provider_name
            )

            results.append(result)

        return results


def print_benchmark_results(results: List[BenchmarkResult]) -> None:
    """
    Print benchmark results in a nice table format.

    Args:
        results: List of benchmark results
    """
    print("\n" + "=" * 100)
    print("Benchmark Results")
    print("=" * 100)
    print(f"{'Quant Type':<15} {'Provider':<12} {'Size (GB)':<12} "
          f"{'Mean KLD':<12} {'99.9% KLD':<12} {'Max KLD':<12}")
    print("-" * 100)

    for result in sorted(results, key=lambda x: x.model_size_gb):
        print(f"{result.quant_type:<15} {result.provider:<12} "
              f"{result.model_size_gb:<12.2f} {result.mean_kld:<12.6f} "
              f"{result.kld_99_9:<12.6f} {result.max_kld:<12.6f}")

    print("=" * 100 + "\n")


if __name__ == "__main__":
    print("Quantization Benchmarking Demo")
    print("=" * 80)

    # Initialize benchmark
    benchmark = QuantizationBenchmark(
        vocab_size=50000,
        num_samples=500,  # Reduced for demo
        seed=42
    )

    # Run benchmark suite
    print("\nRunning benchmark suite...")
    results = benchmark.run_benchmark_suite(
        base_model_size_gb=35.0,  # Simulating Qwen3.6-35B
        show_progress=True
    )

    # Print results
    print_benchmark_results(results)

    # Save results
    benchmark.save_results(results, "results/benchmark_results.json")

    # Demo: Comparative benchmark
    print("\n\nComparative Benchmark (Different Providers)")
    print("-" * 80)

    _, original_probs = benchmark.generate_model_outputs()

    comparative = ComparativeBenchmark(vocab_size=50000, num_samples=500)

    # Simulate different providers
    # Lower noise_scale = better quantization
    providers = {
        "Unsloth": 0.8,      # Best (as shown in their benchmarks)
        "AesSedai": 1.0,     # Good
        "bartowski": 1.2,    # Decent
        "Generic": 1.5       # Standard
    }

    comparison_results = comparative.compare_providers(
        QuantizationType.Q4_K_M,
        providers,
        original_probs[:500],
        base_size_gb=35.0
    )

    print_benchmark_results(comparison_results)

    print("\nKey Insights:")
    print("- Lower Mean KLD = Better fidelity to original model")
    print("- 99.9% KLD captures outliers without being too sensitive")
    print("- Unsloth optimizes layer-by-layer for minimal KLD")
