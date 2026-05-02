"""
KL Divergence for Model Quantization Evaluation

A comprehensive toolkit for understanding and applying KL divergence
to evaluate model quantization quality, inspired by Unsloth's methodology.
"""

from .kl_divergence import (
    KLDivergence,
    calculate_kl_divergence
)

from .quantization import (
    QuantizationType,
    QuantizationConfig,
    WeightQuantizer,
    ModelQuantizer,
    get_quantization_info,
    QUANT_CONFIGS
)

from .benchmarks import (
    BenchmarkResult,
    QuantizationBenchmark,
    ComparativeBenchmark,
    print_benchmark_results
)

from .visualization import (
    BenchmarkVisualizer
)

__version__ = "1.0.0"
__author__ = "Educational Project"
__all__ = [
    # KL Divergence
    'KLDivergence',
    'calculate_kl_divergence',

    # Quantization
    'QuantizationType',
    'QuantizationConfig',
    'WeightQuantizer',
    'ModelQuantizer',
    'get_quantization_info',
    'QUANT_CONFIGS',

    # Benchmarking
    'BenchmarkResult',
    'QuantizationBenchmark',
    'ComparativeBenchmark',
    'print_benchmark_results',

    # Visualization
    'BenchmarkVisualizer',
]
