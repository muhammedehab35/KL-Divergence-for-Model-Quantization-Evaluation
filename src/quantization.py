"""
Model Quantization Simulator

Simulates different quantization schemes used in GGUF format,
following the methodology used by Unsloth for benchmarking.

Quantization types:
- Q2_K: 2-bit quantization
- Q3_K: 3-bit quantization
- Q4_K_M: 4-bit quantization (medium)
- Q4_K_XL: 4-bit quantization (extra large - Unsloth Dynamic)
- Q5_K: 5-bit quantization
- Q6_K: 6-bit quantization
- Q8_0: 8-bit quantization
- IQ2_XXS: 2-bit importance matrix quantization
- IQ3_XXS: 3-bit importance matrix quantization
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


class QuantizationType(Enum):
    """Quantization types matching GGUF formats."""
    # Standard K-quants
    Q2_K = "Q2_K"
    Q3_K_M = "Q3_K_M"
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"

    # Unsloth Dynamic XL quants
    Q2_K_XL = "Q2_K_XL"
    Q3_K_XL = "Q3_K_XL"
    Q4_K_XL = "Q4_K_XL"
    Q5_K_XL = "Q5_K_XL"
    Q6_K_XL = "Q6_K_XL"

    # Importance matrix quants (I-quants)
    IQ2_XXS = "IQ2_XXS"
    IQ2_S = "IQ2_S"
    IQ3_XXS = "IQ3_XXS"
    IQ3_S = "IQ3_S"

    # Full precision
    F16 = "F16"
    BF16 = "BF16"


@dataclass
class QuantizationConfig:
    """Configuration for quantization schemes."""
    bits: float  # Average bits per weight
    block_size: int  # Size of quantization block
    has_importance_matrix: bool  # Uses imatrix
    compression_ratio: float  # Approximate size reduction

    @property
    def effective_bits(self) -> float:
        """Effective bits considering overhead."""
        return self.bits


# Quantization configurations based on GGUF specifications
QUANT_CONFIGS = {
    QuantizationType.F16: QuantizationConfig(16.0, 1, False, 1.0),
    QuantizationType.BF16: QuantizationConfig(16.0, 1, False, 1.0),
    QuantizationType.Q8_0: QuantizationConfig(8.5, 32, False, 0.53),
    QuantizationType.Q6_K: QuantizationConfig(6.5, 256, False, 0.41),
    QuantizationType.Q6_K_XL: QuantizationConfig(6.5, 256, True, 0.40),
    QuantizationType.Q5_K_M: QuantizationConfig(5.5, 256, False, 0.34),
    QuantizationType.Q5_K_XL: QuantizationConfig(5.5, 256, True, 0.33),
    QuantizationType.Q4_K_M: QuantizationConfig(4.5, 256, False, 0.28),
    QuantizationType.Q4_K_XL: QuantizationConfig(4.5, 256, True, 0.27),
    QuantizationType.Q3_K_M: QuantizationConfig(3.5, 256, False, 0.22),
    QuantizationType.Q3_K_XL: QuantizationConfig(3.5, 256, True, 0.21),
    QuantizationType.Q2_K: QuantizationConfig(2.5, 256, False, 0.16),
    QuantizationType.Q2_K_XL: QuantizationConfig(2.5, 256, True, 0.15),
    QuantizationType.IQ3_XXS: QuantizationConfig(3.3, 256, True, 0.20),
    QuantizationType.IQ3_S: QuantizationConfig(3.4, 256, True, 0.21),
    QuantizationType.IQ2_XXS: QuantizationConfig(2.2, 256, True, 0.14),
    QuantizationType.IQ2_S: QuantizationConfig(2.3, 256, True, 0.15),
}


class WeightQuantizer:
    """
    Simulates weight quantization for neural network models.

    This class implements various quantization schemes to compress
    model weights, similar to how GGUF quantization works.
    """

    @staticmethod
    def quantize_uniform(
        weights: np.ndarray,
        n_bits: int,
        symmetric: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Uniform quantization of weights.

        Args:
            weights: Original weights (float32/float16)
            n_bits: Number of bits for quantization
            symmetric: If True, use symmetric quantization around zero

        Returns:
            Tuple of (quantized_weights, quantization_params)
        """
        n_levels = 2 ** n_bits

        if symmetric:
            # Symmetric quantization: [-max_val, max_val]
            max_val = np.abs(weights).max()
            scale = max_val / (n_levels / 2 - 1)
            zero_point = 0
        else:
            # Asymmetric quantization: [min_val, max_val]
            min_val, max_val = weights.min(), weights.max()
            scale = (max_val - min_val) / (n_levels - 1)
            zero_point = -min_val / scale

        # Quantize
        quantized = np.round(weights / scale + zero_point)
        quantized = np.clip(quantized, 0, n_levels - 1)

        # Dequantize
        dequantized = (quantized - zero_point) * scale

        params = {
            'scale': float(scale),
            'zero_point': float(zero_point),
            'n_bits': n_bits,
            'n_levels': n_levels
        }

        return dequantized, params

    @staticmethod
    def quantize_block(
        weights: np.ndarray,
        n_bits: int,
        block_size: int = 32
    ) -> np.ndarray:
        """
        Block-wise quantization (used in GGUF Q4_0, Q8_0, etc.).

        Each block has its own scale factor, improving accuracy
        compared to uniform quantization.

        Args:
            weights: Original weights
            n_bits: Bits per weight
            block_size: Size of each quantization block

        Returns:
            Dequantized weights
        """
        weights_flat = weights.flatten()
        n_blocks = int(np.ceil(len(weights_flat) / block_size))

        # Pad to multiple of block_size
        padded_size = n_blocks * block_size
        padded_weights = np.zeros(padded_size)
        padded_weights[:len(weights_flat)] = weights_flat

        # Reshape into blocks
        blocks = padded_weights.reshape(n_blocks, block_size)

        # Quantize each block
        n_levels = 2 ** n_bits
        quantized_blocks = np.zeros_like(blocks)

        for i, block in enumerate(blocks):
            # Per-block scale
            max_val = np.abs(block).max()
            if max_val > 0:
                scale = max_val / (n_levels / 2 - 1)
                quantized = np.round(block / scale)
                quantized = np.clip(quantized, -(n_levels/2), n_levels/2 - 1)
                quantized_blocks[i] = quantized * scale
            else:
                quantized_blocks[i] = block

        # Reshape back
        result = quantized_blocks.flatten()[:len(weights_flat)]
        return result.reshape(weights.shape)

    @staticmethod
    def add_quantization_noise(
        values: np.ndarray,
        quant_type: QuantizationType,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Add realistic quantization noise based on quantization type.

        This simulates the error introduced by quantization without
        actually quantizing, useful for fast KL divergence estimation.

        Args:
            values: Original values (e.g., probability distributions)
            quant_type: Type of quantization to simulate
            seed: Random seed for reproducibility

        Returns:
            Values with added quantization noise
        """
        if seed is not None:
            np.random.seed(seed)

        config = QUANT_CONFIGS[quant_type]

        if quant_type in [QuantizationType.F16, QuantizationType.BF16]:
            # Full precision - minimal noise
            noise_scale = 1e-7
        elif config.has_importance_matrix:
            # I-matrix quants have better performance
            noise_scale = 0.01 / config.bits
        else:
            # Standard quantization noise
            noise_scale = 0.02 / config.bits

        # Add Gaussian noise proportional to quantization level
        noise = np.random.normal(0, noise_scale, values.shape)
        noisy_values = values + noise

        # Ensure valid probabilities if input was probabilities
        if np.all(values >= 0) and np.allclose(values.sum(axis=-1), 1.0, atol=1e-4):
            noisy_values = np.abs(noisy_values)
            noisy_values = noisy_values / noisy_values.sum(axis=-1, keepdims=True)

        return noisy_values


class ModelQuantizer:
    """
    High-level interface for quantizing models and analyzing impact.

    Simulates the quantization of different model components
    (weights, activations) and their effect on output distributions.
    """

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize quantizer.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.quantizer = WeightQuantizer()

    def quantize_logits(
        self,
        logits: np.ndarray,
        quant_type: QuantizationType
    ) -> np.ndarray:
        """
        Simulate quantization effect on model logits.

        Args:
            logits: Original logits from full-precision model
            quant_type: Quantization type to simulate

        Returns:
            Logits after simulated quantization
        """
        config = QUANT_CONFIGS[quant_type]

        # Apply quantization to logits
        if quant_type in [QuantizationType.F16, QuantizationType.BF16]:
            return logits  # No quantization

        quantized_logits = self.quantizer.quantize_block(
            logits,
            n_bits=int(config.bits),
            block_size=config.block_size
        )

        return quantized_logits

    def logits_to_probs(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Convert logits to probability distributions using softmax.

        Args:
            logits: Model logits
            temperature: Temperature for softmax

        Returns:
            Probability distributions
        """
        # Numerically stable softmax
        logits_scaled = logits / temperature
        logits_max = np.max(logits_scaled, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_scaled - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

    def get_size_reduction(self, quant_type: QuantizationType) -> float:
        """
        Get approximate model size reduction for quantization type.

        Args:
            quant_type: Quantization type

        Returns:
            Size as fraction of original (e.g., 0.28 = 28% of original size)
        """
        return QUANT_CONFIGS[quant_type].compression_ratio


# Convenience functions
def get_quantization_info(quant_type: QuantizationType) -> QuantizationConfig:
    """Get configuration for a quantization type."""
    return QUANT_CONFIGS[quant_type]


def compare_quantization_types() -> None:
    """Print comparison table of all quantization types."""
    print("\n" + "=" * 80)
    print("Quantization Types Comparison")
    print("=" * 80)
    print(f"{'Type':<15} {'Bits':<8} {'Block':<8} {'iMatrix':<10} {'Size Ratio':<12}")
    print("-" * 80)

    for quant_type in QuantizationType:
        config = QUANT_CONFIGS[quant_type]
        imatrix = "Yes" if config.has_importance_matrix else "No"
        print(f"{quant_type.value:<15} {config.bits:<8.1f} {config.block_size:<8} "
              f"{imatrix:<10} {config.compression_ratio:<12.2%}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("Quantization Simulator Demo")
    print("=" * 60)

    # Show quantization types
    compare_quantization_types()

    # Demo: Quantize some weights
    print("\nDemo: Weight Quantization")
    print("-" * 60)

    # Generate random weights
    np.random.seed(42)
    weights = np.random.randn(1000, 512).astype(np.float32)

    quantizer = WeightQuantizer()

    # Test different bit widths
    for n_bits in [2, 4, 8]:
        quantized, params = quantizer.quantize_uniform(weights, n_bits)
        error = np.mean(np.abs(weights - quantized))
        print(f"\n{n_bits}-bit quantization:")
        print(f"  Scale: {params['scale']:.6f}")
        print(f"  Mean absolute error: {error:.6f}")
        print(f"  Max error: {np.max(np.abs(weights - quantized)):.6f}")

    # Demo: Simulate quantization effect on probabilities
    print("\n\nDemo: Quantization Effect on Output Probabilities")
    print("-" * 60)

    # Simulate model output logits
    vocab_size = 50000
    batch_size = 10
    logits = np.random.randn(batch_size, vocab_size).astype(np.float32)

    model_quant = ModelQuantizer()

    # Convert to probabilities
    original_probs = model_quant.logits_to_probs(logits)

    # Test different quantization types
    try:
        from .kl_divergence import KLDivergence
    except ImportError:
        from kl_divergence import KLDivergence

    print("\nQuantization Type Impact:")
    test_types = [
        QuantizationType.Q2_K,
        QuantizationType.Q4_K_M,
        QuantizationType.Q8_0,
        QuantizationType.F16
    ]

    for quant_type in test_types:
        # Simulate quantized model
        noisy_probs = WeightQuantizer.add_quantization_noise(
            original_probs,
            quant_type,
            seed=42
        )

        # Calculate KL divergence
        kl_stats = KLDivergence.calculate_statistics(original_probs, noisy_probs)

        print(f"\n{quant_type.value}:")
        print(f"  Size: {model_quant.get_size_reduction(quant_type):.1%} of original")
        print(f"  Mean KL: {kl_stats['mean_kld']:.6f}")
        print(f"  99.9% KL: {kl_stats['kld_99_9']:.6f}")
        print(f"  Max KL: {kl_stats['max_kld']:.6f}")

    print("\n" + "=" * 60)
