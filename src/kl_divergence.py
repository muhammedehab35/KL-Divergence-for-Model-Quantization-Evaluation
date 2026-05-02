"""
KL Divergence Implementation from Scratch

This module implements Kullback-Leibler divergence following the methodology
used by Unsloth for evaluating model quantization quality.

Based on: "Accuracy is Not All You Need" - https://arxiv.org/pdf/2407.09141
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings


class KLDivergence:
    """
    Kullback-Leibler Divergence Calculator

    Implements various KL divergence metrics used for quantization evaluation:
    - Mean KLD: Average divergence across all tokens
    - 99.9% KLD: Captures outliers (used by Unsloth)
    - Max KLD: Worst-case divergence
    """

    @staticmethod
    def calculate(
        p: np.ndarray,
        q: np.ndarray,
        epsilon: float = 1e-10,
        validate: bool = True
    ) -> float:
        """
        Calculate KL Divergence: D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

        Args:
            p: Reference probability distribution (original model)
            q: Approximation distribution (quantized model)
            epsilon: Small constant to avoid log(0) and division by zero
            validate: Whether to validate input distributions

        Returns:
            KL divergence value (non-negative)

        Mathematical Definition:
            For discrete distributions:
                D_KL(P || Q) = Σ_i P(i) * log(P(i) / Q(i))

            Properties:
                - Always non-negative: D_KL(P || Q) ≥ 0
                - Zero iff P = Q
                - Not symmetric: D_KL(P || Q) ≠ D_KL(Q || P)

        Example:
            >>> p = np.array([0.5, 0.3, 0.2])
            >>> q = np.array([0.4, 0.4, 0.2])
            >>> kl = KLDivergence.calculate(p, q)
            >>> print(f"KL Divergence: {kl:.4f}")
        """
        # Convert to numpy arrays
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        # Validate inputs
        if validate:
            KLDivergence._validate_distribution(p, "p")
            KLDivergence._validate_distribution(q, "q")

            if p.shape != q.shape:
                raise ValueError(f"Shape mismatch: p.shape={p.shape}, q.shape={q.shape}")

        # Add epsilon to avoid division by zero and log(0)
        p_safe = np.clip(p, epsilon, 1.0)
        q_safe = np.clip(q, epsilon, 1.0)

        # Calculate KL divergence
        # Only compute where p > epsilon (avoid unnecessary computation)
        mask = p > epsilon
        kl_div = np.sum(p_safe[mask] * np.log(p_safe[mask] / q_safe[mask]))

        return float(kl_div)

    @staticmethod
    def calculate_batch(
        p_batch: np.ndarray,
        q_batch: np.ndarray,
        epsilon: float = 1e-10
    ) -> np.ndarray:
        """
        Calculate KL divergence for a batch of distributions.

        Args:
            p_batch: Shape (batch_size, vocab_size) - reference distributions
            q_batch: Shape (batch_size, vocab_size) - approximation distributions
            epsilon: Small constant to avoid numerical issues

        Returns:
            Array of KL divergences, shape (batch_size,)

        Example:
            >>> p_batch = np.random.dirichlet(np.ones(100), size=1000)
            >>> q_batch = np.random.dirichlet(np.ones(100), size=1000)
            >>> kl_values = KLDivergence.calculate_batch(p_batch, q_batch)
            >>> print(f"Mean KL: {np.mean(kl_values):.4f}")
        """
        p_batch = np.asarray(p_batch, dtype=np.float64)
        q_batch = np.asarray(q_batch, dtype=np.float64)

        if p_batch.shape != q_batch.shape:
            raise ValueError(f"Shape mismatch: {p_batch.shape} vs {q_batch.shape}")

        # Clip to avoid numerical issues
        p_safe = np.clip(p_batch, epsilon, 1.0)
        q_safe = np.clip(q_batch, epsilon, 1.0)

        # Calculate KL for each distribution in the batch
        kl_values = np.sum(p_safe * np.log(p_safe / q_safe), axis=1)

        return kl_values

    @staticmethod
    def calculate_statistics(
        p_batch: np.ndarray,
        q_batch: np.ndarray,
        epsilon: float = 1e-10
    ) -> dict:
        """
        Calculate KL divergence statistics as used by Unsloth.

        Returns mean, 99.9 percentile, and max KL divergence - the three
        metrics Unsloth uses to evaluate quantization quality.

        Args:
            p_batch: Reference distributions
            q_batch: Approximation distributions
            epsilon: Numerical stability constant

        Returns:
            Dictionary with:
                - mean_kld: Average KL divergence
                - kld_99_9: 99.9th percentile (outlier detection)
                - max_kld: Maximum KL divergence (worst case)
                - median_kld: Median KL divergence
                - std_kld: Standard deviation

        This follows Unsloth's methodology:
        - Mean KLD: Primary metric for overall quality
        - 99.9% KLD: Detects outliers without being too sensitive to single bad tokens
        - Max KLD: Identifies worst-case degradation
        """
        kl_values = KLDivergence.calculate_batch(p_batch, q_batch, epsilon)

        return {
            'mean_kld': float(np.mean(kl_values)),
            'kld_99_9': float(np.percentile(kl_values, 99.9)),
            'max_kld': float(np.max(kl_values)),
            'median_kld': float(np.median(kl_values)),
            'std_kld': float(np.std(kl_values)),
            'min_kld': float(np.min(kl_values))
        }

    @staticmethod
    def js_divergence(
        p: np.ndarray,
        q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Calculate Jensen-Shannon Divergence (symmetric version of KL).

        JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)

        Args:
            p: First distribution
            q: Second distribution
            epsilon: Numerical stability constant

        Returns:
            Jensen-Shannon divergence (always in [0, log(2)])
        """
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        # Calculate mixture distribution
        m = 0.5 * (p + q)

        # Calculate symmetric divergence
        js = 0.5 * KLDivergence.calculate(p, m, epsilon) + \
             0.5 * KLDivergence.calculate(q, m, epsilon)

        return float(js)

    @staticmethod
    def _validate_distribution(dist: np.ndarray, name: str) -> None:
        """Validate that array is a proper probability distribution."""
        if np.any(dist < 0):
            raise ValueError(f"{name} contains negative values")

        total = np.sum(dist)
        if not np.isclose(total, 1.0, atol=1e-6):
            warnings.warn(
                f"{name} does not sum to 1.0 (sum={total:.6f}). "
                "Consider normalizing.",
                UserWarning
            )


def calculate_kl_divergence(
    original_probs: np.ndarray,
    quantized_probs: np.ndarray,
    return_statistics: bool = False
) -> Union[float, dict]:
    """
    Convenience function to calculate KL divergence.

    Args:
        original_probs: Probabilities from original (full-precision) model
        quantized_probs: Probabilities from quantized model
        return_statistics: If True, return full statistics dict

    Returns:
        KL divergence value or statistics dictionary

    Example:
        >>> original = np.array([0.5, 0.3, 0.2])
        >>> quantized = np.array([0.45, 0.35, 0.2])
        >>> kl = calculate_kl_divergence(original, quantized)
    """
    # Handle single distribution
    if original_probs.ndim == 1:
        if return_statistics:
            raise ValueError("Statistics require batch of distributions (2D array)")
        return KLDivergence.calculate(original_probs, quantized_probs)

    # Handle batch of distributions
    if return_statistics:
        return KLDivergence.calculate_statistics(original_probs, quantized_probs)
    else:
        kl_values = KLDivergence.calculate_batch(original_probs, quantized_probs)
        return float(np.mean(kl_values))


if __name__ == "__main__":
    # Demo: Simple KL divergence calculation
    print("=" * 60)
    print("KL Divergence Demo")
    print("=" * 60)

    # Example 1: Identical distributions
    print("\n1. Identical distributions (should be ~0):")
    p1 = np.array([0.5, 0.3, 0.2])
    q1 = np.array([0.5, 0.3, 0.2])
    kl1 = KLDivergence.calculate(p1, q1)
    print(f"   KL(P || Q) = {kl1:.6f}")

    # Example 2: Slightly different distributions
    print("\n2. Slightly perturbed distribution:")
    p2 = np.array([0.5, 0.3, 0.2])
    q2 = np.array([0.45, 0.35, 0.2])
    kl2 = KLDivergence.calculate(p2, q2)
    print(f"   Original: {p2}")
    print(f"   Quantized: {q2}")
    print(f"   KL(P || Q) = {kl2:.6f}")

    # Example 3: Asymmetry demonstration
    print("\n3. KL divergence is NOT symmetric:")
    kl_pq = KLDivergence.calculate(p2, q2)
    kl_qp = KLDivergence.calculate(q2, p2)
    print(f"   KL(P || Q) = {kl_pq:.6f}")
    print(f"   KL(Q || P) = {kl_qp:.6f}")
    print(f"   Difference: {abs(kl_pq - kl_qp):.6f}")

    # Example 4: Batch calculation (simulating model outputs)
    print("\n4. Batch calculation (1000 token distributions):")
    vocab_size = 50000
    batch_size = 1000

    # Simulate original model outputs
    p_batch = np.random.dirichlet(np.ones(vocab_size), size=batch_size)
    # Simulate quantized model (with added noise)
    q_batch = p_batch + np.random.normal(0, 0.01, p_batch.shape)
    q_batch = np.abs(q_batch)  # Ensure positive
    q_batch = q_batch / q_batch.sum(axis=1, keepdims=True)  # Renormalize

    stats = KLDivergence.calculate_statistics(p_batch, q_batch)
    print(f"   Mean KL: {stats['mean_kld']:.6f}")
    print(f"   99.9% KL: {stats['kld_99_9']:.6f}")
    print(f"   Max KL: {stats['max_kld']:.6f}")
    print(f"   Median KL: {stats['median_kld']:.6f}")

    print("\n" + "=" * 60)
