"""
Visualization Tools for Quantization Benchmarks

Creates graphs and plots similar to those used by Unsloth to visualize
KL divergence benchmarks and quantization analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from .benchmarks import BenchmarkResult


# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BenchmarkVisualizer:
    """
    Create visualizations for quantization benchmarks.

    Generates plots similar to Unsloth's benchmark graphs showing
    KL divergence vs model size, Pareto frontiers, etc.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

        # Provider color mapping (matching Unsloth's style)
        self.provider_colors = {
            'Unsloth': '#10b981',  # Green
            'unsloth': '#10b981',
            'AesSedai': '#3b82f6',  # Blue
            'bartowski': '#f59e0b',  # Orange
            'MaziyarPanahi': '#8b5cf6',  # Purple
            'DeepSeek': '#ef4444',  # Red
            'Generic': '#6b7280',  # Gray
            'custom': '#10b981'
        }

    def plot_kld_vs_size(
        self,
        results: List[BenchmarkResult],
        metric: str = 'mean_kld',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_annotations: bool = True
    ) -> None:
        """
        Plot KL divergence vs model size (Unsloth's primary visualization).

        Args:
            results: List of benchmark results
            metric: Which KLD metric to plot ('mean_kld', 'kld_99_9', 'max_kld')
            title: Plot title
            save_path: Path to save figure
            show_annotations: Whether to annotate points with quant type
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Group by provider
        providers = {}
        for result in results:
            provider = result.provider
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(result)

        # Plot each provider
        for provider, provider_results in providers.items():
            sizes = [r.model_size_gb for r in provider_results]
            kld_values = [getattr(r, metric) for r in provider_results]
            quant_types = [r.quant_type for r in provider_results]

            color = self.provider_colors.get(provider, '#6b7280')

            # Plot line
            sorted_indices = np.argsort(sizes)
            sorted_sizes = np.array(sizes)[sorted_indices]
            sorted_kld = np.array(kld_values)[sorted_indices]

            ax.plot(sorted_sizes, sorted_kld, 'o-',
                   color=color, label=provider,
                   linewidth=2, markersize=8, alpha=0.7)

            # Add annotations
            if show_annotations:
                for i, (size, kld, qtype) in enumerate(zip(sizes, kld_values, quant_types)):
                    if i % 2 == 0:  # Annotate every other point to avoid crowding
                        ax.annotate(qtype, (size, kld),
                                  textcoords="offset points",
                                  xytext=(0, 10), ha='center',
                                  fontsize=8, alpha=0.7)

        ax.set_xlabel('Disk Model Size (GB)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')

        if title is None:
            title = f'GGUF Performance Benchmarks\n{metric.replace("_", " ").title()} vs Model Size'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add note about KLD
        note = "Lower is better → indicates higher fidelity to original model"
        fig.text(0.5, 0.02, note, ha='center', fontsize=9, style='italic', alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_pareto_frontier(
        self,
        results: List[BenchmarkResult],
        metric: str = 'kld_99_9',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot Pareto frontier showing optimal size/quality trade-offs.

        Args:
            results: List of benchmark results
            metric: KLD metric to use for quality axis
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Extract data
        sizes = np.array([r.model_size_gb for r in results])
        kld_values = np.array([getattr(r, metric) for r in results])
        providers = [r.provider for r in results]
        quant_types = [r.quant_type for r in results]

        # Plot all points
        for provider in set(providers):
            mask = np.array([p == provider for p in providers])
            color = self.provider_colors.get(provider, '#6b7280')

            ax.scatter(sizes[mask], kld_values[mask],
                      c=color, label=provider, s=100, alpha=0.6, edgecolors='black')

        # Find and plot Pareto frontier
        pareto_mask = np.zeros(len(results), dtype=bool)
        for i in range(len(results)):
            # A point is Pareto optimal if no other point has both
            # smaller size AND smaller KLD
            is_pareto = True
            for j in range(len(results)):
                if i != j:
                    if sizes[j] <= sizes[i] and kld_values[j] < kld_values[i]:
                        is_pareto = False
                        break
            pareto_mask[i] = is_pareto

        # Draw Pareto frontier
        pareto_sizes = sizes[pareto_mask]
        pareto_kld = kld_values[pareto_mask]
        sorted_idx = np.argsort(pareto_sizes)

        ax.plot(pareto_sizes[sorted_idx], pareto_kld[sorted_idx],
               'r--', linewidth=2, label='Pareto Frontier', alpha=0.8)

        ax.set_xlabel('Model Size (GB)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title('Pareto Frontier: Size vs Quality Trade-off',
                    fontsize=14, fontweight='bold', pad=20)

        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_provider_comparison(
        self,
        results: List[BenchmarkResult],
        quant_type_filter: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare providers across different metrics (bar chart).

        Args:
            results: List of benchmark results
            quant_type_filter: Only show this quantization type (optional)
            save_path: Path to save figure
        """
        # Filter by quant type if specified
        if quant_type_filter:
            results = [r for r in results if r.quant_type == quant_type_filter]

        if not results:
            print("No results to plot")
            return

        # Group by provider
        providers = {}
        for result in results:
            provider = result.provider
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(result)

        # Prepare data
        provider_names = list(providers.keys())
        mean_klds = [np.mean([r.mean_kld for r in providers[p]]) for p in provider_names]
        kld_99_9s = [np.mean([r.kld_99_9 for r in providers[p]]) for p in provider_names]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi)

        # Mean KLD comparison
        colors = [self.provider_colors.get(p, '#6b7280') for p in provider_names]
        bars1 = ax1.bar(provider_names, mean_klds, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean KL Divergence', fontsize=11, fontweight='bold')
        ax1.set_title('Mean KLD by Provider', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)

        # 99.9% KLD comparison
        bars2 = ax2.bar(provider_names, kld_99_9s, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('99.9% KL Divergence', fontsize=11, fontweight='bold')
        ax2.set_title('99.9% KLD by Provider (Outlier Detection)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)

        title_suffix = f" ({quant_type_filter})" if quant_type_filter else ""
        fig.suptitle(f'Provider Comparison{title_suffix}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_bits_vs_kld(
        self,
        results: List[BenchmarkResult],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot bits per weight vs KL divergence.

        Shows the relationship between quantization level and quality.

        Args:
            results: List of benchmark results
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Group by provider
        providers = {}
        for result in results:
            provider = result.provider
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(result)

        # Plot each provider
        for provider, provider_results in providers.items():
            bits = [r.bits_per_weight for r in provider_results]
            mean_klds = [r.mean_kld for r in provider_results]
            quant_types = [r.quant_type for r in provider_results]

            color = self.provider_colors.get(provider, '#6b7280')

            ax.scatter(bits, mean_klds, c=color, label=provider,
                      s=150, alpha=0.6, edgecolors='black')

            # Add annotations for some points
            for i, (b, k, qt) in enumerate(zip(bits, mean_klds, quant_types)):
                if i % 3 == 0:
                    ax.annotate(qt, (b, k), fontsize=8, alpha=0.7,
                              xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Bits per Weight', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean KL Divergence', fontsize=12, fontweight='bold')
        ax.set_title('Quantization Level vs Quality',
                    fontsize=14, fontweight='bold', pad=20)

        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def create_summary_dashboard(
        self,
        results: List[BenchmarkResult],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.

        Args:
            results: List of benchmark results
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Plot 1: KLD vs Size
        ax1 = fig.add_subplot(gs[0, :])
        self._subplot_kld_vs_size(ax1, results, 'mean_kld')

        # Plot 2: Provider comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._subplot_provider_bars(ax2, results)

        # Plot 3: Bits vs KLD
        ax3 = fig.add_subplot(gs[1, 1])
        self._subplot_bits_vs_kld(ax3, results)

        fig.suptitle('Quantization Benchmark Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def _subplot_kld_vs_size(self, ax, results, metric):
        """Helper for KLD vs Size subplot."""
        providers = {}
        for result in results:
            provider = result.provider
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(result)

        for provider, provider_results in providers.items():
            sizes = [r.model_size_gb for r in provider_results]
            kld_values = [getattr(r, metric) for r in provider_results]

            color = self.provider_colors.get(provider, '#6b7280')
            sorted_idx = np.argsort(sizes)

            ax.plot(np.array(sizes)[sorted_idx], np.array(kld_values)[sorted_idx],
                   'o-', color=color, label=provider, linewidth=2, markersize=6)

        ax.set_xlabel('Model Size (GB)', fontweight='bold')
        ax.set_ylabel('Mean KL Divergence', fontweight='bold')
        ax.set_title('KL Divergence vs Model Size', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _subplot_provider_bars(self, ax, results):
        """Helper for provider comparison subplot."""
        providers = {}
        for result in results:
            provider = result.provider
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(result)

        provider_names = list(providers.keys())
        mean_klds = [np.mean([r.mean_kld for r in providers[p]]) for p in provider_names]
        colors = [self.provider_colors.get(p, '#6b7280') for p in provider_names]

        bars = ax.bar(provider_names, mean_klds, color=colors, alpha=0.7)
        ax.set_ylabel('Mean KL Divergence', fontweight='bold')
        ax.set_title('Average KLD by Provider', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    def _subplot_bits_vs_kld(self, ax, results):
        """Helper for bits vs KLD subplot."""
        providers = {}
        for result in results:
            provider = result.provider
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(result)

        for provider, provider_results in providers.items():
            bits = [r.bits_per_weight for r in provider_results]
            klds = [r.mean_kld for r in provider_results]
            color = self.provider_colors.get(provider, '#6b7280')

            ax.scatter(bits, klds, c=color, label=provider, s=80, alpha=0.6)

        ax.set_xlabel('Bits per Weight', fontweight='bold')
        ax.set_ylabel('Mean KL Divergence', fontweight='bold')
        ax.set_title('Quantization Level vs Quality', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    # Demo visualization with synthetic data
    print("Creating demo visualizations...")

    # Create some sample results
    from .quantization import QuantizationType

    sample_results = []
    base_size = 35.0

    quant_configs = [
        (QuantizationType.Q2_K, 0.15),
        (QuantizationType.Q4_K_M, 0.28),
        (QuantizationType.Q6_K, 0.41),
        (QuantizationType.Q8_0, 0.53),
    ]

    providers_quality = {
        'Unsloth': 1.0,
        'AesSedai': 1.2,
        'bartowski': 1.4
    }

    for quant_type, ratio in quant_configs:
        for provider, quality_factor in providers_quality.items():
            from .benchmarks import BenchmarkResult
            from .quantization import QUANT_CONFIGS

            config = QUANT_CONFIGS[quant_type]
            size = base_size * ratio

            # Simulate KLD values
            base_kld = 0.001 * (8 - config.bits) ** 2
            mean_kld = base_kld * quality_factor
            kld_99_9 = mean_kld * 3
            max_kld = mean_kld * 5

            result = BenchmarkResult(
                quant_type=quant_type.value,
                model_size_gb=size,
                mean_kld=mean_kld,
                kld_99_9=kld_99_9,
                max_kld=max_kld,
                median_kld=mean_kld * 0.9,
                std_kld=mean_kld * 0.3,
                bits_per_weight=config.bits,
                compression_ratio=ratio,
                provider=provider
            )
            sample_results.append(result)

    # Create visualizations
    viz = BenchmarkVisualizer(figsize=(12, 7))

    print("\n1. KLD vs Size Plot")
    viz.plot_kld_vs_size(sample_results, save_path="results/kld_vs_size.png")

    print("\n2. Pareto Frontier")
    viz.plot_pareto_frontier(sample_results, save_path="results/pareto_frontier.png")

    print("\n3. Provider Comparison")
    viz.plot_provider_comparison(sample_results, save_path="results/provider_comparison.png")

    print("\n4. Complete Dashboard")
    viz.create_summary_dashboard(sample_results, save_path="results/dashboard.png")

    print("\nVisualization demo complete!")
