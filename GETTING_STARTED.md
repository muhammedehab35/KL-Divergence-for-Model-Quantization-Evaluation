# Getting Started with KL Divergence for Quantization

This guide will help you get started with using KL divergence to evaluate model quantization quality, following Unsloth's methodology.

## Quick Start (5 minutes)

### 1. Installation

```bash
# Clone or navigate to the project
cd kl_divergence_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Your First Example

```bash
cd examples
python quick_start.py
```

This will run all examples and create visualizations in the `results/` folder.

## What You'll Learn

### Core Concepts

1. **KL Divergence Basics**
   - What it is mathematically
   - Why it matters for quantization
   - How to calculate it from scratch

2. **Quantization Simulation**
   - Different quantization types (Q2_K, Q4_K_M, Q8_0, etc.)
   - How quantization affects model outputs
   - Simulating quantization noise

3. **Benchmarking**
   - Running comprehensive benchmarks
   - Comparing quantization methods
   - Understanding the metrics (Mean KLD, 99.9% KLD, Max KLD)

4. **Visualization**
   - Creating publication-quality graphs
   - Interpreting Pareto frontiers
   - Understanding trade-offs

## Usage Examples

### Example 1: Basic KL Divergence

```python
from kl_divergence import calculate_kl_divergence
import numpy as np

# Two probability distributions
original = np.array([0.5, 0.3, 0.2])
quantized = np.array([0.48, 0.32, 0.2])

# Calculate KL divergence
kl_div = calculate_kl_divergence(original, quantized)
print(f"KL Divergence: {kl_div:.6f}")
```

### Example 2: Batch Calculation (Simulating LLM)

```python
from kl_divergence import KLDivergence
import numpy as np

# Simulate 1000 model outputs
vocab_size = 50000
num_outputs = 1000

original_probs = np.random.dirichlet(np.ones(vocab_size), size=num_outputs)
quantized_probs = original_probs + np.random.normal(0, 0.001, original_probs.shape)
quantized_probs = np.abs(quantized_probs)
quantized_probs = quantized_probs / quantized_probs.sum(axis=1, keepdims=True)

# Calculate statistics
stats = KLDivergence.calculate_statistics(original_probs, quantized_probs)
print(f"Mean KL: {stats['mean_kld']:.6f}")
print(f"99.9% KL: {stats['kld_99_9']:.6f}")
print(f"Max KL: {stats['max_kld']:.6f}")
```

### Example 3: Compare Quantization Types

```python
from quantization import QuantizationType, ModelQuantizer, WeightQuantizer
from kl_divergence import KLDivergence
import numpy as np

# Setup
quantizer = ModelQuantizer(seed=42)
vocab_size = 10000
num_samples = 500

# Generate model outputs
logits = np.random.randn(num_samples, vocab_size)
original_probs = quantizer.logits_to_probs(logits)

# Test different quantizations
for quant_type in [QuantizationType.Q2_K, QuantizationType.Q4_K_M, QuantizationType.Q8_0]:
    quantized_probs = WeightQuantizer.add_quantization_noise(
        original_probs, quant_type, seed=42
    )

    stats = KLDivergence.calculate_statistics(original_probs, quantized_probs)
    size = quantizer.get_size_reduction(quant_type)

    print(f"{quant_type.value}: Size={size:.1%}, Mean KLD={stats['mean_kld']:.6f}")
```

### Example 4: Full Benchmark Suite

```python
from benchmarks import QuantizationBenchmark, print_benchmark_results

# Initialize
benchmark = QuantizationBenchmark(
    vocab_size=50000,
    num_samples=1000,
    seed=42
)

# Run benchmarks
results = benchmark.run_benchmark_suite(
    base_model_size_gb=35.0,  # Simulating Qwen3.6-35B
    show_progress=True
)

# Display results
print_benchmark_results(results)

# Save results
benchmark.save_results(results, "my_benchmark.json")
```

### Example 5: Visualize Results

```python
from visualization import BenchmarkVisualizer

# Load or use existing results
viz = BenchmarkVisualizer(figsize=(12, 7))

# Create Unsloth-style plot
viz.plot_kld_vs_size(
    results,
    metric='mean_kld',
    title='GGUF Performance Benchmarks',
    save_path='kld_vs_size.png'
)

# Create Pareto frontier
viz.plot_pareto_frontier(
    results,
    metric='kld_99_9',
    save_path='pareto.png'
)

# Create complete dashboard
viz.create_summary_dashboard(results, save_path='dashboard.png')
```

## Interactive Notebooks

For a more interactive learning experience, use the Jupyter notebooks:

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_kl_divergence_tutorial.ipynb
```

The notebooks provide:
- Step-by-step explanations
- Interactive code cells
- Visualizations
- Detailed commentary

## Project Structure

```
kl_divergence_project/
├── src/
│   ├── kl_divergence.py      # Core KL divergence implementation
│   ├── quantization.py        # Quantization simulation
│   ├── benchmarks.py          # Benchmarking framework
│   └── visualization.py       # Plotting tools
├── examples/
│   └── quick_start.py         # Quick start examples
├── notebooks/
│   └── 01_kl_divergence_tutorial.ipynb  # Interactive tutorial
├── results/                   # Generated outputs
├── requirements.txt           # Python dependencies
└── README.md                  # Main documentation
```

## Understanding the Metrics

### Mean KLD
- **What**: Average KL divergence across all outputs
- **Use**: Primary metric for overall quality
- **Interpretation**: Lower = better fidelity to original model

### 99.9% KLD
- **What**: 99.9th percentile of KL divergence
- **Use**: Captures outliers without being too sensitive
- **Interpretation**: Unsloth's preferred robustness metric

### Max KLD
- **What**: Maximum KL divergence observed
- **Use**: Identifies worst-case degradation
- **Interpretation**: Very sensitive, use cautiously

## Tips and Best Practices

### 1. Start Small
- Use smaller vocabulary sizes (1000-10000) for initial experiments
- Reduce num_samples (100-500) for faster iteration
- Scale up once you understand the patterns

### 2. Validate Distributions
- Ensure probabilities sum to 1.0
- Check for negative values
- Use epsilon for numerical stability

### 3. Compare Fairly
- Always use same seed for reproducibility
- Use same calibration data across quantizations
- Test on realistic distributions (peaky, not uniform)

### 4. Interpret Results
- KL divergence is a signal, not the truth
- Always validate with real benchmarks (MMLU, etc.)
- Consider use-case specific requirements

### 5. Visualize
- Use multiple plots to understand trade-offs
- Look for Pareto optimal points
- Compare providers/methods visually

## Common Questions

### Q: What's a "good" KL divergence value?
**A**: It depends on context, but:
- < 0.01: Excellent (minimal degradation)
- 0.01 - 0.1: Good (acceptable for most uses)
- 0.1 - 1.0: Moderate (noticeable differences)
- \> 1.0: High (significant degradation)

### Q: Why use 99.9% instead of mean?
**A**: Mean can be misleading if a few outputs are very bad. 99.9% captures outliers without being too sensitive to single anomalies.

### Q: How does this relate to real models?
**A**: This project simulates the evaluation process. For real models, you'd:
1. Run the full model to get output distributions
2. Run the quantized model on same inputs
3. Calculate KL divergence between them
4. Validate with downstream benchmarks

### Q: Can I use this for my own quantization?
**A**: Yes! The framework is general. You can:
- Add your own quantization methods
- Test on your calibration data
- Compare against baselines
- Analyze layer-specific sensitivity

## Next Steps

1. **Complete the Tutorial Notebook**
   - Work through all cells
   - Experiment with parameters
   - Try different scenarios

2. **Run Custom Benchmarks**
   - Test your own quantization ideas
   - Compare different approaches
   - Analyze results

3. **Understand Unsloth's Work**
   - Read their documentation
   - Study their benchmark graphs
   - Apply learnings to your work

4. **Contribute**
   - Add new features
   - Improve documentation
   - Share your findings

## Resources

- **Unsloth Documentation**: https://unsloth.ai/docs
- **Research Paper**: "Accuracy is Not All You Need" - https://arxiv.org/pdf/2407.09141
- **GGUF Spec**: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- **KL Divergence (Wikipedia)**: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd kl_divergence_project

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Numerical Issues
```python
# Increase epsilon for stability
kl = KLDivergence.calculate(p, q, epsilon=1e-8)

# Validate distributions
from kl_divergence import KLDivergence
KLDivergence._validate_distribution(p, "p")
```

### Memory Issues
```python
# Reduce vocabulary size
vocab_size = 10000  # instead of 50000

# Reduce batch size
num_samples = 500  # instead of 1000

# Process in chunks
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    # process chunk
```

## Support

For questions or issues:
1. Check the documentation
2. Review the examples
3. Run the test notebook
4. Open an issue on GitHub

Happy benchmarking! 🚀
