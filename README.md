# KL Divergence for Model Quantization Evaluation

An educational project that implements Kullback-Leibler (KL) divergence from scratch and demonstrates its use for evaluating the quality of language model quantizations, inspired by Unsloth's methodology.

## 🎯 Objectives

1. **Understand KL Divergence** - Complete mathematical implementation
2. **Evaluate Quantizations** - Measure quantized model fidelity
3. **Visualize Results** - Create graphs like Unsloth's benchmarks
4. **Analyze Sensitivity** - Identify which layers are sensitive to quantization

## 📚 Project Contents

### `/src`
- `kl_divergence.py` - KL divergence implementation from scratch
- `quantization.py` - Model quantization simulator
- `benchmarks.py` - Framework for benchmarking quantizations
- `visualization.py` - Results visualization tools

### `/notebooks`
- `01_kl_divergence_tutorial.ipynb` - Theoretical and practical introduction
- `02_quantization_analysis.ipynb` - Quantization impact analysis
- `03_benchmark_reproduction.ipynb` - Unsloth benchmark reproduction

### `/data`
- Calibration datasets
- Example probability distributions

### `/results`
- Graphs and benchmark results

## 🚀 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Key Concepts

### What is KL Divergence?

KL divergence measures how much a probability distribution Q differs from a reference distribution P:

```
D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```

### Why for Quantization?

As demonstrated by Unsloth and the "Accuracy is Not All You Need" paper:

- **Measures fidelity** to the original model
- **Correlated with "flips"** (answer changes)
- **More reliable than perplexity** alone
- **Detects degradations** before they affect accuracy

## 🔬 Methodology Inspired by Unsloth

1. **Multi-Level Measurement**
   - Mean KLD (average)
   - 99.9% KLD (outliers)
   - Maximum KLD (worst cases)

2. **Layer-by-Layer Analysis**
   - Identify sensitive layers
   - Optimize quantization per tensor type

3. **Combined Benchmarks**
   - KL Divergence for fidelity
   - MMLU for accuracy
   - Real-world benchmarks for validation

## 📊 Usage Example

```python
from src.kl_divergence import calculate_kl_divergence
from src.quantization import quantize_model, QuantizationType
from src.benchmarks import run_benchmark

# Simulate original and quantized model
original_probs = model.get_output_probabilities(text)
quantized_model = quantize_model(model, QuantizationType.Q4_K_M)
quantized_probs = quantized_model.get_output_probabilities(text)

# Calculate KL divergence
kl_div = calculate_kl_divergence(original_probs, quantized_probs)
print(f"KL Divergence: {kl_div:.4f}")

# Benchmark multiple quantizations
results = run_benchmark(model, [
    QuantizationType.Q2_K,
    QuantizationType.Q4_K_M,
    QuantizationType.Q8_0
])
```

## 📈 Expected Results

You will learn to:

- ✅ Calculate KL divergence between distributions
- ✅ Simulate different quantization types
- ✅ Create Pareto plots (KLD vs size)
- ✅ Identify layers sensitive to quantization
- ✅ Understand accuracy/size trade-offs

## 🎓 Ressources

- [Accuracy is Not All You Need (Paper)](https://arxiv.org/pdf/2407.09141)
- [Unsloth Dynamic 2.0 Documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [KL Divergence (Wikipedia)](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

## 🤝 Contributing

This project is for educational purposes. Contributions are welcome!

## 📝 License

MIT License - Educational Project

## 🙏 Acknowledgments

Inspired by the remarkable work of the Unsloth team on optimal language model quantization.
