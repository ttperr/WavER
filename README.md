# WavER: A Unified Pipeline for Scalable Entity Resolution

**Advancing Entity Resolution with Supervised, Unsupervised, Graph-Based, and Large Language Model Approaches**

## 📌 Overview

WavER (Wavestone Entity Resolution) is a comprehensive entity resolution pipeline designed to improve accuracy and scalability across diverse datasets. It integrates various blocking techniques, supervised and unsupervised matching models, and explores recent advances in zero-shot and few-shot learning using Large Language Models (LLMs).

This project builds on benchmark results in entity resolution, comparing against state-of-the-art methods from _Papers with Code_ and other industry benchmarks.

## 🚀 Features

- **Blocking Techniques**: Scalable candidate pair reduction to improve efficiency with graph-based and rule-based strategies.
- **Pairwise Matching Models**: Supervised and unsupervised approaches for entity matching.
- **LLM-Based Matching**: Zero-shot and few-shot learning approaches for adaptable resolution.
- **Benchmark Comparisons**: Evaluation against existing entity resolution models.

## 📂 Repository Structure

```bash
WavER/
│── data/ # Datasets and preprocessed data
│── graphs/ # Graphs representations for entity resolution
│── model/ # Trained models and evaluation scripts
│── papers/ # Related research papers and benchmarks
│── results/ # Benchmark comparisons and results
│── src/ # Core pipeline implementation
│ │── app/ # StreamLit application for interactive testing
│ │── bert/ # Pairwise entity matching models
│ │── blocking/ # Blocking strategies
│ │── cross-encoder/ # Graph-based entity resolution
│ │── zero-shot/ # Zero-shot and few-shot entity matching
│── README.md # Project documentation
│── requirements.txt # Dependencies
```

## 📊 Results

The pipeline has been tested on multiple datasets, demonstrating improvements in both accuracy and efficiency. Key findings include:

- **Reduction in candidate pairs via efficient blocking** (X% improvement in speed).
- **Supervised models outperform traditional baselines** (Y% higher F1-score).
- **Graph-based methods enhance clustering performance** (Z% better precision-recall tradeoff).
- **LLM-based zero-shot matching provides robust generalization** in unseen datasets.

For detailed results, see the [Results Section](./results/) or refer to the [Master's Thesis](#).

## 🔧 Installation

To set up the environment:

```bash
git clone https://github.com/ttperr/WavER.git
cd WavER
pip install -r requirements.txt
```

## ▶️ Usage

Run the POC using:

```bash
streamlit run run.py
```

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@article{yourarticle,
  title={WavER: A Unified Pipeline for Scalable Entity Resolution},
  author={Your Name},
  journal={Journal of Data Science},
  year={2022}
}
```

## 📫 Contact

For questions or feedback, please reach out to me.
