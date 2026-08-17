# Evaluating the Effectiveness of Multi-Agent Systems Based on Small Language Models Compared to Larger Parameter Models

This repository contains the implementation and research data for the study comparing Large Language Models (LLMs) with Multi-Agent Systems (MAS) based on Small Language Models (SLMs), both augmented with Retrieval-Augmented Generation (RAG).

## Project Structure

- `experiment.py` — Main execution script for all three experiments.
- `abstract.md` — Research abstract.
- `methodology.md` — Detailed experimental design.
- `discussion.md` — Results analysis and conclusions.

## Setup and Installation

### 1. Install Dependencies

Ensure you have Python 3.10+ installed. It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows
```

Install the required libraries:

```bash
pip install torch transformers sentence-transformers accelerate
```

### 2. Model Downloading

The project uses models from the Qwen family. The script will automatically download them from Hugging Face upon the first run, but you can also ensure they are available in your cache.

**Models used:**

- **SLM:** `Qwen/Qwen3.5-2B`
- **LLM:** `Qwen/Qwen3.5-9B`
- **Embedding:** `Qwen/Qwen3-Embedding-0.6B`

*Note: Ensure you have enough GPU memory (VRAM) to load the 9B model, or consider using a machine with at least 24GB VRAM for smooth execution.*

## Running Experiments

The `experiment.py` script allows you to run one of the three experimental scenarios described in the methodology.

### Basic Execution

To run a specific experiment, use the following command:

```bash
python experiment.py --experiment <ID> --tasks tasks.json --kb kb.json --out results.json
```

### Experiment IDs:

- `1`: **Baseline Comparison** (Qwen 3.5 2B vs 9B)
- `2`: **RAG Augmentation** (2B+RAG vs 9B+RAG)
- `3`: **Multi-Agent System** (MAS 2B+RAG vs 9B+RAG)

### Optional Arguments:

- `--temperature` (default: 0.7): Controls randomness of generation.
- `--max-tokens` (default: 512): Maximum length of the output.
- `--top-k` (default: 3): Number of documents to retrieve from RAG.
- `--runs` (default: 1): Number of repetitions per task for consistency analysis.
- `--think`: Enable thinking mode for Qwen 3.5.

### Example: Running the Multi-Agent Experiment

```bash
python experiment.py --experiment 3 --tasks tasks.json --kb kb.json --out results_mas.json --runs 3
```

## Evaluation

The script generates a `results.json` file containing all responses and a `human_eval.csv` file formatted for expert review.
