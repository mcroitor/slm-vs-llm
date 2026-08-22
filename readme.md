# Evaluating the Effectiveness of Multi-Agent Systems Based on Small Language Models Compared to Larger Parameter Models

This repository contains the implementation and research data for the study comparing Large Language Models (LLMs) with Multi-Agent Systems (MAS) based on Small Language Models (SLMs), both augmented with Retrieval-Augmented Generation (RAG).

## Project Structure

- `experiment.py` — Main execution script for all three experiments.
- `prepare_kb.py` — Builds the RAG knowledge base from Markdown files.
- `analyze.py` — Analyzes experiment results (statistics, report, human evaluation).
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
pip install torch transformers accelerate
```

### 2. Model Downloading

The project uses models from the Qwen family. The script will automatically download them from Hugging Face upon the first run, but you can also ensure they are available in your cache.

**Models used:**

- **SLM:** `Qwen/Qwen3.5-2B`
- **LLM:** `Qwen/Qwen3.5-9B`
- **Embedding:** `Qwen/Qwen3-Embedding-0.6B`

*Note: The script automatically selects CUDA if available, otherwise falls back to CPU (with float32). The 9B model in float16 requires roughly 18 GB of VRAM; with quantization it fits in ~9 GB. For CPU-only execution expect significantly slower inference.*

### 3. Knowledge Base Preparation

The RAG system requires a knowledge base in JSON format. You can easily create it from a collection of Markdown files:

1. Create a `data/` folder and place your `.md` documents there.
2. Run the preparation script:

```bash
python prepare_kb.py --data data --out kb.json
```

This will scan the folder and generate a `kb.json` file containing all the text from your documents.

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
- `--think {on,off}` (default: off): Enable or disable Qwen 3.5 thinking mode.
- `--small-model`, `--large-model`: Override the Hugging Face model names.
- `--embed-model`: Override the Hugging Face embedding model name.
- `--device auto|cuda|cpu` (default: auto): Override device selection.
- `--out`: Output JSON path (default: `results.json`).
- `--human-eval`: Output CSV path for expert review (default: `human_eval.csv`).

### Example: Running the Multi-Agent Experiment

```bash
python experiment.py --experiment 3 --tasks tasks.json --kb kb.json --out results_mas.json --runs 3
```

## Results Analysis

Run the analyzer on the experiment outputs to produce a statistical report:

```bash
python analyze.py --results results_1.json results_2.json results_3.json
```

### Optional Arguments:

- `--gold <file>`: JSON of gold (reference) answers mapped to task texts — enables exact-match and F1 accuracy in the report.
- `--human-eval <csv...>`: scored human evaluation files (one per rater) to aggregate Likert scores and compute Krippendorff's alpha. If you provide several files, name them after the raters, e.g. `rater_anna.csv rater_bob.csv`.
- `--rater-names <name...>`: explicit rater names matching the `--human-eval` order.
- `--out-dir <dir>` (default: `analysis`): output directory for `report.md`.

The report includes: per-configuration summaries, paired comparisons with Wilcoxon signed-rank test, Cliff's delta, and wins/ties/losses for the core contrasts (`2b` vs `9b`, `2b+RAG` vs `9b+RAG`, `MAS(2b+RAG)` vs `9b+RAG`), quality-per-cost trade-offs, a synthesizer fidelity check for Experiment 3, and human-evaluation statistics.

## Evaluation

The script generates a `results.json` file containing all responses and a `human_eval.csv` file formatted for expert review.

## Human Evaluation

The `human_eval.csv` file contains one row per configuration run, with the model's response and computed metrics (token usage, latency). The remaining columns are to be filled in by human evaluators on a Likert scale from 1 (very poor) to 5 (excellent).

### Evaluation Dimensions

| Column                  | What to Assess                                    |
| ----------------------- | ------------------------------------------------- |
| `accuracy_1_5`          | Factual correctness and task completion           |
| `coherence_1_5`         | Logical flow and structure of the response        |
| `comprehensiveness_1_5` | Coverage of all task requirements                 |
| `reasoning_1_5`         | Quality of step-by-step logic (where applicable)  |
| `consistency_1_5`       | Agreement of the answer with the expected content |

### Procedure

1. **Split the CSV**: distribute responses among evaluators so that each response is rated by at least two independent evaluators (needed for inter-rater agreement).
2. **Blind rating**: evaluators see only the response text, not the configuration name or the model that produced it. For this, use a copy of the CSV with the `config` column removed.
3. **Fill in scores**: evaluators assign a score from 1 to 5 for each dimension and may add notes.
4. **Merge the results**: collect the completed files and merge them into a single `human_eval_scored.csv` for the analysis script.
5. **Check agreement**: with two or more raters per response, the analysis script computes inter-rater reliability (Krippendorff's alpha) to confirm the ratings are consistent.

### Notes

- If a response is empty or an error occurred, rate it `1` and add a note.
- Keep at least two raters per response so that agreement can be measured; the more raters, the more reliable the scores.
- Scores are stored in the CSV only; the analysis script (`analyze.py`) aggregates them per configuration and task and runs the statistical tests described in `methodology.md`.
