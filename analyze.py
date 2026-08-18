"""Analyze experiment results for the LLM vs MAS SLM study.

Levels:
  1. Descriptive summary per configuration (quality, speed, cost).
  2. Paired comparisons (Wilcoxon signed-rank, Cliff's delta, wins/ties/losses)
     between the configurations of each experiment.
  3. Objective accuracy against gold answers (exact match / F1), if provided.
  4. Human evaluation from scored CSVs (Likert means + Krippendorff's alpha).
  5. Quality-vs-cost trade-off (score per 1000 tokens, score per second).
  6. Synthesizer hallucination analysis for experiment 3.

Outputs: console report, analysis/report.md, per-analysis CSVs and JSON.

Statistics are implemented without scipy (Wilcoxon via normal approximation,
Cliff's delta, Krippendorff's alpha for interval data).

Usage:
    python analyze.py --results results_1.json results_2.json results_3.json
    python analyze.py --gold gold_answers.json
    python analyze.py --human-eval human_eval_rater1.csv human_eval_rater2.csv
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- statistics


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test via normal approximation.

    Returns (W+, p-value). Rows with zero difference are dropped.
    """
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(diffs[order[j]]) == abs(diffs[order[i]]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mean = n * (n + 1) / 4.0
    tie_groups = defaultdict(int)
    for d in diffs:
        tie_groups[round(abs(d), 9)] += 1
    tie_correction = sum(t**3 - t for t in tie_groups.values() if t > 1)
    var = (n * (n + 1) * (2 * n + 1) - tie_correction) / 24.0
    if var <= 0:
        return w_plus, 1.0
    z = (w_plus - mean) / math.sqrt(var)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return w_plus, p


def cliff_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta: P(x>y) - P(x<y), in [-1, 1]."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0
    wins = losses = 0
    for a in x:
        for b in y:
            if a > b:
                wins += 1
            elif a < b:
                losses += 1
    return (wins - losses) / (n * m)


def wins_ties_losses(x: Sequence[float], y: Sequence[float]) -> Tuple[int, int, int]:
    wins = sum(1 for a, b in zip(x, y) if a > b)
    losses = sum(1 for a, b in zip(x, y) if a < b)
    ties = sum(1 for a, b in zip(x, y) if a == b)
    return wins, ties, losses


def krippendorff_alpha_interval(scores: Dict[str, Dict[str, float]]) -> float:
    """Krippendorff's alpha for interval data.

    scores: {item_key: {rater: value}}. Missing ratings are allowed.
    Returns alpha in [-1, 1]; 1.0 if no disagreements.
    """
    items = list(scores.keys())
    raters = sorted({r for it in scores.values() for r in it})
    ratings = {i: {r: v for r, v in scores[i].items()} for i in items}
    total = sum(len(v) for v in ratings.values())

    item_sums = {i: sum(v.values()) for i, v in ratings.items()}
    item_counts = {i: len(v) for i, v in ratings.items()}

    # Observed disagreement Do
    do = 0.0
    for i in items:
        vals = list(ratings[i].values())
        cnt = len(vals)
        for a in range(cnt):
            for b in range(a + 1, cnt):
                do += (vals[a] - vals[b]) ** 2

    # Expected disagreement De
    all_vals = [v for it in ratings.values() for v in it.values()]
    mean_all = sum(all_vals) / total if total else 0.0
    de = 0.0
    for i in items:
        n_i = item_counts[i]
        if n_i > 1:
            contrib = 0.0
            for v in ratings[i].values():
                contrib += (v - mean_all) ** 2
            de += contrib * (n_i / (n_i - 1)) if n_i > 1 else contrib

    if de == 0:
        return 1.0
    return 1.0 - (do / de)


# ------------------------------------------------------------------ metrics


def word_f1(reference: str, hypothesis: str) -> float:
    ref = set(re.findall(r"\b\w+\b", reference.lower()))
    hyp = set(re.findall(r"\b\w+\b", hypothesis.lower()))
    if not ref:
        return 1.0
    if not hyp:
        return 0.0
    prec = len(ref & hyp) / len(hyp)
    rec = len(ref & hyp) / len(ref)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _response_text(payload: Dict[str, Any]) -> str:
    """Extract the representative text from a configuration payload.

    Uses the last successful run; for MAS configs reads the 'final' answer.
    """
    runs = payload.get("runs", [])
    for run in reversed(runs):
        text = run.get("response") or run.get("final", "")
        if text:
            return text
    return ""


def _task_metric(payload: Dict[str, Any], key: str) -> Optional[float]:
    if key in payload.get("auto", {}):
        return payload["auto"][key]
    metrics = payload.get("metrics", {})
    if key in metrics:
        return float(metrics[key])
    return None


# ------------------------------------------------------------------- output


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


class Report:
    def __init__(self):
        self.lines: List[str] = []

    def h1(self, text: str) -> None:
        self.lines.append(f"\n# {text}\n")

    def h2(self, text: str) -> None:
        self.lines.append(f"\n## {text}\n")

    def h3(self, text: str) -> None:
        self.lines.append(f"\n### {text}\n")

    def p(self, text: str) -> None:
        self.lines.append(text + "\n")

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        self.lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        self.lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            self.lines.append("| " + " | ".join(fmt(c) for c in row) + " |")
        self.lines.append("")

    def text(self) -> str:
        return "\n".join(self.lines)


# ------------------------------------------------------------- data loading


def load_results(paths: Sequence[str]) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        out.append((path, data))
    return out


def load_gold(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return {str(i): str(item) for i, item in enumerate(data)}
    return {str(k): str(v) for k, v in data.items()}


def load_human_eval(paths: Sequence[str], rater_names: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """Parse scored human_eval CSVs into {item: {rater: value}} per dimension.

    Returns a dict {dimension: {item_key: {rater_name: score}}}.
    """
    dimensions = ["accuracy_1_5", "coherence_1_5", "comprehensiveness_1_5",
                  "reasoning_1_5", "consistency_1_5"]
    by_dim: Dict[str, Dict[str, Dict[str, float]]] = {d: {} for d in dimensions}
    for idx, path in enumerate(paths):
        rater = rater_names[idx] if rater_names and idx < len(rater_names) else os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                item = (row.get("task", ""), row.get("config", ""), row.get("run", ""))
                for d in dimensions:
                    raw = row.get(d, "").strip()
                    if raw in {"", "-"}:
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    by_dim[d].setdefault(item, {})[rater] = value
    return by_dim


# --------------------------------------------------------------- experiment


EXPERIMENT_PAIRS = {
    "1": ("2b", "9b"),
    "2": ("2b+RAG", "9b+RAG"),
    "3": ("MAS(2b+RAG)", "9b+RAG"),
}


def collect_pairs(data: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
    """Return [(task, config_a, config_b) payloads] for the experiment's pair."""
    exp = str(data.get("experiment", ""))
    if exp not in EXPERIMENT_PAIRS:
        return []
    a, b = EXPERIMENT_PAIRS[exp]
    pairs = []
    for task, configs in data.get("tasks", {}).items():
        if a in configs and b in configs:
            pairs.append((task, configs[a], configs[b]))
    return pairs


def summarize_config(configs: Dict[str, Dict[str, Any]], gold: Optional[Dict[str, str]]) -> List[List[Any]]:
    rows = []
    for task, payload in configs.items():
        text = _response_text(payload)
        acc = f1 = None
        if gold is not None and task in gold:
            acc = 1.0 if text.strip().lower() == gold[task].strip().lower() else 0.0
            f1 = word_f1(gold[task], text)
        rows.append([
            task,
            _task_metric(payload, "keyword_coverage"),
            acc,
            f1,
            _task_metric(payload, "word_count"),
            _task_metric(payload, "consistency"),
            _task_metric(payload, "total_tokens"),
            _task_metric(payload, "latency_s"),
        ])
    return rows


def compare_pairs(pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
                  metric: str,
                  gold: Optional[Dict[str, str]]) -> List[List[Any]]:
    rows = []
    for task, pa, pb in pairs:
        va = _task_metric(pa, metric)
        vb = _task_metric(pb, metric)
        rows.append([task, va, vb])
    vals_a = [r[1] for r in rows if r[1] is not None]
    vals_b = [r[2] for r in rows if r[2] is not None]
    if not vals_a or not vals_b:
        return []
    n = min(len(vals_a), len(vals_b))
    w, p = wilcoxon_signed_rank(vals_a[:n], vals_b[:n])
    cd = cliff_delta(vals_a, vals_b)
    wins, ties, losses = wins_ties_losses(vals_a, vals_b)
    return [{
        "n": n,
        "wilcoxon_W": w,
        "wilcoxon_p": p,
        "cliff_delta": cd,
        "wins_a": wins,
        "ties": ties,
        "losses_a": losses,
        "mean_a": sum(vals_a) / len(vals_a),
        "mean_b": sum(vals_b) / len(vals_b),
        "rows": rows,
    }]


def run_analysis(results_files: Sequence[str], gold_path: Optional[str],
                 human_eval_files: Sequence[str], out_dir: str) -> Report:
    report = Report()
    gold = load_gold(gold_path) if gold_path else None
    loaded = load_results(results_files)

    if gold:
        report.h1("Objective accuracy (gold answers)")
        report.p("Exact-match and F1 are computed per task in the summary tables "
                 "below (columns 'exact_match' and 'F1').")

    for path, data in loaded:
        exp = str(data.get("experiment", ""))
        report.h1(f"Experiment {exp} — {path}")
        tasks = data.get("tasks", {})
        report.p(f"Tasks: {len(tasks)} | Configurations: {sorted(tasks[next(iter(tasks))].keys()) if tasks else '-'}")

        for config in sorted({c for t in tasks.values() for c in t}):
            report.h3(f"Summary — {config}")
            config_payloads = {task: payload[config] for task, payload in tasks.items() if config in payload}
            rows = summarize_config(config_payloads, gold)
            if not rows:
                continue
            headers = ["Task", "keyword_cov", "exact_match", "F1", "words",
                       "consistency", "tokens", "latency_s"]
            report.table(headers, rows)
            report.table(["Aggregate", "mean", "mean", "mean", "mean", "mean", "total", "total"],
                         [[
                             "total",
                             sum(r[1] or 0 for r in rows) / len(rows),
                             sum(r[2] or 0 for r in rows) / len(rows),
                             sum(r[3] or 0 for r in rows) / len(rows),
                             sum(r[4] or 0 for r in rows) / len(rows),
                             sum(r[5] or 0 for r in rows) / len(rows),
                             sum(r[6] or 0 for r in rows),
                             sum(r[7] or 0 for r in rows),
                         ]])

        pairs = collect_pairs(data)
        if pairs:
            report.h2("Paired comparison")
            for metric in ["keyword_coverage", "total_tokens", "latency_s"]:
                res = compare_pairs(pairs, metric, gold)
                if not res:
                    continue
                r = res[0]
                report.h3(f"Metric: {metric}")
                report.table(["Task", f"{EXPERIMENT_PAIRS[exp][0]}", f"{EXPERIMENT_PAIRS[exp][1]}"],
                             [[row[0], fmt(row[1]), fmt(row[2])] for row in r["rows"]])
                report.table(
                    ["n", "Wilcoxon W+", "p-value", "Cliff's delta", "wins(A)", "ties", "wins(B)", "mean A", "mean B"],
                    [[r["n"], fmt(r["wilcoxon_W"]), fmt(r["wilcoxon_p"]), fmt(r["cliff_delta"]),
                      r["wins_a"], r["ties"], r["losses_a"], fmt(r["mean_a"]), fmt(r["mean_b"])]],
                )
                significance = "significant" if r["wilcoxon_p"] < 0.05 else "not significant"
                report.p(f"Interpretation: {significance} at alpha=0.05.")

        # Trade-off: quality per 1000 tokens
        report.h2("Quality per cost")
        qrows = []
        for task, payload in tasks.items():
            for config, p in payload.items():
                text = _response_text(p)
                q = p.get("auto", {}).get("keyword_coverage", 0.0)
                tokens = p.get("metrics", {}).get("total_tokens", 0)
                lat = p.get("metrics", {}).get("latency_s", 0.0)
                if tokens > 0:
                    qrows.append([task, config, q, tokens, lat,
                                  q / (tokens / 1000.0) if tokens else 0.0,
                                  q / lat if lat else 0.0])
        if qrows:
            report.table(["Task", "Config", "keyword_cov", "tokens", "latency_s",
                          "score/1000tok", "score/sec"], qrows)

        if exp == "3":
            report.h2("Synthesizer fidelity (hallucination check)")
            for task, payload in tasks.items():
                mas = payload.get("MAS(2b+RAG)")
                if not mas:
                    continue
                for run in mas.get("runs", []):
                    final = run.get("final", "")
                    responses = run.get("responses", {})
                    if not responses:
                        continue
                    avg_f1 = sum(word_f1(final, r) for r in responses.values()) / len(responses)
                    report.p(f"Task '{task[:50]}...': mean F1(final, experts) = {avg_f1:.3f} "
                             f"({len(responses)} expert responses)")
                    break

    # Human evaluation
    if human_eval_files:
        report.h1("Human evaluation")
        raters = [os.path.splitext(os.path.basename(f))[0] for f in human_eval_files]
        by_dim = load_human_eval(human_eval_files, raters)
        for dim, items in by_dim.items():
            if not items:
                continue
            report.h2(dim)
            report.p(f"Raters: {', '.join(sorted({r for it in items.values() for r in it}))}")
            alpha = krippendorff_alpha_interval(items)
            report.p(f"Krippendorff's alpha = {alpha:.3f}")
            agg: Dict[str, List[float]] = defaultdict(list)
            for it, rat in items.items():
                for r, v in rat.items():
                    agg[it[1]].append(v)
            report.table(["Config", "n_ratings", "mean"],
                         [[config, len(v), sum(v) / len(v)] for config, v in sorted(agg.items())])

    # Write outputs
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report.text())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True,
                        help="result JSON files (results_1.json, ...)")
    parser.add_argument("--gold", help="gold answers JSON (task -> reference text)")
    parser.add_argument("--human-eval", nargs="+",
                        help="scored human_eval CSV files (one per rater)")
    parser.add_argument("--rater-names", nargs="+",
                        help="optional rater names matching --human-eval order")
    parser.add_argument("--out-dir", default="analysis",
                        help="output directory for report.md and tables")
    args = parser.parse_args()

    report = run_analysis(args.results, args.gold, args.human_eval or [], args.out_dir)
    print(report.text())


if __name__ == "__main__":
    main()