"""Experiments 1-3: Qwen 3.5:2b vs Qwen 3.5:9b with RAG and multi-agent variants.

Experiment 1: direct model comparison (2b vs 9b)
Experiment 2: RAG augmentation (2b+RAG vs 9b+RAG)
Experiment 3: multi-agent system (input agent -> 3 RAG experts -> output agent)
              vs 9b+RAG

All models are loaded directly via the transformers library (no external
backend/server). Each configuration records: answer text, latency, token usage.

Usage:
    python experiment.py --experiment 3 --tasks tasks.json --kb kb.json --out results.json
"""

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

SMALL_MODEL = "Qwen/Qwen3.5-2B"
LARGE_MODEL = "Qwen/Qwen3.5-9B"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOP_K = 3
DEFAULT_ROLES = ["analyst", "researcher", "reviewer"]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to",
    "with", "is", "are", "was", "were", "be", "been", "it", "this", "that",
    "what", "how", "why", "when", "which", "who", "your", "you", "do", "does",
    "can", "please", "provide", "write", "explain", "give", "list", "about",
    "between", "vs", "versus", "compare", "using", "use", "as", "at", "by",
}


@dataclass
class Generation:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    tokens_per_s: float = 0.0
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseModel:
    def generate(self, prompt: str, temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> Generation:
        raise NotImplementedError


class HFModel(BaseModel):
    """Decoder-only model loaded directly from Hugging Face Hub via transformers."""

    def __init__(self, name: str):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16 # , device_map="auto"
        ).to("cuda")
        self.model.eval()

    def generate(self, prompt: str, temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS, think: bool = False) -> Generation:
        try:
            started = time.perf_counter()
            
            # System prompt to control thinking mode
            if think:
                full_prompt = f"<system>You are a reasoning model. Please think step-by-step inside <think> tags before providing your final answer.</system>\\n{prompt}"
            else:
                full_prompt = f"<system>You are a helpful assistant. Provide a direct answer without using <think> tags or internal reasoning blocks.</system>\\n{prompt}"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            latency = time.perf_counter() - started
            response_text = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip()

            return Generation(
                text=response_text,
                prompt_tokens=input_len,
                completion_tokens=outputs.shape[1] - input_len,
                latency_s=latency,
                tokens_per_s=(outputs.shape[1] - input_len) / latency if latency else 0.0,
            )
        except Exception as exc:
            return Generation(error=f"{type(exc).__name__}: {exc}")


class TransformersEmbedder:
    """Sentence embedder loaded from Hugging Face Hub via transformers.

    Uses EOS-token pooling (last hidden state) followed by L2 normalization,
    as recommended for Qwen3-Embedding models.
    """

    def __init__(self, name: str = EMBED_MODEL, max_length: int = 512):
        self.name = name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()

    def embed(self, text: str) -> List[float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        vector = outputs.last_hidden_state[:, -1, :].squeeze(0)
        vector = torch.nn.functional.normalize(vector, p=2, dim=-1)
        return vector.cpu().tolist()


class ModelWithRag:
    def __init__(self, model: BaseModel, rag: "RagModule"):
        self.model = model
        self.rag = rag

    def generate(self, prompt: str, temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> Generation:
        context = self.rag.retrieve(prompt)
        if context:
            prompt = f"Relevant context:\n{context}\n\nTask:\n{prompt}"
        return self.model.generate(prompt, temperature, max_tokens)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    if not denom:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


class RagModule:
    def __init__(self, documents: Sequence[str],
                 embedder: Callable[[str], List[float]],
                 top_k: int = DEFAULT_TOP_K):
        self.documents = [str(d).strip() for d in documents if str(d).strip()]
        self.top_k = top_k
        self._embedder = embedder
        self._vectors = [embedder(doc) for doc in self.documents]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        if not self.documents:
            return ""
        query_vector = self._embedder(query)
        scores = [cosine(query_vector, v) for v in self._vectors]
        k = top_k or self.top_k
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return "\n\n".join(self.documents[i] for i in best)


class InputAgent:
    def __init__(self, model: BaseModel):
        self.model = model

    def decompose(self, task: str, roles: Sequence[str] = DEFAULT_ROLES) -> Dict[str, str]:
        subtasks = {}
        for role in roles:
            subtasks[role] = (
                f"{task}\n\nAct as a {role}. Examine the task from the "
                f"perspective of a {role} and report your specialized analysis."
            )
        return subtasks


class ExpertAgent:
    def __init__(self, name: str, generator: ModelWithRag):
        self.name = name
        self.generator = generator

    def process(self, subtask: str) -> Generation:
        return self.generator.generate(subtask)


class OutputAgent:
    def __init__(self, model: BaseModel):
        self.model = model

    def synthesize(self, task: str, responses: Dict[str, Generation]) -> Generation:
        blocks = "\n".join(f"[{role}]:\n{resp.text}" for role, resp in responses.items())
        prompt = (
            f"Task:\n{task}\n\nExpert responses:\n{blocks}\n\n"
            "Synthesize a single coherent final answer integrating the expert responses."
        )
        return self.model.generate(prompt)


class MultiAgentSystem:
    def __init__(self, model: BaseModel, rag: RagModule,
                 roles: Sequence[str] = DEFAULT_ROLES):
        self.input_agent = InputAgent(model)
        self.experts = [
            ExpertAgent(f"expert-{role}", ModelWithRag(model, rag)) for role in roles
        ]
        self.output_agent = OutputAgent(model)
        self.roles = list(roles)

    def run(self, task: str) -> Dict[str, Any]:
        subtasks = self.input_agent.decompose(task, self.roles)
        responses = {}
        for role, expert in zip(self.roles, self.experts):
            responses[role] = expert.process(subtasks[role])
        final = self.output_agent.synthesize(task, responses)
        return {
            "subtasks": subtasks,
            "responses": {r: g.text for r, g in responses.items()},
            "final": final,
            "all_generations": list(responses.values()) + [final],
        }


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text.lower()))


def keyword_coverage(task: str, response: str) -> float:
    content = set(re.findall(r"\b[a-z]+\b", task.lower())) - STOPWORDS
    if not content:
        return 1.0
    resp_words = set(re.findall(r"\b[a-z]+\b", response.lower()))
    return len(content & resp_words) / len(content)


def jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"\b\w+\b", a.lower()))
    wb = set(re.findall(r"\b\w+\b", b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def consistency(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return 1.0
    pairs = 0
    total = 0.0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total += jaccard(texts[i], texts[j])
            pairs += 1
    return total / pairs


def aggregate_metrics(generations: Sequence[Generation]) -> Dict[str, Any]:
    return {
        "calls": len(generations),
        "prompt_tokens": sum(g.prompt_tokens for g in generations),
        "completion_tokens": sum(g.completion_tokens for g in generations),
        "total_tokens": sum(g.total_tokens for g in generations),
        "latency_s": round(sum(g.latency_s for g in generations), 3),
    }


def pack_run(run: List[Generation], task: str) -> Dict[str, Any]:
    texts = [g.text for g in run if g.text]
    return {
        "runs": [
            {
                "response": g.text,
                "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens,
                "total_tokens": g.total_tokens,
                "latency_s": round(g.latency_s, 3),
                "tokens_per_s": g.tokens_per_s,
                "error": g.error,
            }
            for g in run
        ],
        "metrics": aggregate_metrics(run),
        "consistency": consistency(texts),
        "auto": {
            "word_count": word_count(texts[-1]) if texts else 0,
            "keyword_coverage": keyword_coverage(task, texts[-1]) if texts else 0.0,
        },
    }


def run_experiment(experiment_id: str, tasks: Sequence[str],
                   knowledge_base: Optional[Sequence[str]] = None,
                   temperature: float = DEFAULT_TEMPERATURE,
                   max_tokens: int = DEFAULT_MAX_TOKENS,
                   top_k: int = DEFAULT_TOP_K,
                   runs: int = 1,
                   embedder: Optional[Callable[[str], List[float]]] = None,
                   small_model_name: str = SMALL_MODEL,
                   large_model_name: str = LARGE_MODEL,
                   think: bool = False) -> Dict[str, Any]:
    print(f"\n>>> Starting Experiment {experiment_id} (Thinking: {'ON' if think else 'OFF'})")
    if experiment_id in ("2", "3"):
        if not knowledge_base:
            raise ValueError(f"knowledge_base required for experiment {experiment_id}")
        if embedder is None:
            raise ValueError("embedder required when knowledge_base provided")

    print(f"Loading models: {small_model_name}, {large_model_name}...")
    small = HFModel(small_model_name)
    large = HFModel(large_//model_name)

    rag = RagModule(knowledge_base, embedder, top_k=top_k) if knowledge_base else None
    small_rag = ModelWithRag(small, rag) if rag else None
    large_rag = ModelWithRag(large, rag) if rag else None
    mas = MultiAgentSystem(small, rag) if rag else None

    results: Dict[str, Any] = {"experiment": experiment_id, "tasks": {}}

    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}/{len(tasks)}: {task[:60]}...")
        configs: Dict[str, Dict[str, Any]] = {}

        if experiment_id == "1":
            print("  Running 2B model...")
            configs["2b"] = pack_run([small.generate(task, think=think) for _ in range(runs)], task)
            print("  Running 9B model...")
            configs["9b"] = pack_run([large.generate(task, think=think) for _ in range(runs)], task)

        elif experiment_id == "2":
            print("  Running 2B+RAG...")
            configs["2b+RAG"] = pack_run(
                [small_rag.generate(task, think=think) for _ in range(runs)], task)
            print("  Running 9B+RAG...")
            configs["9b+RAG"] = pack_run(
                [large_rag.generate(task, think=think) for _ in range(runs)], task)

        elif experiment_id == "3":
            print("  Running MAS(2B+RAG)...")
            mas_runs = []
            for _ in range(runs):
                mas_result = mas.run(task)
                mas_runs.append({
                    "subtasks": mas_result["subtasks"],
                    "responses": mas_result["responses"],
                    "final": mas_result["final"].text,
                    "metrics": aggregate_metrics(mas_result["all_generations"]),
                })
            final_texts = [m["final"] for m in mas_runs]
            mas_metrics = {
                "calls": mas_runs[0]["metrics"]["calls"] if mas_runs else 0,
                "prompt_tokens": sum(m["metrics"]["prompt_tokens"] for m in mas_runs),
                "completion_tokens": sum(m["metrics"]["completion_tokens"] for m in mas_runs),
                "total_tokens": sum(m["metrics"]["total_tokens"] for m in mas_runs),
                "latency_s": round(sum(m["metrics"]["latency_s"] for m in mas_runs), 3),
            }
            configs["MAS(2b+RAG)"] = {
                "runs": mas_runs,
                "metrics": mas_metrics,
                "consistency": consistency(final_texts),
                "auto": {
                    "word_count": word_count(final_texts[-1]) if final_texts else 0,
                    "keyword_coverage": keyword_coverage(task, final_texts[-1]) if final_texts else 0.0,
                },
            }
            print("  Running 9B+RAG...")
            configs["9b+RAG"] = pack_run(
                [large_rag.generate(task) for _ in range(runs)], task)

        results["tasks"][task] = configs
        print(f"  ✓ Task {i} completed.")

    print(f"\n>>> Experiment {experiment_id} finished successfully.")
    return results


DEFAULT_TASKS = [
    "Write a short paragraph about the benefits of multi-agent AI systems.",
    "Analyze the challenges of deploying large language models on local devices.",
    "Compare the reasoning capabilities of small vs large language models.",
]

DEFAULT_KB = [
    "Qwen 3.5 is a language model series by Alibaba Cloud.",
    "Multi-agent systems distribute tasks among specialized agents.",
    "RAG (Retrieval-Augmented Generation) combines LLMs with external knowledge bases.",
    "Small language models require fewer computational resources than large models.",
    "Role specialization in multi-agent systems improves task decomposition outcomes.",
]


def write_human_eval_csv(results: Dict[str, Any], out_path: str) -> None:
    fieldnames = [
        "experiment", "task", "config", "run", "response",
        "prompt_tokens", "completion_tokens", "total_tokens", "latency_s",
        "accuracy_1_5", "coherence_1_5", "comprehensiveness_1_5",
        "reasoning_1_5", "consistency_1_5", "notes",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for task, configs in results["tasks"].items():
            for config, payload in configs.items():
                for idx, run in enumerate(payload["runs"], 1):
                    writer.writerow({
                        "experiment": results["experiment"],
                        "task": task,
                        "config": config,
                        "run": idx,
                        "response": run.get("response", ""),
                        "prompt_tokens": run.get("prompt_tokens", 0),
                        "completion_tokens": run.get("completion_tokens", 0),
                        "total_tokens": run.get("total_tokens", 0),
                        "latency_s": run.get("latency_s", 0.0),
                    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["1", "2", "3"], required=True)
    parser.add_argument("--tasks", help="JSON file with a list of tasks")
    parser.add_argument("--kb", help="JSON file with a list of knowledge base documents")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--runs", type=int, default=1,
                        help="repetitions per configuration (for consistency)")
    parser.add_argument("--think", choices=["on", "off"], default="off",
                        help="enable or disable thinking mode")
    parser.add_argument("--small-model", default=SMALL_MODEL)
    parser.add_argument("--large-model", default=LARGE_MODEL)
    parser.add_argument("--embed-model", default=EMBED_MODEL,
                        help="Hugging Face model name for RAG embeddings")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--human-eval", default="human_eval.csv")
    args = parser.parse_args()

    tasks = json.load(open(args.tasks, encoding="utf-8")) if args.tasks else DEFAULT_TASKS
    kb = json.load(open(args.kb, encoding="utf-8")) if args.kb else DEFAULT_KB

    try:
        embedder = TransformersEmbedder(args.embed_model).embed
    except Exception as exc:
        raise SystemExit(f"Failed to load embedding model {args.embed_model}: {exc}")

    results = run_experiment(
        experiment_id=args.experiment,
        tasks=tasks,
        knowledge_base=kb if args.experiment in ("2", "3") else None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        runs=args.runs,
        embedder=embedder,
        small_model_name=args.small_model,
        large_model_name=args.large_model,
    )

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    write_human_eval_csv(results, args.human_eval)
    print(f"wrote {args.out} and {args.human_eval}")


if __name__ == "__main__":
    main()