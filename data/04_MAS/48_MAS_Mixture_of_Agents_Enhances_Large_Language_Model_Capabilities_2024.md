# Mixture-of-Agents Enhances Large Language Model Capabilities

**Junlin Wang** — Duke University, Together AI (junlin.wang2@duke.edu)
**Jue Wang** — Together AI (jue@together.ai)
**Ben Athiwaratkun** — Together AI (ben@together.ai)
**Ce Zhang** — University of Chicago, Together AI (cez@uchicago.edu)
**James Zou** — Stanford University, Together AI (jamesz@stanford.edu)

arXiv:2406.04692v1 [cs.CL] 7 Jun 2024 — *Preprint. Under review.*

Code: https://github.com/togethercomputer/moa

---

## Abstract

Recent advances in large language models (LLMs) demonstrate substantial capabilities in natural language understanding and generation tasks. With the growing number of LLMs, how to harness the collective expertise of multiple LLMs is an exciting open direction. Toward this goal, we propose a new approach that leverages the collective strengths of multiple LLMs through a Mixture-of-Agents (MoA) methodology. In our approach, we construct a layered MoA architecture wherein each layer comprises multiple LLM agents. Each agent takes all the outputs from agents in the previous layer as auxiliary information in generating its response. MoA models achieve state-of-the-art performance on AlpacaEval 2.0, MT-Bench, and FLASK, surpassing GPT-4 Omni. For example, our MoA using only open-source LLMs is the leader of AlpacaEval 2.0 by a substantial gap, achieving a score of 65.1% compared to 57.5% by GPT-4 Omni.

## 1. Introduction

Large language models (LLMs) have significantly advanced the field of natural language understanding and generation in recent years. These models are pretrained on vast amounts of data and subsequently aligned with human preferences to generate helpful and coherent outputs. However, despite the plethora of LLMs and their impressive achievements, they still face inherent constraints on model size and training data. Further scaling up these models is exceptionally costly, often requiring extensive retraining on several trillion tokens.

At the same time, different LLMs possess unique strengths and specialize in various task aspects. For instance, some models excel at complex instruction following while others may be better suited for code generation. This diversity in skill sets among different LLMs presents an intriguing question: Can we harness the collective expertise of multiple LLMs to create a more capable and robust model?

Our answer to this question is Yes. We identify an inherent phenomenon we term the **collaborativeness of LLMs** — wherein an LLM tends to generate better responses when presented with outputs from other models, even if these other models are less capable by itself. Figure 1 showcases the LC win rate on the AlpacaEval 2.0 benchmark for 6 popular LLMs.

*Figure 1 (described): A bar chart comparing "Single Model" LC win rates against "With responses from other models" LC win rates for six LLMs — Qwen1.5-72B-Chat, Qwen1.5-110B-Chat, Wizard 8x22b, Mixtral-8x22B-Instruct-v0.1, Llama-3-70B-Instruct, and dbrx-instruct. In every case, the win rate is notably higher when the model is given other models' responses as auxiliary input.*

When these models are provided with answers generated independently by these models, their LC win rates significantly improve. This indicates that the collaborativeness phenomenon is widespread among LLMs. Remarkably, this improvement occurs even when the auxiliary responses provided by the other models are of lower quality than what an individual LLM could generate independently.

Based on this finding, this paper introduces a Mixture-of-Agents (MoA) methodology that leverages multiple LLMs to iteratively enhance the generation quality. The structure of MoA is illustrated in Figure 2. Initially, LLMs in the first layer, denoted as agents A1,1, ...A1,n, independently generate responses to a given prompt. These responses are then presented to agents in the next layer A2,1, ...A2,n (which may reuse a model from the first layer) for further refinement. This iterative refinement process continues for several cycles until obtaining a more robust and comprehensive response.

*Figure 2 (described): A diagram illustrating the Mixture-of-Agents structure across 4 layers. In Layer 1, three agents (A1,1, A1,2, A1,3) each take the prompt and produce token-sequence outputs, which are concatenated. In Layers 2 and 3, three agents again take the concatenated intermediate output and produce refined outputs. Layer 4 consists of a single agent (A4,1) producing the final output. Agents can share the same underlying model.*

To ensure effective collaboration among models and improve overall response quality, careful selection of LLMs for each MoA layer is crucial. This selection process is guided by two primary criteria: (a) **Performance Metrics** — the average win rate of models in layer *i* plays a significant role in determining their suitability for inclusion in layer *i+1*, so selecting models based on demonstrated performance metrics ensures higher-quality outputs; and (b) **Diversity Considerations** — the diversity of model outputs is also crucial, since responses generated by heterogeneous models contribute significantly more than those produced by the same model (as shown later in §3.3). By leveraging these criteria — performance and diversity — MoA aims to mitigate individual model deficiencies and enhance overall response quality through collaborative synthesis.

We conduct comprehensive evaluations using AlpacaEval 2.0, MT-Bench, and FLASK benchmarks for assessing response quality across various dimensions. The results demonstrate substantial improvements with our proposed method, achieving a new SOTA win rate of 65.8% on AlpacaEval 2.0 compared to the previous best of 57.5% achieved by GPT-4 Omni.

The **contributions** of this work are summarized as follows:

1. **Novel framework** — we propose a Mixture-of-Agents framework designed to leverage the strengths of multiple LLMs, thereby improving their reasoning and language generation capabilities.
2. **Finding of collaborativeness of language models** — we highlight the inherent collaborativeness among LLMs, where models tend to generate better quality responses when they have access to outputs from other models, even if those outputs are of lower quality.
3. **State-of-the-art LLM performance** — we conducted extensive experiments using multiple highly-competitive benchmarks such as AlpacaEval 2.0, MT-Bench, and FLASK; our MoA framework achieves state-of-the-art performance on these benchmarks.

## 2. Mixture-of-Agents Methodology

In this section, we present our proposed methodology for leveraging multiple models to achieve boosted performance. We begin by demonstrating that LLMs possess collaborativeness and thus can improve their responses based on the outputs of other models. Following this, we introduce the Mixture-of-Agents methodology and discuss its design implications.

### 2.1. Collaborativeness of LLMs

We begin by demonstrating the collaborativeness of LLMs, specifically their ability to generate higher quality responses when they can reference outputs from other models. As shown in the introduction and Figure 1, many of today's available LLMs exhibit this collaborative capability.

An important pathway to extract maximum benefit from collaboration of multiple LLMs is to characterize what different models are good at in various aspects of collaboration. During the collaboration process, we can categorize LLMs into two distinct roles:

**Proposers** excel at generating useful reference responses for use by other models. While a good proposer may not necessarily produce responses with high scores by itself, it should offer more context and diverse perspectives, ultimately contributing to better final responses when used by an aggregator.

**Aggregators** are models proficient at synthesizing responses from other models into a single, high-quality output. An effective aggregator should maintain or enhance output quality even when integrating inputs that are of lesser quality than its own.

Section 3.3 empirically validates the roles of aggregators and proposers. Specifically, we show that many LLMs possess capabilities both as aggregators and proposers, while certain models displayed specialized proficiencies in distinct roles. GPT-4o, Qwen1.5, and LLaMA-3 emerged as versatile models effective in both assisting and aggregating tasks. In contrast, WizardLM demonstrated excellent performance as a proposer model but struggled to maintain its effectiveness in aggregating responses from other models.

Given that an aggregator can generate higher-quality responses by building upon outputs from other models, we propose further enhancing this collaborative potential by introducing additional aggregators. One intuitive idea is to replicate the exercise with multiple aggregators — initially using several to aggregate better answers and then re-aggregating these aggregated answers. By incorporating more aggregators into the process, we can iteratively synthesize and refine the responses, leveraging the strengths of multiple models to produce superior outcomes. This leads to the design of our proposed Mixture-of-Agents.

### 2.2. Mixture-of-Agents

The structure of MoA is illustrated in Figure 2. It has *l* layers, and each layer-*i* consists of *n* LLMs, denoted by Ai,1, Ai,2, ..., Ai,n. LLMs can be reused either within the same layer or across different layers. When many LLMs in a layer are identical, this configuration leads to a special structure that corresponds to a model generating multiple possibly different outputs (due to the stochasticity of temperature sampling). We refer to this setting as **single-proposer**, where only a sparse subset of models are activated.

Each LLM Ai,j processes an input text and generates its continuation. Our method does not require any fine-tuning and only utilizes the interface of prompting and generation of LLMs. Formally, given an input prompt x₁, the output of the *i*-th MoA layer y*ᵢ* can be expressed as:

**yᵢ = ⊕ₙⱼ₌₁ [Ai,j(xᵢ)] + x₁,  xᵢ₊₁ = yᵢ**   (Eq. 1)

where + means concatenation of texts, and ⊕ means application of the Aggregate-and-Synthesize prompt (Table 1) to these model outputs.

In practice, we do not need to concatenate the prompt and all model responses, so only one LLM is needed in the last layer. Therefore, we use the output of an LLM from the *l*-th layer (A*l*,1(x*l*)) as the final output and evaluate the metrics based on it.

**Table 1: Aggregate-and-Synthesize Prompt to integrate responses from other models.**

> You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
>
> Responses from models:
> 1. [Model Response from Ai,1]
> 2. [Model Response from Ai,2]
> ...
> n. [Model Response from Ai,n]

### 2.3. Analogy to Mixture-of-Experts

Mixture-of-Experts (MoE) (Shazeer et al., 2017) is a prominent and well-established technique in machine learning where multiple expert networks specialize in different skill sets. The MoE approach has shown significant success across various applications due to its ability to leverage diverse model capabilities for complex problem-solving tasks. Our MoA method draws inspiration from this methodology.

A typical MoE design consists of a stack of layers known as MoE layers. Each layer comprises a set of *n* expert networks alongside a gating network and includes residual connections for improved gradient flow. Formally, for layer *i*, this design can be expressed as:

**yᵢ = Σⁿⱼ₌₁ Gi,j(xᵢ)Ei,j(xᵢ) + xᵢ**   (Eq. 2)

where Gi,j represents the output from the gating network corresponding to expert *j*, and Ei,j denotes the function computed by expert network *j*. The leverage of multiple experts allows the model to learn different skill sets and focus on various aspects of the task at hand.

From a high-level perspective, our proposed MoA framework extends the MoE concept to the model level by operating at the model level rather than at the activation level. Specifically, our MoA approach leverages LLMs and operates entirely through the prompt interface rather than requiring modifications to internal activations or weights. Instead of having specialized sub-networks within a single model as in MoE, we utilize multiple full-fledged LLMs across different layers. In our approach, we consolidate the roles of the gating network and expert networks using an LLM, since the intrinsic capacity of LLMs allows them to effectively regularize inputs by interpreting prompts and generating coherent outputs without needing external mechanisms for coordination.

Moreover, since this method relies solely on prompting capabilities inherent within off-the-shelf models: (1) it eliminates computational overhead associated with fine-tuning; (2) it provides flexibility and scalability — our method can be applied to the latest LLMs regardless of their size or architecture.

## 3. Evaluation

This section presents a comprehensive evaluation of our proposed MoA. Our findings show that:

1. We achieve significant improvements on AlpacaEval 2.0, MT-Bench, and FLASK benchmarks. Notably, with open-source models only, our approach outperforms GPT-4o on AlpacaEval 2.0 and FLASK.
2. We conduct extensive experiments to provide better understanding of the internal mechanism of MoA.
3. Through a detailed budget analysis, several implementations of MoA can deliver performance comparable to GPT-4 Turbo while being 2× more cost-effective.

### 3.1. Setup

**Benchmarks.** We mainly evaluate models on AlpacaEval 2.0 (Dubois et al., 2024), a leading benchmark for assessing the alignment of LLMs with human preferences. It contains 805 instructions representative of real use cases. Each model's response is directly compared against that of GPT-4 (gpt-4-1106-preview), with a GPT-4-based evaluator determining the likelihood of preferring the evaluated model's response. To ensure fairness, the evaluation employs length-controlled (LC) win rates, effectively neutralizing length bias (this metric tracks closely with human preferences, achieving a Spearman correlation of 0.98 with actual human evaluations).

Additionally, we also evaluate on MT-Bench (Zheng et al., 2023) and FLASK (Ye et al., 2023). MT-Bench uses GPT-4 to grade and give a score to a model's answer. FLASK, on the other hand, offers a more granular evaluation with 12 skill-specific scores.

**Models.** In our study, we constructed our default MoA using only open-source models to achieve competitive performance. The models included are: Qwen1.5-110B-Chat (Bai et al., 2023), Qwen1.5-72B-Chat, WizardLM-8x22B (Xu et al., 2023a), LLaMA-3-70B-Instruct (Touvron et al., 2023b), Mixtral-8x22B-v0.1 (Jiang et al., 2024), and dbrx-instruct (The Mosaic Research Team, 2024). We construct 3 MoA layers and use the same set of models in each MoA layer, using Qwen1.5-110B-Chat as the aggregator in the last layer. We also developed a variant called **MoA w/ GPT-4o**, which prioritizes high-quality outputs by using GPT-4o as the aggregator in the final MoA layer. Another variant, **MoA-Lite**, emphasizes cost-effectiveness: it uses the same set of models as proposers but includes only 2 MoA layers and employs Qwen1.5-72B-Chat as the aggregator, making it more cost-effective than GPT-4o while achieving a 1.8% improvement in quality on AlpacaEval 2.0. We ensure strict adherence to the licensing terms of all models utilized in this research. For open-source models, all inferences were run through the Together Inference Endpoint.

### 3.2. Benchmark Results

**Table 2: Results on AlpacaEval 2.0 and MT-Bench.** For AlpacaEval 2.0, MoA and MoA-Lite correspond to the 6-proposer setup with 3 layers and 2 layers respectively. MoA w/ GPT-4o corresponds to using GPT-4o as the final aggregator in MoA.

**(a) AlpacaEval 2.0**

| Model | LC win. | win. |
|---|---|---|
| MoA w/ GPT-4o | 65.7±0.7% | 78.7±0.2% |
| MoA | 65.1±0.6% | 59.8±0.3% |
| MoA-Lite | 59.3±0.2% | 57.0±0.7% |
| GPT-4 Omni (05/13) | 57.5% | 51.3% |
| GPT-4 Turbo (04/09) | 55.0% | 46.1% |
| WizardLM 8x22B† | 51.3% | 62.3% |
| GPT-4 Preview (11/06) | 50.0% | 50.0% |
| Qwen1.5 110B Chat | 43.9% | 33.8% |
| Qwen1.5 72B Chat | 36.6% | 26.5% |
| GPT-4 (03/14) | 35.3% | 22.1% |
| Llama 3 70B Instruct | 34.4% | 33.2% |
| Mixtral 8x22B v0.1 | 30.9% | 22.2% |

*(† denotes the authors' replication of the AlpacaEval results.)*

**(b) MT-Bench**

| Model | Avg. | 1st turn | 2nd turn |
|---|---|---|---|
| MoA w/ GPT-4o | 9.40±0.06 | 9.49 | 9.31 |
| GPT-4 Turbo (04/09) | 9.31 | 9.35 | 9.28 |
| MoA | 9.25±0.10 | 9.44 | 9.07 |
| GPT-4 Preview (11/06) | 9.20 | 9.38 | 9.03 |
| GPT-4 Omni (05/13) | 9.19 | 9.31 | 9.07 |
| MoA-Lite | 9.18±0.09 | 9.38 | 8.99 |
| Qwen1.5 110B Chat | 8.96 | 9.23 | 8.63 |
| Llama 3 70B Instruct | 8.94 | 9.2 | 8.68 |
| Mixtral 8x22B v0.1 | 8.78 | 9.11 | 8.44 |
| WizardLM 8x22B | 8.78 | 8.96 | 8.61 |
| Qwen1.5 72B Chat | 8.44 | 8.55 | 8.34 |
| GPT-4 (06/13) | 8.84 | 9.08 | 8.61 |

**AlpacaEval 2.0.** We conducted comparisons against leading models such as GPT-4 and other state-of-the-art open-source models. The detailed results are presented in Table 2a, where our MoA methodology achieved top positions on the AlpacaEval 2.0 leaderboard, demonstrating a remarkable 8.2% absolute improvement over the previous top model, GPT-4o. Moreover, it is particularly noteworthy that our model outperformed GPT-4o using solely open-source models, achieving a margin of 7.6% absolute improvement from 57.5% (GPT-4o) to 65.1% (MoA). Our MoA-Lite setup uses fewer layers and is more cost-effective; even with this lighter approach, we still outperform the best model by 1.8%, improving from 57.5% (GPT-4o) to 59.3% (MoA-Lite). This further highlights the effectiveness of our method in leveraging open-source model capabilities with varying compute budgets to their fullest potential.

**MT-Bench.** Though improvements over individual models on MT-Bench are relatively incremental, this is understandable given that current models already perform exceptionally well on this benchmark, as a single model alone can achieve scores greater than 9 out of 10. Despite the marginal enhancements, our approach still secures the top position on the leaderboard, demonstrating that even with already highly optimized benchmarks, our method can push the boundaries further, maintaining the leadership.

**FLASK.** FLASK provides fine-grained evaluation of models across 12 skill-specific dimensions. Among those metrics, MoA excels in several key aspects — specifically showing significant improvement in robustness, correctness, efficiency, factuality, commonsense, insightfulness, and completeness, compared to the single model score of the aggregator, Qwen-110B-Chat. Additionally, MoA outperforms GPT-4 Omni in terms of correctness, factuality, insightfulness, completeness, and metacognition. One metric where MoA did not do as well was conciseness — the model produced outputs that were marginally more verbose.

*Figure 3 (described): A radar/spider chart showing FLASK scores across 12 dimensions (robustness, correctness, efficiency, factuality, commonsense, comprehension, insightfulness, completeness, metacognition, readability, conciseness, harmlessness) for four systems: GPT-4 Omni (05/13), GPT-3.5-turbo-0125, Qwen1.5-110B-Chat, and MoA (6-proposer setup with Qwen1.5-110B-Chat as aggregator). The MoA curve generally extends further out — indicating higher scores — on most dimensions except conciseness.*

### 3.3. What Makes Mixture-of-Agents Work Well?

In this subsection, we conduct experiments that provide us with a better understanding of the internal mechanism of Mixture-of-Agents. Key insights below.

**Mixture-of-Agents significantly outperforms LLM rankers.** First, we compare Mixture-of-Agents with an LLM-based ranker, which uses the aggregator model to select one of the answers generated by the proposers, instead of generating a new output. The results (Figure 4) show that the MoA approach significantly outperforms an LLM-ranker baseline, suggesting that the aggregator does not simply select one of the generated answers, but potentially performs sophisticated aggregation over all proposed generations.

**MoA tends to incorporate the best proposed answers.** We also compare the aggregator's response with the proposers' responses via similarity scores such as BLEU (Papineni et al., 2002), which reflects n-gram overlaps. Within each sample, given *n* proposed answers, we calculate the Spearman's rank correlation coefficient between the *n* similarity scores and the *n* preference scores determined by the GPT-4-based evaluator. The results (Figure 4b) confirm a positive correlation between win rate and BLEU score. Results with Levenshtein similarity or TF-IDF (Appendix A) also yield positive correlation with preference scores.

*Figure 4 (described): (a) A line chart of LC win rate across MoA Layers 1–4, with separate lines for GPT-4o, Qwen1.5-110B-Chat, Qwen1.5-72B-Chat, Wizard 8x22b, Mixtral-8x22B-Instruct-v0.1, Llama-3-70B-Instruct, dbrx-instruct, and an LLM-Ranker baseline, plus reference lines for GPT-4 Omni and GPT-4 Preview scores. All curves use the same 6 proposer agents and differ only in the final aggregator; the MoA aggregation curves rise well above the LLM-Ranker curve across layers. (b) A horizontal bar chart of Spearman correlation coefficients between BLEU similarity and win rate, grouped by aggregator model (QWen1.5-110B, QWen1.5-72B, WizardLM, Llama-3-70B, Mixtral-8x22B, dbrx-instruct) and by aggregation round (1st, 2nd, 3rd), all showing positive correlation.*

**Table 3: Effects of the number of proposer models on AlpacaEval 2.0.** *n* denotes either the number of agents in an MoA layer or the number of proposed outputs in the single-proposer setting. Qwen1.5-110B-Chat is used as the aggregator, with 2 MoA layers for all settings.

| Setting | Multiple-Proposer | Single-Proposer |
|---|---|---|
| n = 6 | 61.3% | 56.7% |
| n = 3 | 58.0% | 56.1% |
| n = 2 | 58.8% | 54.5% |
| n = 1 | 47.8% | 47.8% |

**Table 4: Impact of different models serving as proposers vs. aggregators.** When evaluating different aggregators, all six models serve as proposers; when evaluating proposers, Qwen1.5-110B-Chat serves as the aggregator. 2 MoA layers are used.

| Model | As aggregator | As proposer |
|---|---|---|
| Qwen1.5-110B-Chat | 61.3% | 56.7% |
| Qwen1.5-72B-Chat | 59.3% | 53.3% |
| LLaMA-3-70b-Instruct | 45.0% | 60.6% |
| WizardLM 8x22B | 52.9% | 63.8% |
| Mixtral-8x22B-Instruct | 48.4% | 54.8% |
| dbrx-instruct | 41.5% | 55.1% |

**Effect of model diversity and the number of proposers.** We analyze how the number of proposals affects final output quality by varying *n*, the number of proposers in each layer. Table 3 shows that scores increase monotonically with *n*, reflecting the benefits of having more auxiliary information. We also quantify the impact of using a diverse set of LLMs as proposers, comparing "single-proposer" (the *n* responses generated by the same LLM at temperature 0.7) against "multiple-proposer" (each response generated by a different LLM). Using multiple different LLMs consistently yielded better results, suggesting that a larger number of diverse LLM agents in each MoA layer can improve performance — further scaling the width of MoA is a promising direction of future investigation.

**Specialization of models in the Mixture-of-Agents ecosystem.** Table 4 shows that GPT-4o, Qwen, and LLaMA-3 emerged as versatile models effective in both proposing and aggregating tasks. In contrast, WizardLM demonstrated excellent performance as a proposer model but struggled to maintain effectiveness when aggregating responses from other models.

### 3.4. Budget and Token Analysis

To understand the relationship between budget, token usage, and LC win rates, we conducted a budget and token analysis (Figure 5a and 5b).

*Figure 5 (described): (a) A scatter plot of LC win rate ("Score") vs. Cost, with points colored/sized by model type (Multi Proposer vs. Single Proposer) and MoA layer (1, 2, 3), plus reference points for GPT-4o and GPT-4-turbo. A dashed Pareto frontier line runs from the lower-left up through MoA-Lite to MoA at the top-right, showing MoA and MoA-Lite deliver higher scores at lower cost than GPT-4o/GPT-4-turbo. (b) An analogous scatter plot of Score vs. tflops (used as a latency proxy), showing a similar Pareto frontier with MoA and MoA-Lite dominating GPT-4o and GPT-4-turbo.*

**Cost Effectiveness.** In Figure 5a, we plot the LC win rate against the average inference cost for each instance in the AlpacaEval 2.0 benchmark (costs calculated from API provider pricing as of May 22, 2024). The chart reveals a Pareto front where certain models strike an optimal balance between cost and performance. If we prioritize quality, MoA is the best configuration. If we want to strike a good balance between quality and cost, MoA-Lite can match GPT-4o's cost while achieving a higher level of quality — notably outperforming GPT-4 Turbo by approximately 4% while being more than twice as cost-effective.

**Tflops Consumption.** Figure 5b depicts the relationship between LC win rate and the number of tflops, used as a proxy for latency since latency can vary depending on the inference system. A Pareto front can be observed here as well, with models on this front effectively utilizing their computational resources to maximize LC win rate. (Note: we calculate the sum over layers of the max number of tflops among proposers in each MoA layer, since multiple proposers can run in parallel. The actual tflops of GPT-4 is unknown, so the rumored size of an 8x220B architecture is used.)

## 4. Related Work

### 4.1. LLM Reasoning

In order to improve the generation quality of LLMs, recent research has made great progress in optimizing LLMs for various downstream tasks through prompt engineering. Chain of Thought (CoT) (Wei et al., 2022; Kojima et al., 2022) prompting techniques represent a linear problem-solving approach where each step builds upon the previous one. Fu et al. (2022) applied CoT to multi-step reasoning tasks. To automate CoT prompting, Auto-CoT (Zhang et al., 2022b) constructs demonstrations by sampling diverse questions and generating reasoning chains. Active-Prompt (Diao et al., 2023) focuses on selecting the most uncertain questions for task-specific annotations. Plan-and-Solve prompting (Wang et al., 2023) decomposes tasks into subtasks. Tree-of-Thought (ToT) (Yao et al., 2023a) expands on the reasoning process by considering multiple paths of reasoning and self-evaluating choices. Graph-of-Thought (Yao et al., 2023b) frames thoughts as graphs. Natural Program prompting (Ling et al., 2023) is proposed for better solving deductive reasoning tasks. Re-reading prompting (Xu et al., 2023b) revisits question information embedded within input prompts.

### 4.2. Model Ensemble

A straightforward solution to leverage the strengths of multiple models is reranking outputs from different models. For instance, Jiang et al. (2023) introduce PAIRRANKER, which performs pairwise comparisons on candidate outputs to select the best one, showing improvements on a self-constructed instruction dataset. To address the substantial computational costs associated with multi-LLM inference, other studies have explored training a router that predicts the best-performing model from a fixed set of LLMs for a given input (Wang et al., 2024a; Shnitzer et al., 2024; Lu et al., 2023). Additionally, FrugalGPT (Chen et al., 2023b) proposed reducing the cost of using LLMs by employing different models in a cascading manner. To better leverage the responses of multiple models, Jiang et al. (2023) also trained GENFUSER, a model trained to generate an improved response capitalizing on the strengths of multiple candidates. Huang et al. (2024) proposed fusing the outputs of different models by averaging their output probability distributions.

Another line of work is multi-agent collaboration. Several studies explore using multiple large language models as agents that collectively discuss and reason through given problems interactively. Du et al. (2023) establishes a mechanism for symmetric discussions among agents. Around the same time, MAD (Liang et al., 2023) introduces an asymmetric mechanism design with different roles, i.e., debater and judge. Other similar works include Chan et al. (2023). Moreover, ReConcile (Chen et al., 2023a) exemplifies an asymmetric discussion involving weighted voting. To understand discussion more deeply, Zhang et al. (2023) aim to explain such collaboration mechanisms from a social psychology view. Wang et al. (2024b) systematically compared multi-agent approaches and found a single agent with a strong prompt including detailed demonstrations can achieve comparable response quality to multi-agent approaches.

## 5. Conclusion

This paper introduces a Mixture-of-Agents approach aimed at leveraging the capabilities of multiple LLMs via successive stages for iterative collaboration. Our method harnesses the collective strengths of agents in the Mixture-of-Agents family and can significantly improve upon the output quality of each individual model. Empirical evaluations conducted on AlpacaEval 2.0, MT-Bench, and FLASK demonstrated substantial improvements in response quality, with our approach achieving an LC win rate up to 65%. These findings validate our hypothesis that integrating diverse perspectives from various models can lead to superior performance compared to relying on a single model alone. In addition, we provide insights into improving the design of MoA; systematic optimization of MoA architecture is an interesting direction for future work.

**Limitations.** Our proposed method requires iterative aggregation of model responses, which means the model cannot decide the first token until the last MoA layer is reached. This potentially results in a high Time to First Token (TTFT), which can negatively impact user experience. To mitigate this issue, we can limit the number of MoA layers, as the first response aggregation has the most significant boost on generation quality. Future work could explore chunk-wise aggregation instead of aggregating entire responses at once, which can reduce TTFT while maintaining response quality.

**Broader Impact.** This study holds the potential to enhance the effectiveness of LLM-driven chat assistants, thereby making AI more accessible. Moreover, since the intermediate outputs are expressed in natural language, the MoA presented improves the interpretability of models. This enhanced interpretability facilitates better alignment with human reasoning.

## References

- Bai, J., Bai, S., Chu, Y., et al. Qwen technical report. *arXiv:2309.16609*, 2023.
- Brown, T., Mann, B., Ryder, N., et al. Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33:1877–1901, 2020.
- Chan, C.-M., Chen, W., Su, Y., et al. Chateval: Towards better LLM-based evaluators through multi-agent debate. *arXiv:2308.07201*, 2023.
- Chen, J. C.-Y., Saha, S., and Bansal, M. Reconcile: Round-table conference improves reasoning via consensus among diverse LLMs. *arXiv:2309.13007*, 2023a.
- Chen, L., Zaharia, M., and Zou, J. Frugalgpt: How to use large language models while reducing cost and improving performance. *arXiv:2305.05176*, 2023b.
- Chowdhery, A., Narang, S., Devlin, J., et al. Palm: Scaling language modeling with pathways. *arXiv:2204.02311*, 2022.
- Diao, S., Wang, P., Lin, Y., and Zhang, T. Active prompting with chain-of-thought for large language models. *arXiv:2302.12246*, 2023.
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. *arXiv:2305.14325*, 2023.
- Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B. Length-controlled AlpacaEval: A simple way to debias automatic evaluators. *arXiv:2404.04475*, 2024.
- Fu, Y., Peng, H., Sabharwal, A., Clark, P., and Khot, T. Complexity-based prompting for multi-step reasoning. *arXiv:2210.00720*, 2022.
- Guo, D., Zhu, Q., Yang, D., et al. Deepseek-coder: When the large language model meets programming — the rise of code intelligence. *arXiv:2401.14196*, 2024.
- Hendrycks, D., Burns, C., Kadavath, S., et al. Measuring mathematical problem solving with the MATH dataset. *arXiv:2103.03874*, 2021.
- Huang, Y., Feng, X., Li, B., et al. Enabling ensemble learning for heterogeneous large language models with deep parallel collaboration. *arXiv:2404.12715*, 2024.
- Jiang, A. Q., Sablayrolles, A., Roux, A., et al. Mixtral of experts. *CoRR*, abs/2401.04088, 2024.
- Jiang, D., Ren, X., and Lin, B. Y. LLM-blender: Ensembling large language models with pairwise ranking and generative fusion. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 14165–14178, 2023.
- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. *Advances in Neural Information Processing Systems*, 35:22199–22213, 2022.
- Liang, T., He, Z., Jiao, W., et al. Encouraging divergent thinking in large language models through multi-agent debate. *arXiv:2305.19118*, 2023.
- Ling, Z., Fang, Y., Li, X., et al. Deductive verification of chain-of-thought reasoning. *arXiv:2306.03872*, 2023.
- Lu, K., Yuan, H., Lin, R., et al. Routing to the expert: Efficient reward-guided ensemble of large language models, 2023.
- OpenAI. GPT-4 technical report, 2023.
- Ouyang, L., Wu, J., Jiang, X., et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022.
- Papineni, K., Roukos, S., Ward, T., and Zhu, W. Bleu: a method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the ACL*, pp. 311–318, 2002.
- RapidFuzz. python-Levenshtein by rapidfuzz, 2023.
- Roziere, B., Gehring, J., Gloeckle, F., et al. Code Llama: Open foundation models for code. *arXiv:2308.12950*, 2023.
- Shazeer, N., Mirhoseini, A., Maziarz, K., et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv:1701.06538*, 2017.
- Shnitzer, T., Ou, A., Silva, M., et al. Large language model routing with benchmark datasets, 2024.
- Team, G., Anil, R., Borgeaud, S., et al. Gemini: a family of highly capable multimodal models. *arXiv:2312.11805*, 2023.
- The Mosaic Research Team. Introducing DBRX: A new state-of-the-art open LLM. 2024.
- Touvron, H., Lavril, T., Izacard, G., et al. LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*, 2023a.
- Touvron, H., Martin, L., Stone, K., et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*, 2023b.
- Wang, H., Polo, F. M., Sun, Y., et al. Fusing models with complementary expertise. *The Twelfth International Conference on Learning Representations*, 2024a.
- Wang, L., Xu, W., Lan, Y., et al. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. *arXiv:2305.04091*, 2023.
- Wang, Q., Wang, Z., Su, Y., Tong, H., and Song, Y. Rethinking the bounds of LLM reasoning: Are multi-agent discussions the key? *arXiv:2402.18272*, 2024b.
- Wang, X., Wei, J., Schuurmans, D., et al. Self-consistency improves chain of thought reasoning in language models. *arXiv:2203.11171*, 2022.
- Wei, J., Wang, X., Schuurmans, D., et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35:24824–24837, 2022.
- Xu, C., Sun, Q., Zheng, K., et al. WizardLM: Empowering large language models to follow complex instructions. *arXiv:2304.12244*, 2023a.
- Xu, X., Tao, C., Shen, T., et al. Re-reading improves reasoning in language models. *arXiv:2309.06275*, 2023b.
- Yao, S., Yu, D., Zhao, J., et al. Tree of thoughts: Deliberate problem solving with large language models. *arXiv:2305.10601*, 2023a.
- Yao, Y., Li, Z., and Zhao, H. Beyond chain-of-thought, effective graph-of-thought reasoning in large language models. *arXiv:2305.16582*, 2023b.
- Ye, S., Kim, D., Kim, S., et al. Flask: Fine-grained language model evaluation based on alignment skill sets. *arXiv:2307.10928*, 2023.
- Zhang, J., Xu, X., and Deng, S. Exploring collaboration mechanisms for LLM agents: A social psychology view. *arXiv:2310.02124*, 2023.
- Zhang, S., Roller, S., Goyal, N., et al. OPT: Open pre-trained transformer language models. *arXiv e-prints*, arXiv–2205, 2022a.
- Zhang, Z., Zhang, A., Li, M., and Smola, A. Automatic chain of thought prompting in large language models. *arXiv:2210.03493*, 2022b.
- Zheng, L., Chiang, W.-L., Sheng, Y., et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *arXiv:2306.05685*, 2023.

---

## Supplementary Material

### Appendix A: Spearman Correlation using Different Similarity Functions

We present results using TF-IDF-based similarity and Levenshtein similarity when calculating the Spearman correlation. Specifically, within each sample of *n* proposed answers, we calculate the Spearman correlation coefficient between the *n* similarity scores and the *n* preference scores determined by the GPT-4-based evaluator. As shown in Figure 6, there is indeed a positive correlation between win rate and both TF-IDF similarity and Levenshtein similarity.

*Figure 6 (described): Two horizontal bar charts, one for (a) Spearman correlation using TF-IDF similarity and one for (b) Spearman correlation using Levenshtein similarity, both broken down by aggregator model (QWen1.5-110B, QWen1.5-72B, WizardLM, Llama-3-70B, Mixtral-8x22B, dbrx-instruct) and aggregation round (1st, 2nd, 3rd) — all bars show positive correlation coefficients.*

### Appendix B: LLM Ranker

This section introduces the setup of the LLM-Ranker used in this paper. The LLM-Ranker is designed to evaluate and rank the best output generated by some LLMs. Table 5 presents the template for prompting the model during these evaluations. We use this LLM-Ranker to pick the best answer and use the AlpacaEval evaluator to evaluate the best ranked answer.

**Table 5: Prompt for ranking with LLMs**

> You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers.
>
> I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.
>
> **Instruction:** `{ "instruction": """{instruction}""" }`
>
> **Model Outputs:** a set of unordered outputs, each tagged with a `model_identifier` and its `output` text (up to six models in this template).
>
> **Task:** Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer with only the model identifier of the best model — no quotes, no spaces, no new lines.

### Appendix C: Case Study

We present a case study in this section. Due to the length of the responses generated by all models, we show only selected fragments for brevity. To illustrate how the aggregator synthesizes the response, similar expressions between the proposed responses and the aggregated response were underlined in different colors in the original figures (not reproduced here). Content mentioned by all proposed responses is omitted.

**Table 6: Case — some models produce high quality answers** *(topic: the song "Smooth" by Rob Thomas)*

| Role | Preference | Content (excerpt) |
|---|---|---|
| Qwen1.5-110B-Chat | 0.35 | Describes it as a blend of rock, pop, and Latin music, highlighting Santana's guitar work and Rob Thomas's vocals. |
| Qwen1.5-72B-Chat | 0.00 | — |
| Llama-3-70B-Instruct | 0.00 | Notes it's a collaboration between Rob Thomas (Matchbox Twenty) and Santana. |
| WizardLM-2-8x22B | 0.99 | Notes its 12 weeks atop the Billboard Hot 100, describing its sultry, energetic vibe. |
| Mixtral-8x22B-Instruct-v0.1 | 0.00 | — |
| dbrx-instruct | 0.00 | — |
| **Aggregated (Qwen1.5-110B-Chat)** | **0.99** | Combines the collaboration background, genre-blend description, chart success (12 weeks at #1), and music video setting into a single cohesive answer. |

Table 6 showcases the responses generated by different proposers. The aggregated response generated by Qwen1.5-110B-Chat reflects a high preference for its own content but also incorporates key points from Llama-3-70B-Instruct and WizardLM 8x22B. Notably, GPT-4's preference score for WizardLM 8x22B's response is 0.99, and the final aggregated answer also achieves a preference score of 0.99.

**Table 7: Case — none of the proposed responses is good enough** *(topic: how to become an author)*

| Role | Preference | Content (excerpt) |
|---|---|---|
| Qwen1.5-110B-Chat | 0.00 | Multi-step list covering niche selection, understanding the publishing industry, self-publishing options. |
| Qwen1.5-72B-Chat | 0.00 | — |
| Llama-3-70B-Instruct | 0.16 | List covering developing a love of reading/writing, marketing, continuous improvement. |
| WizardLM-2-8x22B | 0.03 | Short list including "finish your work." |
| Mixtral-8x22B-Instruct-v0.1 | 0.00 | — |
| dbrx-instruct | 0.00 | — |
| **Aggregated (Qwen1.5-110B-Chat)** | **0.33** | Combined list: cultivating a love for writing/reading, choosing a niche, finishing the work, self-publishing, marketing and promotion, continuous learning. |

Table 7 presents another case where none of the proposed responses achieve a high GPT-4 preference score. Despite this, the aggregator successfully identifies and incorporates the strong points from these responses, achieving a preference score of 0.33.

### Appendix D: MATH Task

Here, we demonstrate that our approach is applicable to reasoning tasks, such as those in the MATH dataset (Hendrycks et al., 2021). The results (Table 8) show that our method consistently enhances accuracy by a significant margin, indicating that our approach is also effective for this type of task. Notably, our method is complementary to existing reasoning techniques such as Chain of Thought (Wei et al., 2022) and Self-consistency (Wang et al., 2022).

**Table 8: Results on the MATH task.** Different aggregators are evaluated, with all six models serving as proposers in each MoA layer.

| Aggregator | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| Qwen1.5-72B-Chat | 0.428 | 0.526 | 0.552 |
| Qwen1.5-110B-Chat | 0.500 | 0.570 | 0.576 |
| Wizard 8x22b | 0.544 | 0.574 | 0.580 |
| Mixtral-8x22B-Instruct-v0.1 | 0.282 | 0.534 | 0.556 |
| Llama-3-70B-Instruct | 0.456 | 0.584 | 0.578 |
| dbrx-instruct | 0.314 | 0.456 | 0.522 |
