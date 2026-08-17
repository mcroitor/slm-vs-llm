# Methodology

## Research Design

This study employs a quantitative experimental design to evaluate the effectiveness of multi-agent systems based on small language models (SLMs) compared to larger parameter models. The research follows a controlled comparison approach where identical tasks are solved using different system configurations under equivalent conditions.

## Experimental Setup

### Models Evaluated

- **Qwen 3.5 2b**: Small language model (2 billion parameters)
- **Qwen 3.5 9b**: Medium language model (9 billion parameters)  
- **Multi-agent system**: Three Qwen 3.5 2b agents operating collaboratively

### RAG Integration

Retrieval-Augmented Generation (RAG) is implemented as an external knowledge base to mitigate the inherent knowledge limitations of smaller models. The RAG pipeline includes:

1. Document ingestion and chunking
2. Vector embedding using sentence-transformers
3. Semantic search for relevant context retrieval
4. Context injection into model prompts

## Experiments

### Experiment 1: Baseline Model Comparison

- **Comparison**: Qwen 3.5:2b vs Qwen 3.5:9b
- **Objective**: Establish performance baseline difference between small and medium models without external knowledge
- **Tasks**: Standard prompt-response tasks without decomposition

### Experiment 2: RAG Augmentation

- **Comparison**: Qwen 3.5:2b + RAG vs Qwen 3.5:9b + RAG
- **Objective**: Determine if RAG can level the playing field between small and large models
- **Tasks**: Same task set as Experiment 1 with retrieved contextual information

### Experiment 3: Multi-Agent System

- **Comparison**: 3-agent Qwen 3.5:2b + RAG vs Qwen 3.5:9b + RAG
- **Objective**: Evaluate whether multi-agent architecture improves reasoning quality through task decomposition, role specialization, and synthesis
- **Agent Roles**:
  - Controller agent: Receives user prompt, distributes tasks
  - Decision-making agents (3 specialized agents): Process decomposed tasks
  - Synthesis agent: Collects and integrates agent responses into unified answer

## Task Suite

Experiments utilize a diverse task suite covering:

- Academic writing and structuring
- Brainstorming and idea generation
- Problem analysis
- Decision support scenarios
- Structured response generation

Each task is administered in identical form across all configurations to ensure comparability.

## Evaluation Metrics

Quality assessment employs both automatic and human evaluation:

- **Answer accuracy**: Factual correctness and task completion
- **Coherence and structure**: Logical flow, organization of response
- **Comprehensiveness**: Coverage of task requirements
- **Reasoning quality**: Step-by-step logic (for relevant task types)
- **Consistency**: Agreement across multiple runs

Human evaluators (domain experts) rate responses on a Likert scale (1-5) across the above dimensions. Automatic metrics include response length, keyword coverage, and syntactic complexity measures.

## Procedure

1. **Model Configuration**: All models run on equivalent hardware with standardized inference parameters (temperature=0.7, max tokens=512)
2. **RAG Pipeline Calibration**: Vector database constructed from domain-specific corpus; retrieval top-k=3
3. **Task Administration**: Each task presented to all configurations in random order
4. **Response Collection**: Generated answers stored for analysis
5. **Evaluation**: Blind rating by independent evaluators unaware of configuration identities
6. **Statistical Analysis**: Non-parametric tests (Wilcoxon signed-rank) to determine significant differences between configurations

## Expected Contributions

This methodology advances understanding of:

- When and under what conditions multi-agent SLM architectures provide measurable benefits
- The effectiveness of RAG as an equalizer between model sizes
- The trade-offs between model scale and architectural complexity in resource-constrained environments
