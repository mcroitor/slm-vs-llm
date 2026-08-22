
llama-ui
New chat
MCP Servers
Settings
Recent conversations

hello

Hello! 👋 How can I help you today?
how to connect llama.cpp server to opencode?
Connecting llama.cpp Server to OpenCode
1. Start the llama.cpp server
bash

llama-server -m your-model.gguf --port 8080

This exposes an OpenAI-compatible API at http://localhost:8080/v1.
2. Configure OpenCode

Create or edit your opencode.json config (project root or ~/.config/opencode/):
json

{
  "provider": {
    "llama-cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        "your-model": {
          "name": "My Local Llama"
        }
      }
    }
  }
}

    Note: The model key under "models" should match the model name that llama-server reports at http://localhost:8080/v1/models. You can check with:
    bash

    curl http://localhost:8080/v1/models

3. Select the model in OpenCode

Start OpenCode and switch to your local model:
text

/model

Then pick llama-cpp/your-model from the list.
Troubleshooting
Issue 	Fix
Connection refused 	Make sure llama-server is running and on the correct port
Model not found 	Verify the model name matches /v1/models output exactly
Auth errors 	Add "apiKey": "sk-local" to options (llama.cpp ignores it, but some SDKs require it)
Timeout on long prompts 	Increase context: llama-server -m model.gguf -c 8192
Quick sanity check

Before opening OpenCode, confirm the endpoint works:
bash

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"hi"}]}'

If that returns a completion, OpenCode will connect fine.

OpenCode moves fast — if the config shape above doesn't match your version, check opencode.ai/docs/providers for the latest schema. Let me know if you hit any snags!
read the article and return me articles's text as a Markdown file.
markdown

# REALM: Retrieval-Augmented Language Model Pre-Training

**arXiv:2002.08909v1 [cs.CL] 10 Feb 2020**

**Authors:** Kelvin Guu\*¹, Kenton Lee\*¹, Zora Tung¹, Panupong Pasupat¹, Ming-Wei Chang¹

\* Equal contribution
¹ Google Research.
Correspondence to: Kelvin Guu <kguu@google.com>, Kenton Lee <kentontl@google.com>, Zora Tung <gatoatigrado@google.com>, Panupong Pasupat <ppasupat@google.com>, Ming-Wei Chang <mingweichang@google.com>.

---

## Abstract

Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network, requiring ever-larger networks to cover more facts. To capture knowledge in a more modular and interpretable way, we augment language model pre-training with a *latent knowledge retriever*, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents. We demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as interpretability and modularity.

---

## 1. Introduction

Recent advances in language model pre-training have shown that models such as BERT (Devlin et al., 2018), RoBERTa (Liu et al., 2019) and T5 (Raffel et al., 2019) store a surprising amount of world knowledge, acquired from the massive text corpora they are trained on (Petroni et al., 2019). For example, BERT is able to correctly predict the missing word in the following sentence: *"The ___ is the ___ currency of the United Kingdom"* (answer: "pound").

In these language models, the learned world knowledge is stored *implicitly* in the parameters of the underlying neural network. This makes it difficult to determine what knowledge is stored in the network and where. Furthermore, storage space is limited by the size of the network—to capture more world knowledge, one must train ever-larger networks, which can be prohibitively slow or expensive.

To capture knowledge in a more interpretable and modular way, we propose a novel framework, **Retrieval-Augmented Language Model (REALM) pre-training**, which augments language model pre-training algorithms with a learned *textual knowledge retriever*. In contrast to models that store knowledge in their parameters, this approach *explicitly* exposes the role of world knowledge by asking the model to decide what knowledge to retrieve and use during inference. Before making each prediction, the language model uses the retriever to retrieve documents from a large corpus such as Wikipedia, and then attends over those documents to help inform its prediction. Learning this model end-to-end requires backpropagating through a retrieval step that considers an entire corpus of textual knowledge, as shown in Figure 1.

> **Figure 1.** REALM augments language model pre-training with a *neural knowledge retriever* that retrieves knowledge from a *textual knowledge corpus*, *Z* (e.g., all of Wikipedia). Signal from the language modeling objective backpropagates all the way through the retriever, which must consider millions of documents in *Z*—a significant computational challenge that we address.

The key intuition of REALM is to train the retriever using a *performance-based* signal from unsupervised text: a retrieval that *improves* the language model's perplexity *is helpful and should be rewarded*, while an uninformative retrieval should be penalized. For example, in Figure 1, if the model needs to fill the blank in *"the ___ at the top of the pyramid"*, the retriever should be rewarded for selecting a document containing *"The pyramidion on top allows for less material higher up the pyramid"*. We achieve this behavior by modeling our *retrieve-then-predict* approach as a latent variable language model and optimizing the marginal likelihood.

Incorporating a large-scale neural retrieval module during pre-training constitutes a significant computational challenge, since the retriever must consider millions of candidate documents for each pre-training step, and we must backpropagate through its decisions. To address this, we structure the retriever such that the computation performed for each document can be cached and asynchronously updated, and selection of the best documents can be formulated as Maximum Inner Product Search (MIPS).

Numerous prior works have demonstrated the benefit of adding a discrete retrieval step to neural networks (Miller et al., 2016; Chen et al., 2017), but did not apply the framework to language model pre-training and employed non-learned retrievers to handle large-scale document collections. In the language modeling literature, the k-Nearest Neighbor Language Model (Khandelwal et al., 2019) (kNN-LM) retrieves similar LM examples to improve memorization. However, kNN-LM was not fine-tuned for downstream tasks, perhaps because it is unclear how to adapt the retrieval mechanism: a kNN can only use examples labeled for the target task—during fine-tuning, this precludes LM examples, which contain the desired world knowledge. In contrast, REALM's retriever is designed to transfer to other tasks, and the retrieval is just text, not a labeled example.

We evaluate our approach by fine-tuning the models pre-trained with REALM on the task of **Open-domain Question Answering (Open-QA)**, one of the most knowledge-intensive tasks in natural language processing. We evaluate on three popular Open-QA benchmarks (**NaturalQuestions-Open**, **WebQuestions**, and **CuratedTrec**) and compare to state-of-the-art Open-QA models, including both extremely large models that store knowledge implicitly (such as T5) as well as previous approaches that also use a knowledge retriever to access external knowledge, but implement retrieval in a more heuristic fashion (Lee et al., 2019; Min et al., 2019a; Asai et al., 2019). REALM achieves new state-of-the-art results on all three benchmarks, significantly outperforming all previous systems by 4-16% absolute accuracy. We also demonstrate qualitative benefits of REALM, including interpretability and modularity.

---

## 2. Background

### Language model pre-training

The goal of language model pre-training is to learn useful representations of language, usually from unlabeled text corpora. The resulting pre-trained model can then be further trained (*fine-tuned*) for a downstream task of primary interest (in our case, Open-QA), often leading to better generalization than training from scratch (Dai & Le, 2015; Radford et al., 2019).

We focus on the *masked language model*² (MLM) variant of pre-training popularized by BERT (Devlin et al., 2018). In its basic form, an MLM is trained to predict the missing tokens in an input text passage. Given an unlabeled pre-training corpus *X* (e.g., Wikipedia text), a training example (*x, y*) can be generated by randomly masking tokens in a sampled piece of text (e.g., *x* = "The [MASK] is the ___ currency [MASK] the UK"; *y* = ("pound", "of")). The model uses its representation of the masked input *x* to predict the token that should go in each mask. A good MLM must learn to encode syntactic and semantic information (e.g., to predict "of") as well as some world knowledge (e.g., to predict "pound").

### Open-domain question answering (Open-QA)

To measure a model's ability to incorporate world knowledge, we need a downstream task where world knowledge is critical. Perhaps one of the most knowledge-intensive tasks in natural language processing is open-domain question answering (Open-QA): given a question *x* such as *"What is the currency of the UK?"*, a model must output the correct answer string *y*, "pound". The "open" part of Open-QA refers to the fact that the model does *not* receive a pre-identified document that is known to contain the answer, unlike traditional reading comprehension (RC) tasks such as SQuAD (Rajpurkar et al., 2016; 2018). While RC models comprehend a single document, Open-QA models must retain knowledge from millions of documents, since a question could be about any of them.

We focus on Open-QA systems that utilize a *textual knowledge corpus* *Z* as the knowledge source. Many of these systems employ a *retrieval-based* approach: given a question *x*, retrieve potentially relevant documents *z* from the corpus *Z*, and then extract an answer *y* from the documents (Brill et al., 2002; Chen et al., 2017; Lee et al., 2019). Our approach, REALM, is inspired by this paradigm and extends it to language model pre-training.

Alternatively, some recent work has proposed *generation-based* systems that apply a sequence-to-sequence model on *x* to directly generate *y* token-by-token (Lewis et al., 2019; Raffel et al., 2019). We will compare against state-of-the-art systems from both paradigms in our experiments.

---

## 3. Approach

We start by formalizing REALM's pre-training and fine-tuning tasks as a *retrieve-then-predict* generative process in Section 3.1. Then in Section 3.2, we describe the model architectures for each component of that process. In Section 3.3, we show how to implement REALM pre-training and fine-tuning by maximizing the likelihood of REALM's generative process. En route, we address important computational challenges, explain why training works, and also discuss strategies for injecting useful inductive biases. The overall framework is illustrated in Figure 2.

> **Figure 2.** The overall framework of REALM. *Left:* Unsupervised pre-training. The knowledge retriever and knowledge-augmented encoder are jointly pre-trained on the unsupervised language modeling task. *Right:* Supervised fine-tuning. After the parameters of the retriever (θ) and encoder (φ) have been pre-trained, they are then fine-tuned on a task of primary interest, using supervised examples.

### 3.1. REALM's generative process

For both pre-training and fine-tuning, REALM takes some input *x* and learns a distribution *p(y | x)* over possible outputs *y*. For pre-training, the task is masked language modeling: *x* is a sentence from a pre-training corpus *X* with some tokens masked out, and the model must predict the value of those missing tokens, *y*. For fine-tuning, the task is Open-QA: *x* is a question, and *y* is the answer.

REALM decomposes *p(y | x)* into two steps: *retrieve*, then *predict*. Given an input *x*, we first retrieve possibly helpful documents *z* from a knowledge corpus *Z*. We model this as a sample from the distribution *p(z | x)*. Then, we condition on both the retrieved *z* and the original input *x* to generate the output *y*—modeled as *p(y | z, x)*. To obtain the overall likelihood of generating *y*, we treat *z* as a latent variable and marginalize over all possible documents *z*, yielding:

$$p(y | x) = \sum_{z \in Z} p(y | z, x) \; p(z | x) \tag{1}$$

### 3.2. Model architecture

We now describe the two key components: the *neural knowledge retriever*, which models *p(z | x)*, and the *knowledge-augmented encoder*, which models *p(y | z, x)*.

#### Knowledge Retriever

The retriever is defined using a dense inner product model:

$$p(z | x) = \frac{\exp f(x, z)}{\sum_{z'} \exp f(x, z')}, \quad f(x, z) = \text{Embed}_{\text{input}}(x)^\top \text{Embed}_{\text{doc}}(z)$$

where $\text{Embed}_{\text{input}}$ and $\text{Embed}_{\text{doc}}$ are embedding functions that map *x* and *z* respectively to *d*-dimensional vectors. The *relevance score* $f(x, z)$ between *x* and *z* is defined as the inner product of the vector embeddings. The retrieval distribution is the softmax over all relevance scores.

We implement the embedding functions using BERT-style Transformers (Devlin et al., 2018). Following standard practices, we join spans of text by applying wordpiece tokenization, separating them with `[SEP]` tokens, prefixing a `[CLS]` token, and appending a final `[SEP]` token:

$$\text{join}_{\text{BERT}}(x) = \texttt{[CLS]}\; x \;\texttt{[SEP]}$$

$$\text{join}_{\text{BERT}}(x_1, x_2) = \texttt{[CLS]}\; x_1 \;\texttt{[SEP]}\; x_2 \;\texttt{[SEP]}$$

As in Devlin et al. (2018), we pass this into a Transformer, which produces one vector for each token, including the vector corresponding to `[CLS]` which is used as a "pooled" representation of the sequence (denoted $\text{BERT}_{\text{CLS}}$). Finally, we perform a linear projection to reduce the dimensionality of the vector, denoted as a projection matrix *W*:

$$\text{Embed}_{\text{input}}(x) = W_{\text{input}} \; \text{BERT}_{\text{CLS}}(\text{join}_{\text{BERT}}(x))$$

$$\text{Embed}_{\text{doc}}(z) = W_{\text{doc}} \; \text{BERT}_{\text{CLS}}(\text{join}_{\text{BERT}}(z_{\text{title}}, z_{\text{body}}))$$

where $z_{\text{title}}$ is the document's title and $z_{\text{body}}$ is its body. We let $\theta$ denote all parameters associated with the retriever, which include the Transformer and projection matrices.

#### Knowledge-Augmented Encoder

Given an input *x* and a retrieved document *z*, the knowledge-augmented encoder defines *p(y | z, x)*. We join *x* and *z* into a single sequence that we feed into a Transformer (distinct from the one used in the retriever). This allows us to perform rich cross-attention between *x* and *z* before predicting *y*.

At this stage, the architectures for pre-training and fine-tuning differ slightly. For the masked language model pre-training task, we must predict the original value of each `[MASK]` token in *x*. To do so, we use the same masked language modeling (MLM) loss as in Devlin et al. (2018):

$$p(y | z, x) = \prod_{j=1}^{J_x} p(y_j | z, x)$$

$$p(y_j | z, x) \propto \exp\left(w_j^\top \; \text{BERT}_{\text{MASK}}^{(j)}(\text{join}_{\text{BERT}}(x, z_{\text{body}}))\right)$$

where $\text{BERT}_{\text{MASK}}^{(j)}$ denotes the Transformer output vector corresponding to the *j*-th masked token, $J_x$ is the total number of `[MASK]` tokens in *x*, and $w_j$ is a learned word embedding for token $y_j$.

For Open-QA fine-tuning, we wish to produce the answer string *y*. Following previous reading comprehension work (Rajpurkar et al., 2016; Seo et al., 2016; Lee et al., 2016; Clark & Gardner, 2017), we will assume that the answer *y* can be found as a contiguous sequence of tokens in some document *z*. Let $S(z, y)$ be the set of spans matching *y* in *z*. Then we can define *p(y | z, x)* as:

$$p(y | z, x) \propto \sum_{s \in S(z,y)} \exp\left(\text{MLP}\left([h_{\text{START}}(s); \; h_{\text{END}}(s)]\right)\right)$$

$$h_{\text{START}}(s) = \text{BERT}_{\text{START}}^{(s)}(\text{join}_{\text{BERT}}(x, z_{\text{body}}))$$

$$h_{\text{END}}(s) = \text{BERT}_{\text{END}}^{(s)}(\text{join}_{\text{BERT}}(x, z_{\text{body}}))$$

where $\text{BERT}_{\text{START}}^{(s)}$ and $\text{BERT}_{\text{END}}^{(s)}$ denote the Transformer output vectors corresponding to the start and end tokens of span *s*, respectively, while MLP denotes a feed-forward neural network. We will let $\phi$ denote all parameters associated with the knowledge-augmented encoder.

### 3.3. Training

For both pre-training and fine-tuning, we train by maximizing the log-likelihood $\log p(y | x)$ of the correct output *y*. Since both the knowledge retriever and knowledge-augmented encoder are differentiable neural networks, we can compute the gradient of $\log p(y | x)$ (defined in Equation 1) with respect to the model parameters $\theta$ and $\phi$, and optimize using stochastic gradient descent.

The key computational challenge is that the marginal probability $p(y | x) = \sum_{z \in Z} p(y | x, z) \; p(z | x)$ involves a summation over all documents *z* in the knowledge corpus *Z*. We approximate this by instead summing over the top *k* documents with highest probability under $p(z | x)$—this is reasonable if most documents have near zero probability. Even with this approximation, we still need an efficient way to find the top *k* documents. Note that the ordering of documents under $p(z | x)$ is the same as under the relevance score $f(x, z) = \text{Embed}_{\text{input}}(x)^\top \text{Embed}_{\text{doc}}(z)$, which is an inner product. Thus, we can employ Maximum Inner Product Search (MIPS) algorithms to find the approximate top *k* documents, using running time and storage space that scale sub-linearly with the number of documents (Ram & Gray, 2012; Shrivastava & Li, 2014; Shen et al., 2015).

To employ MIPS, we must pre-compute $\text{Embed}_{\text{doc}}(z)$ for every $z \in Z$ and construct an efficient search index over these embeddings. However, this data structure will no longer be consistent with $p(z | x)$ if the parameters $\theta$ of $\text{Embed}_{\text{doc}}$ are later updated. Hence, the search index goes "stale" after every gradient update on $\theta$.

Our solution is to "refresh" the index by asynchronously re-embedding and re-indexing all documents every several hundred training steps. The MIPS index is slightly stale between refreshes, but note that it is *only* used to select the top *k* documents. We recompute $p(z | x)$ and its gradient, using the fresh $\theta$, for these top *k* documents after retrieving them. In Section 4.5, we empirically demonstrate that this procedure results in stable optimization, provided that refreshes happen at a sufficiently frequent rate.

#### Implementing asynchronous MIPS refreshes

We asynchronously refresh the MIPS index by running two jobs in parallel: a primary *trainer* job, which performs gradient updates on the parameters, and a secondary *index builder* job, which embeds and indexes the documents. As shown below, the trainer sends the index builder a snapshot of its parameters, $\theta'$. The trainer then continues to train while the index builder uses $\theta'$ to construct a new index in the background. As soon as the index builder is done, it sends the new index back to the trainer, and the process repeats.

> **Figure 3.** REALM pre-training with asynchronous MIPS refreshes.

While asynchronous refreshes can be used for both pre-training and fine-tuning, in our experiments we only use it for pre-training. For fine-tuning, we just build the MIPS index once (using the pre-trained $\theta$) for simplicity and do not update $\text{Embed}_{\text{doc}}$.³ Note that we still fine-tune $\text{Embed}_{\text{input}}$, so the retrieval function is still updated from the query side.

#### What does the retriever learn?

Since the knowledge retrieval of REALM is latent, it is not obvious how the training objective encourages meaningful retrievals. Here, we show how it rewards retrievals that improve prediction accuracy.

For a given query *x* and document *z*, recall that $f(x, z)$ is the "relevance score" that the knowledge retriever assigns to document *z*. We can see how a single step of gradient descent during REALM pre-training alters this score by analyzing the gradient with respect to the parameters of the knowledge retriever, $\theta$:

$$\nabla_\theta \log p(y | x) = \sum_{z \in Z} r(z) \; \nabla_\theta f(x, z)$$

$$r(z) = \left[\frac{p(y | z, x)}{p(y | x)} - 1\right] p(z | x)$$

For each document *z*, the gradient encourages the retriever to change the score $f(x, z)$ by $r(z)$—increasing if $r(z)$ is positive, and decreasing if negative. The multiplier $r(z)$ is positive if and only if $p(y | z, x) > p(y | x)$. The term $p(y | z, x)$ is the probability of predicting the correct output *y* when using document *z*. The term $p(y | x)$ is the expected value of $p(y | x, z)$ when randomly sampling a document from $p(z | x)$. Hence, document *z* receives a positive update whenever it performs better than expected.

### 3.4. Injecting inductive biases into pre-training

In the process of developing REALM, we discovered several additional strategies that further guide the model towards meaningful retrievals, described below.

**Salient span masking.** During REALM pre-training, we want to focus on examples *x* that require world knowledge to predict the masked tokens. As explained in Section 2, some MLM spans only require local context. To focus on problems that require world knowledge, we mask *salient spans* such as "United Kingdom" or "July 1969". We use a BERT-based tagger trained on CoNLL-2003 data (Sang & De Meulder, 2003) to identify named entities, and a regular expression to identify dates. We select and mask one of these salient spans within a sentence for the masked language modeling task. We show that this significantly outperforms other masking strategies in Section 4.5.

**Null document.** Even with salient span masking, not all masked tokens require world knowledge to predict. We model this by adding an empty *null document* $\emptyset$ to the top *k* retrieved documents, allowing appropriate credit to be assigned to a consistent sink when no retrieval is necessary.

**Prohibiting trivial retrievals.** If the pre-training corpus *X* and the knowledge corpus *Z* are the same, there exists a trivial retrieval candidate *z* that is *too* informative: if the masked sentence *x* comes from document *z*, the knowledge augmented encoder can trivially predict *y* by looking at the unmasked version of *x* in *z*. This results in a large positive gradient for $p(z | x)$. If this occurs too often, the knowledge retriever ends up learning to look for exact string matches between *x* and *z*, which does not capture other forms of relevance. For this reason, we exclude this trivial candidate during pre-training.

**Initialization.** At the beginning of training, if the retriever does not have good embeddings for $\text{Embed}_{\text{input}}(x)$ and $\text{Embed}_{\text{doc}}(z)$, the retrieved documents *z* will likely be unrelated to *x*. This causes the knowledge augmented encoder to learn to ignore the retrieved documents. Once this occurs, the knowledge retriever does not receive a meaningful gradient and cannot improve, creating a vicious cycle. To avoid this cold-start problem, we warm-start $\text{Embed}_{\text{input}}$ and $\text{Embed}_{\text{doc}}$ using a simple training objective known as the Inverse Cloze Task (ICT) where, given a sentence, the model is trained to retrieve the document where that sentence came from. We defer to Lee et al. (2019) for details. For the knowledge-augmented encoder, we warm-start it with BERT pre-training—specifically, the uncased BERT-base model (12 layers, 768 hidden units, 12 attention heads).

---

## 4. Experiments

We now evaluate our approach on the Open-QA task. In this section, we describe in detail the benchmarks used and the different approaches to which we compare empirically.

### 4.1. Open-QA Benchmarks

A number of benchmarks have been proposed for Open-QA. In this work, we focus on datasets where the question writers did not already know the answer. This yields questions that reflect more realistic information-seeking needs, and also avoids artifacts that can arise if the question is formulated with a particular answer in mind. A deeper justification is given in Lee et al. (2019). In all cases, the predicted answer is evaluated via exact match with any reference answer, following previous Open-QA work (Chen et al., 2017).

**NaturalQuestions-Open.** The NaturalQuestions dataset (Kwiatkowski et al., 2019) consists of naturally occurring Google queries and their answers. Each answer also comes with an "answer type": following Lee et al. (2019), we only keep questions that are categorized as "short answer type" with at most five tokens. The dataset also provides a suggested Wikipedia document to retrieve; like all models we compare against, we do not provide this to our model.

**WebQuestions.** The WebQuestions dataset (Berant et al., 2013) was collected from the Google Suggest API, using one seed question and expanding the set to related questions. We follow the setting defined by Chen et al. (2017).

**CuratedTrec.** The CuratedTrec dataset is a collection of question-answer pairs drawn from real user queries issued on sites such as MSNSearch and AskJeeves. To account for multiple correct answers or different spelling variations, the answers in this dataset are defined as regular expressions that match all correct answers. It is unclear how to train generation-based models with this type of supervision, so we do not evaluate them on this dataset.

### 4.2. Approaches compared

**Retrieval-based Open-QA.** Most existing Open-QA systems answer the input question by first retrieving potentially relevant documents from a knowledge corpus, and then using a reading comprehension system to extract an answer from the documents. In this paradigm, the knowledge is stored *explicitly* in the corpus. We wish to compare different methods for implementing retrieval.

Many approaches use non-learned heuristic retrieval such as sparse bag-of-words matching (Robertson et al., 2009) or entity linking on the question to select a small set of relevant documents (e.g., 20). These documents are typically then re-ranked using a learned model, but coverage may be limited by the initial heuristic retrieval step. Approaches such as DrQA (Chen et al., 2017), HardEM (Min et al., 2019a), GraphRetriever (Min et al., 2019b), and PathRetriever (Asai et al., 2019) in Table 1 are in this category.

Some recent approaches have proposed to implement learnable retrieval using a MIPS index. ORQA (Lee et al., 2019) formulates Open-QA using a similar latent variable model as REALM, and also trains by maximizing the marginal likelihood. However, REALM adds a novel language model pre-training step, and backpropagates into the MIPS index, rather than using a fixed index. In Table 1, we directly compare the two. It is also important to note that the retrievers for both REALM pre-training and ORQA are initialized using the Inverse Cloze Task, described in Section 3.4.

**Generation-based Open-QA.** An emerging alternative approach to Open-QA is to model it as a sequence prediction task: simply encode the question, and then decode the answer token-by-token based on the encoding. While it was initially unclear how large amounts of knowledge could be injected into the model, GPT-2 (Radford et al., 2019) hinted at the possibility of directly generating answers without using any given context via sequence-to-sequence. However, their performance was not competitive possibly due to the lack of fine-tuning. Orthogonally, T5 (Raffel et al., 2019) showed that directly generating answers without explicit extraction from the given context is a viable approach, but they only experimented on the reading comprehension task, where a context document is provided.

For the most competitive and comparable generation-based baseline, we compare to concurrent work which fine-tunes T5 for Open-QA (Roberts et al., 2020).⁴ We compare against the Base, Large, and even larger 11-billion parameter model to measure the effect of model size.

### 4.3. Implementation Details

**Fine-tuning.** We reuse all hyperparameters from Lee et al. (2019), to enable direct comparison. Our knowledge corpus is derived from the December 20, 2018 snapshot of English Wikipedia. Documents are greedily split into chunks of up to 288 BERT wordpieces, resulting in just over 13 million retrieval candidates. During fine-tuning inference, we consider the top-5 candidates, and the entire model can be run on a single machine with a 12GB GPU.

**Pre-training.** We pre-train for 200k steps on 64 Google Cloud TPUs, with a batch size of 512 and a learning rate of 3e-5, using BERT's default optimizer. The document embedding step for the MIPS index is parallelized over 16 TPUs. For each example, we retrieve and marginalize over 8 candidate documents, including the null document $\emptyset$. We experiment with two choices of the pre-training corpus *X*: (1) Wikipedia, which is identical to the knowledge corpus *Z*, and (2) CC-News, our reproduction of the corpus of English news proposed by Liu et al. (2019).

### 4.4. Main results

Table 1 shows the accuracy of different approaches on the three Open-QA datasets. REALM outperforms all previous approaches by a significant margin. Table 1 also shows the number of parameters for each model.

As reported in the concurrent work of Roberts et al. (2020), the generative Open-QA systems based on T5 are surprisingly powerful, with the largest T5-11B model outperforming the previous best Open-QA system. Increasing the size of T5 yields consistent improvement, but comes at significant computational cost (from Base to 11B, the model is 50 times larger, and gains roughly 5 points in accuracy). In contrast, REALM outperforms the largest T5-11B model while being 30 times smaller. It is also important to note that T5 accesses additional reading comprehension data from SQuAD during its pre-training (100,000+ examples). Access to such data could also benefit REALM, but was not used in our experiments.

Among all systems, the most direct comparison with REALM is ORQA (Lee et al., 2019), where the fine-tuning setup, hyperparameters and training data are identical. The improvement of REALM over ORQA is purely due to better pre-training methods. The results also indicate that our method of pre-training can be applied both on (1) the single-corpus setting (*X* = Wikipedia, *Z* = Wikipedia), or (2) the separate-corpus setting (*X* = CC-News, *Z* = Wikipedia).

Compared to other retrieval-based systems (Asai et al., 2019; Min et al., 2019a; b) which often retrieve from 20 to 80 documents, our system gets the overall best performance while only retrieving 5 documents.

#### Table 1. Test results on Open-QA benchmarks.

The number of train/test examples are shown in parentheses below each benchmark. Predictions are evaluated with exact match against any reference answer. Sparse retrieval denotes methods that use sparse features such as TF-IDF and BM25. Our model, REALM, outperforms all existing systems.

| Name | Architectures | Pre-training | NQ (79k/4k) | WQ (3k/2k) | CT (1k/1k) | # params |
|------|--------------|-------------|:-----------:|:----------:|:----------:|:--------:|
| BERT-Baseline (Lee et al., 2019) | Sparse Retr. + Transformer | BERT | 26.5 | 17.7 | 21.3 | 110m |
| T5 (base) (Roberts et al., 2020) | Transformer Seq2Seq | T5 (Multitask) | 27.0 | 29.1 | - | 223m |
| T5 (large) (Roberts et al., 2020) | Transformer Seq2Seq | T5 (Multitask) | 29.8 | 32.2 | - | 738m |
| T5 (11b) (Roberts et al., 2020) | Transformer Seq2Seq | T5 (Multitask) | 34.5 | 37.4 | - | 11318m |
| DrQA (Chen et al., 2017) | Sparse Retr. + DocReader | N/A | - | 20.7 | 25.7 | 34m |
| HardEM (Min et al., 2019a) | Sparse Retr. + Transformer | BERT | 28.1 | - | - | 110m |
| GraphRetriever (Min et al., 2019b) | GraphRetriever + Transformer | BERT | 31.8 | 31.6 | - | 110m |
| PathRetriever (Asai et al., 2019) | PathRetriever + Transformer | MLM | 32.6 | - | - | 110m |
| ORQA (Lee et al., 2019) | Dense Retr. + Transformer | ICT + BERT | 33.3 | 36.4 | 30.1 | 330m |
| **Ours (X = Wikipedia, Z = Wikipedia)** | **Dense Retr. + Transformer** | **REALM** | **39.2** | **40.2** | **46.8** | **330m** |
| **Ours (X = CC-News, Z = Wikipedia)** | **Dense Retr. + Transformer** | **REALM** | **40.4** | **40.7** | **42.9** | **330m** |

### 4.5. Analysis

In Table 2 we present results for NaturalQuestions-Open after ablating critical components of REALM. In addition to the end-to-end results, we also report how often the gold answer appears in the top-5 retrievals before applying any fine-tuning. The latter metric more significantly isolates the contribution of improving the retriever during pre-training.

#### Table 2. Ablation experiments on NQ's development set.

| Ablation | Exact Match | Zero-shot Retrieval Recall@5 |
|----------|:-----------:|:----------------------------:|
| REALM | 38.2 | 38.5 |
| REALM retriever + Baseline encoder | 37.4 | 38.5 |
| Baseline retriever + REALM encoder | 35.3 | 13.9 |
| Baseline (ORQA) | 31.3 | 13.9 |
| REALM with random uniform masks | 32.3 | 24.2 |
| REALM with random span masks | 35.3 | 26.1 |
| 30× stale MIPS | 28.7 | 15.1 |

**Encoder or Retriever.** We first aim to determine whether REALM pre-training improves the retriever or the encoder, or both. To do so, we can reset the parameters of either the retriever or the encoder to their baseline state before REALM pre-training, and feed that into fine-tuning. Resetting both the retriever and encoder reduces the system to our main baseline, ORQA. We find that both the encoder and retriever benefit from REALM training separately, but the best result requires both components acting in unison.

**Masking scheme.** We compare our salient span masking scheme (Section 3.4) with (1) random token masking introduced in BERT (Devlin et al., 2018) and (2) random span masking proposed by SpanBERT (Joshi et al., 2019). While such salient span masking has not been shown to be impactful in previous work with standard BERT training (Joshi et al., 2019), it is crucial for REALM. Intuitively, the latent variable learning relies heavily on the utility of retrieval and is therefore more sensitive to a consistent learning signal.

**MIPS index refresh rate.** During pre-training, we run a parallel process to re-embed corpus documents and rebuild the MIPS index. This results in one index refresh per approximately 500 training steps. To demonstrate the importance of frequent index refreshes, we compare against using a slower refresh rate. The results in Table 2 suggest that a stale index can hurt model training, and further reducing this staleness could offer better optimization.

**Examples of retrieved documents.** Table 3 shows an example of the REALM masked language model prediction. In this example, "Fermat" is the correct word, and REALM (row (c)) gives the word a much higher probability compared to the BERT model (row (a)). Since REALM manages to retrieve some documents with a related fact (row (b)), the marginalized probability of the correct answer dramatically increases. This shows that REALM is able to retrieve documents to fill in the masked word even though it is trained with unsupervised text only.

#### Table 3. An example where REALM utilizes retrieved documents to better predict masked tokens.

*x*: "An equilateral triangle is easily constructed using a straightedge and compass, because 3 is a ___ prime."

| Model | Result |
|-------|--------|
| (a) BERT | $p(y = \text{"Fermat"} \mid x) = 1.1 \times 10^{-14}$ (No retrieval.) |
| (b) REALM | $p(y = \text{"Fermat"} \mid x, z) = 1.0$ (Conditional probability with document *z* = "257 is . . . a Fermat prime. Thus a regular polygon with 257 sides is constructible with compass . . .") |
| (c) REALM | $p(y = \text{"Fermat"} \mid x) = 0.129$ (Marginal probability, marginalizing over top 8 retrieved documents.) |

---

## 5. Discussion and Related Work

We previously discussed related methods for Open-QA. Here we present several alternate ways of viewing REALM that connect it to a broader set of ideas beyond Open-QA:

**Language modeling with corpus as context.** Language representation models have been incorporating contexts of increasingly large scope when making predictions. Examples of this progression include models that condition on surrounding words (Mikolov et al., 2013a; b), sentences (Kiros et al., 2015; Peters et al., 2018), and paragraphs (Radford et al., 2018; Devlin et al., 2018). We can view REALM as a generalization of the above work to the next level of scope: the entire text *corpus*.

**Retrieve-and-edit with learned retrieval.** In order to better explain the variance in the input text and enable controllable generation, Guu et al. (2018) proposed a language model with the retrieve-and-edit framework (Hashimoto et al., 2018) that conditions on text with high lexical overlap. REALM has a similar approach, except that the model learns for itself which texts are most useful for reducing perplexity. By jointly learning the retriever, REALM has the capacity to depend on information beyond lexical overlap.

**Scalable grounded neural memory.** The document index can be viewed as a memory where the keys are the document embeddings. From this view, our work shares motivations with works such as product key memory (Lample et al., 2019), which enables sub-linear memory access in a memory network (Weston et al., 2014; Graves et al., 2014; Sukhbaatar et al., 2015), allowing these scalable memory layers to be integrated into large language models. One main difference is that our memories are *grounded*—each memory is associated with a document rather than unnamed value vectors. This level of interpretability is crucial for applications like Open-QA, where users would require provenance for a predicted answer to be trustworthy.

**Unsupervised Corpus Alignment.** In sequence-to-sequence models with attention (Bahdanau et al., 2014), text is generated with latent selection of relevant tokens. This results in a set of *model-centric* unsupervised alignments between target and source tokens. Analogously, REALM also generates text with latent selection of relevant documents. A by-product of our method is that we offer a set of *model-centric* unsupervised alignments between text in the pre-training corpus *X* and knowledge corpus *Z*.

---

## 6. Future Work

The work presented here is the minimal instantiation of a family of REALM-like approaches where a representation is pre-trained to perform reasoning over a large corpus of knowledge on-the-fly during inference. We are particularly optimistic about generalizations of this work to (1) *structured knowledge*, which would result in a generalization of Peters et al. (2019) where we would also learn the decision of which entities are informative, (2) the *multi-lingual setting*, e.g., retrieving knowledge in a high-resource language to better represent text in a low-resource language, and (3) the *multi-modal setting*, e.g., retrieving images or videos that can provide knowledge rarely observed in text.

---

## References

- Asai, A., Hashimoto, K., Hajishirzi, H., Socher, R., and Xiong, C. Learning to retrieve reasoning paths over wikipedia graph for question answering. *arXiv preprint arXiv:1911.10470*, 2019.
- Bahdanau, D., Cho, K., and Bengio, Y. Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*, 2014.
- Berant, J., Chou, A., Frostig, R., and Liang, P. Semantic parsing on freebase from question-answer pairs. In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pp. 1533–1544, 2013.
- Brill, E., Dumais, S., and Banko, M. An analysis of the askmsr question-answering system. In *Empirical Methods in Natural Language Processing*, 2002.
- Chen, D., Fisch, A., Weston, J., and Bordes, A. Reading wikipedia to answer open-domain questions. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, volume 1, pp. 1870–1879, 2017.
- Clark, C. and Gardner, M. Simple and effective multi-paragraph reading comprehension. In *Annual Meeting of the Association for Computational Linguistics*, 2017.
- Dai, A. M. and Le, Q. V. Semi-supervised sequence learning. In *Advances in neural information processing systems*, pp. 3079–3087, 2015.
- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.
- Graves, A., Wayne, G., and Danihelka, I. Neural turing machines. *ArXiv*, abs/1410.5401, 2014.
- Guu, K., Hashimoto, T. B., Oren, Y., and Liang, P. Generating sentences by editing prototypes. *Transactions of the Association for Computational Linguistics*, 6:437–450, 2018.
- Hashimoto, T. B., Guu, K., Oren, Y., and Liang, P. S. A retrieve-and-edit framework for predicting structured outputs. In *Advances in Neural Information Processing Systems*, pp. 10052–10062, 2018.
- Joshi, M., Chen, D., Liu, Y., Weld, D. S., Zettlemoyer, L., and Levy, O. SpanBERT: Improving pre-training by representing and predicting spans. *arXiv preprint arXiv:1907.10529*, 2019.
- Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and Lewis, M. Generalization through memorization: Nearest neighbor language models. *ArXiv*, abs/1911.00172, 2019.
- Kiros, R., Zhu, Y., Salakhutdinov, R. R., Zemel, R., Urtasun, R., Torralba, A., and Fidler, S. Skip-thought vectors. In *Advances in neural information processing systems*, pp. 3294–3302, 2015.
- Kwiatkowski, T., Palomaki, J., Rhinehart, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Kelcey, M., Devlin, J., et al. Natural questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 2019.
- Lample, G., Sablayrolles, A., Ranzato, M., Denoyer, L., and Jégou, H. Large memory layers with product keys. In *Advances in Neural Information Processing Systems*, pp. 8546–8557, 2019.
- Lee, K., Salant, S., Kwiatkowski, T., Parikh, A., Das, D., and Berant, J. Learning recurrent span representations for extractive question answering. *arXiv preprint arXiv:1611.01436*, 2016.
- Lee, K., Chang, M.-W., and Toutanova, K. Latent retrieval for weakly supervised open domain question answering. In *Proceedings of the Conference of Association for Computational Linguistics*, 2019.
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., and Zettlemoyer, L. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *ArXiv*, abs/1910.13461, 2019.
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.
- Mikolov, T., Chen, K., Corrado, G., and Dean, J. Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*, 2013a.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. Distributed representations of words and phrases and their compositionality. In *Advances in neural information processing systems*, pp. 3111–3119, 2013b.
- Miller, A., Fisch, A., Dodge, J., Karimi, A.-H., Bordes, A., and Weston, J. Key-value memory networks for directly reading documents. *arXiv preprint arXiv:1606.03126*, 2016.
- Min, S., Chen, D., Hajishirzi, H., and Zettlemoyer, L. A discrete hard em approach for weakly supervised question answering. *arXiv preprint arXiv:1909.04849*, 2019a.
- Min, S., Chen, D., Zettlemoyer, L., and Hajishirzi, H. Knowledge guided text retrieval and reading for open domain question answering. *arXiv preprint arXiv:1911.03868*, 2019b.
- Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., and Zettlemoyer, L. Deep contextualized word representations. In *Proc. of NAACL*, 2018.
- Peters, M. E., Neumann, M., IV, R. L. L., Schwartz, R., Joshi, V., Singh, S., and Smith, N. A. Knowledge enhanced contextual word representations, 2019.
- Petroni, F., Rocktäschel, T., Lewis, P., Bakhtin, A., Wu, Y., Miller, A. H., and Riedel, S. Language models as knowledge bases? *arXiv preprint arXiv:1909.01066*, 2019.
- Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding with unsupervised learning. Technical report, OpenAI, 2018.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multi-task learners. *OpenAI Blog*, 2019.
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *arXiv preprint arXiv:1910.10683*, 2019.
- Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pp. 2383–2392, 2016.
- Rajpurkar, P., Jia, R., and Liang, P. Know what you don't know: Unanswerable questions for SQuAD. *arXiv preprint arXiv:1806.03822*, 2018.
- Ram, P. and Gray, A. G. Maximum inner-product search using cone trees. In *Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining*, pp. 931–939, 2012.
- Roberts, A., Raffel, C., and Shazeer, N. How much knowledge can you pack into the parameters of a language model? *arXiv preprint*, 2020.
- Robertson, S., Zaragoza, H., et al. The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4):333–389, 2009.
- Sang, E. T. K. and De Meulder, F. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In *Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003*, pp. 142–147, 2003.
- Seo, M., Kembhavi, A., Farhadi, A., and Hajishirzi, H. Bidirectional attention flow for machine comprehension. In *International Conference on Learning Representations*, 2016.
- Shen, F., Liu, W., Zhang, S., Yang, Y., and Tao Shen, H. Learning binary codes for maximum inner product search. In *Proceedings of the IEEE International Conference on Computer Vision*, pp. 4148–4156, 2015.
- Shrivastava, A. and Li, P. Asymmetric lsh (alsh) for sublinear time maximum inner product search (mips). In *Advances in Neural Information Processing Systems*, pp. 2321–2329, 2014.
- Sukhbaatar, S., Weston, J., Fergus, R., et al. End-to-end memory networks. In *Advances in neural information processing systems*, 2015.
- Weston, J., Chopra, S., and Bordes, A. Memory networks. *arXiv preprint arXiv:1410.3916*, 2014.

---

## Appendix A. Derivation of the gradient with respect to the knowledge retriever

We compute the gradient of the REALM pre-training objective (a log-likelihood) with respect to the parameters of the knowledge retriever, $\theta$:

$$\nabla_\theta \log p(y | x) = \frac{1}{p(y | x)} \nabla_\theta p(y | x)$$

$$= \frac{1}{p(y | x)} \sum_z p(y | z, x) \; \nabla_\theta p(z | x)$$

$$= \frac{1}{p(y | x)} \sum_z p(y | z, x) \; p(z | x) \; \nabla_\theta \log p(z | x)$$

$$= \sum_z p(z | y, x) \; \nabla_\theta \log p(z | x)$$

where the last line follows from applying conditional Bayes' rule. We can then expand $\nabla_\theta \log p(z | x)$ as:

$$\nabla_\theta \log p(z | x) = \nabla_\theta \log \frac{\exp f(x, z)}{\sum_{z'} \exp f(x, z')}$$

$$= \nabla_\theta \left[ f(x, z) - \log \sum_{z'} \exp f(x, z') \right]$$

$$= \nabla_\theta f(x, z) - \sum_{z'} p(z' | x) \; \nabla_\theta f(x, z')$$

Plugging this back into the first set of equations yields:

$$\nabla_\theta \log p(y | x) = \sum_z p(z | y, x) \left[ \nabla_\theta f(x, z) - \sum_{z'} p(z' | x) \; \nabla_\theta f(x, z') \right]$$

$$= \sum_z p(z | y, x) \; \nabla_\theta f(x, z) - \sum_{z'} p(z' | x) \; \nabla_\theta f(x, z')$$

$$= \sum_z \left[ p(z | y, x) - p(z | x) \right] \nabla_\theta f(x, z)$$

$$= \sum_z \left[ \frac{p(y | z, x) \; p(z | x)}{p(y | x)} - p(z | x) \right] \nabla_\theta f(x, z)$$

$$= \sum_z \left[ \frac{p(y | z, x)}{p(y | x)} - 1 \right] p(z | x) \; \nabla_\theta f(x, z)$$

In the second line, we used the fact that the overall expression is an expectation with respect to $p(z | y, x)$, and the terms which depend on $z'$ but not $z$ can be moved out of that expectation.

---

## Appendix B. Connection between REALM and supervised learning

From the equations in Appendix A, we saw that:

$$\nabla_\theta \log p(y | x) = \sum_z \left[ p(z | y, x) - p(z | x) \right] \nabla_\theta f(x, z)$$

Suppose that there exists one document $z^*$ which causes the model to achieve perfect prediction accuracy (i.e., $p(y | z^*, x) = 1$), while all other documents $z'$ result in zero accuracy (i.e., $p(y | z', x) = 0$). Under this setting, $p(z^* | y, x) = 1$ (provided that $p(z^* | x)$ is non-zero), which causes the gradient to become:

$$\nabla_\theta \log p(y | x) = \nabla_\theta f(x, z^*) - \sum_z p(z | x) \; \nabla_\theta f(x, z) = \nabla_\theta \log p(z^* | x)$$

From this, we see that gradient descent on the REALM objective is equivalent to gradient descent on $\log p(z^* | x)$. This is none other than the typical maximum likelihood training objective used in supervised learning, where $z^*$ is the "gold" document.

---

## Appendix C. Adapting to new knowledge

An explicit retrieval system allows us to adapt to new world knowledge simply by modifying the corpus documents. To demonstrate this ability, we replace the knowledge corpus with a more recent version of Wikipedia corpus after pre-training is done. When the input query is about a fact where the two corpora disagree, REALM can change the prediction to reflect the updated information, as exemplified in Table 4. However, even with an explicit retrieval mechanism, the knowledge-augmented encoder will still end up remembering some world knowledge, making the prediction of some input sentences not updated with the new corpus. (For instance, the model predicts "Thatcher" for *"___ is the prime minister of United Kingdom."* on both corpora, perhaps due to the frequent mention of her name in Wikipedia articles.)

#### Table 4. An example where REALM adapts to the updated knowledge corpus.

*x*: "Jennifer ___ formed the production company Excellent Cadaver."

| Model | Top Predictions |
|-------|----------------|
| BERT | also (0.13), then (0.08), later (0.05), . . . |
| REALM (*Z* = 20 Dec 2018 corpus) | smith (0.01), brown (0.01), jones (0.01) |
| REALM (*Z* = 20 Jan 2020 corpus) | **lawrence (0.13)**, brown (0.01), smith (0.01), . . . |

The Wikipedia page "Excellent Cadaver" was added in 2019, so the model was not able to recover the word when the knowledge corpus is outdated (2018). Interestingly, the same REALM model pre-trained on the 2018 corpus is able to retrieve the document in the updated corpus (2020) and generate the correct token, "Lawrence".

---

## Appendix D. Retrieval Utility

The null document $\emptyset$ described in Section 3.4 provides a way to measure the importance of a retrieved document *z*: we define the *retrieval utility* (RU) of *z* for the masked input *x* as the difference between the log-likelihood of the knowledge-augmented encoder when conditioning on *z* versus on $\emptyset$:

$$RU(z | x) = \log p(y | z, x) - \log p(y | \emptyset, x) \tag{2}$$

A negative RU shows that *z* is less useful for predicting *y* than the null document. This could mean that *z* is irrelevant to *x*, but could also mean that the masked tokens in *x* do not require world knowledge to predict, or that the world knowledge is sufficiently commonplace it has been baked into the model's parameters. In practice, we find that RU increases steadily over the course of pre-training, and is more predictive of good performance on the downstream task of Open-QA than even the overall log-likelihood. An example of how RU behaves over time and across different settings is in Figure 4.

> **Figure 4.** The Retrieval Utility (RU, described in Eq. 2) vs the number of pre-training steps. RU roughly estimates the "usefulness" of retrieval. RU is impacted by the choice of masking and the number of pre-training steps. (X-axis: Pre-training Steps in Thousands, 0–200; Y-axis: Retrieval Utility, 0–3. Curves for Salient span masking, Random span masking, and Random uniform masking.)

---

## Footnotes

1. We use the term "document" loosely to refer to a passage from the knowledge corpus, not necessarily a whole article.
2. Strictly speaking, MLM is not a standard language model, since it does not define a distribution over the entire sequence of tokens. In the paper we sometimes abuse the term "language model" slightly to make the phrase shorter.
3. This works because pre-training already yields a good $\text{Embed}_{\text{doc}}$ function. However, it is possible that refreshing the index would further improve performance.
4. We initially conducted our own T5 experiments using the code from https://tinyurl.com/t5-openqa-colab (Raffel et al., 2019). We now report results from the concurrent work of Roberts et al. (2020), which has an improved fine-tuning procedure.

сравни содержимое документов и исправь 21_RAG_REALM_Retrieval_Augmented_Language_Model_Pre_Training_2020.md если найдешь несоответствия

I compared both documents carefully. The Markdown is largely accurate, but I found the following discrepancies:
Discrepancies found
# 	Location 	PDF (original) 	MD (current) 	Type
1 	§3.2, Knowledge-Augmented Encoder 	"…before predicting y. See Figure 1 for a concrete example." 	"…before predicting y." 	Missing sentence
2 	Table 3 caption 	"…predict masked tokens. It assigns much higher probability (0.129) to the correct term, "Fermat", compared to BERT. (Note that the blank corresponds to 3 BERT wordpieces.)" 	"…predict masked tokens." 	Missing caption text

The remaining differences are minor grammar corrections the MD made over typos in the original PDF (e.g., "outperform" → "outperforms", "a much high probability" → "a much higher probability", "our work share" → "our work shares"). These are acceptable editorial fixes, not content errors.
Corrected sections

Apply these two fixes to the MD file:

Fix 1 — §3.2, end of Knowledge-Augmented Encoder paragraph:
markdown

Given an input *x* and a retrieved document *z*, the knowledge-augmented encoder defines *p(y | z, x)*. We join *x* and *z* into a single sequence that we feed into a Transformer (distinct from the one used in the retriever). This allows us to perform rich cross-attention between *x* and *z* before predicting *y*. See Figure 1 for a concrete example.

Fix 2 — Table 3 caption:
markdown

#### Table 3. An example where REALM utilizes retrieved documents to better predict masked tokens. It assigns much higher probability (0.129) to the correct term, "Fermat", compared to BERT. (Note that the blank corresponds to 3 BERT wordpieces.)

No other content mismatches (numbers, formulas, table values, references, appendix derivations) were found. The rest of the MD faithfully reproduces the PDF.
read the article and return me articles's text as a Markdown file.
markdown

# Dense Passage Retrieval for Open-Domain Question Answering

**Vladimir Karpukhin**\*¹, **Barlas Oğuz**\*¹, **Sewon Min**†¹, **Patrick Lewis**¹, **Ledell Wu**¹, **Sergey Edunov**¹, **Danqi Chen**‡¹, **Wen-tau Yih**¹

¹ Facebook AI
† University of Washington
‡ Princeton University

\* Equal contribution

{vladk, barlaso, plewis, ledell, edunov, scottyih}@fb.com
sewon@cs.washington.edu
danqic@cs.princeton.edu

arXiv:2004.04906v3 [cs.CL] 30 Sep 2020

---

## Abstract

Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the *de facto* method. In this work, we show that retrieval can be practically implemented using *dense* representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system greatly by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks.

---

## 1. Introduction

Open-domain question answering (QA) (Voorhees, 1999) is a task that answers factoid questions using a large collection of documents. While early QA systems are often complicated and consist of multiple components (Ferrucci, 2012; Moldovan et al., 2003, *inter alia*), the advances of reading comprehension models suggest a much simplified two-stage framework: (1) a context *retriever* first selects a small subset of passages where some of them contain the answer to the question, and then (2) a machine *reader* can thoroughly examine the retrieved contexts and identify the correct answer (Chen et al., 2017). Although reducing open-domain QA to machine reading is a very reasonable strategy, a huge performance degradation is often observed in practice,¹ indicating the needs of improving retrieval.

Retrieval in open-domain QA is usually implemented using TF-IDF or BM25 (Robertson and Zaragoza, 2009), which matches keywords efficiently with an inverted index and can be seen as representing the question and context in high-dimensional, sparse vectors (with weighting). Conversely, the *dense*, latent semantic encoding is *complementary* to sparse representations by design. For example, synonyms or paraphrases that consist of completely different tokens may still be mapped to vectors close to each other. Consider the question *"Who is the bad guy in Lord of the Rings?"*, which can be answered from the context *"Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy."* A term-based system would have difficulty retrieving such a context, while a dense retrieval system would be able to better match *"bad guy"* with *"villain"* and fetch the correct context. Dense encodings are also *learnable* by adjusting the embedding functions, which provides additional flexibility to have a task-specific representation. With special in-memory data structures and indexing schemes, retrieval can be done efficiently using maximum inner product search (MIPS) algorithms (e.g., Shrivastava and Li, 2014; Guo et al., 2016).

However, it is generally believed that learning a good dense vector representation needs a large number of labeled pairs of question and contexts. Dense retrieval methods have thus never been shown to outperform TF-IDF/BM25 for open-domain QA before ORQA (Lee et al., 2019), which proposes a sophisticated inverse cloze task (ICT) objective, predicting the blocks that contain the masked sentence, for additional pre-training. The question encoder and the reader model are then fine-tuned using pairs of questions and answers jointly. Although ORQA successfully demonstrates that dense retrieval can outperform BM25, setting new state-of-the-art results on multiple open-domain QA datasets, it also suffers from two weaknesses. First, ICT pre-training is computationally intensive and it is not completely clear that regular sentences are good surrogates of questions in the objective function. Second, because the context encoder is not fine-tuned using pairs of questions and answers, the corresponding representations could be suboptimal.

In this paper, we address the question: can we train a better dense embedding model using only pairs of questions and passages (or answers), *without* additional pre-training? By leveraging the now standard BERT pretrained model (Devlin et al., 2019) and a dual-encoder architecture (Bromley et al., 1994), we focus on developing the right training scheme using a relatively small number of question and passage pairs. Through a series of careful ablation studies, our final solution is surprisingly simple: the embedding is optimized for maximizing inner products of the question and relevant passage vectors, with an objective comparing all pairs of questions and passages in a batch.

Our **Dense Passage Retriever** (DPR) is exceptionally strong. It not only outperforms BM25 by a large margin (65.2% vs. 42.9% in Top-5 accuracy), but also results in a substantial improvement on the end-to-end QA accuracy compared to ORQA (41.5% vs. 33.3%) in the open Natural Questions setting (Lee et al., 2019; Kwiatkowski et al., 2019).

Our contributions are twofold. First, we demonstrate that with the proper training setup, simply fine-tuning the question and passage encoders on existing question-passage pairs is sufficient to greatly outperform BM25. Our empirical results also suggest that additional pre-training may not be needed. Second, we verify that, in the context of open-domain question answering, a higher retrieval precision indeed translates to a higher end-to-end QA accuracy. By applying a modern reader model to the top retrieved passages, we achieve comparable or better results on multiple QA datasets in the open-retrieval setting, compared to several, much complicated systems.

---

## 2. Background

The problem of open-domain QA studied in this paper can be described as follows. Given a factoid question, such as *"Who first voiced Meg on Family Guy?"* or *"Where was the 8th Dalai Lama born?"*, a system is required to answer it using a large corpus of diversified topics. More specifically, we assume the extractive QA setting, in which the answer is restricted to a span appearing in one or more passages in the corpus. Assume that our collection contains $D$ documents, $d_1, d_2, \ldots, d_D$. We first split each of the documents into text passages of equal lengths as the basic retrieval units² and get $M$ total passages in our corpus $C = \{p_1, p_2, \ldots, p_M\}$, where each passage $p_i$ can be viewed as a sequence of tokens $w_1^{(i)}, w_2^{(i)}, \ldots, w_{|p_i|}^{(i)}$. Given a question $q$, the task is to find a span $w_s^{(i)}, w_{s+1}^{(i)}, \ldots, w_e^{(i)}$ from one of the passages $p_i$ that can answer the question.

Notice that to cover a wide variety of domains, the corpus size can easily range from millions of documents (e.g., Wikipedia) to billions (e.g., the Web). As a result, any open-domain QA system needs to include an efficient *retriever* component that can select a small set of relevant texts, before applying the reader to extract the answer (Chen et al., 2017).³

Formally speaking, a retriever $R: (q, C) \rightarrow C_F$ is a function that takes as input a question $q$ and a corpus $C$ and returns a much smaller *filter set* of texts $C_F \subset C$, where $|C_F| = k \ll |C|$. For a fixed $k$, a *retriever* can be evaluated in isolation on *top-k retrieval accuracy*, which is the fraction of questions for which $C_F$ contains a span that answers the question.

---

## 3. Dense Passage Retriever (DPR)

We focus our research in this work on improving the *retrieval* component in open-domain QA. Given a collection of $M$ text passages, the goal of our dense passage retriever (DPR) is to index all the passages in a low-dimensional and continuous space, such that it can retrieve efficiently the top $k$ passages relevant to the input question for the reader at run-time. Note that $M$ can be very large (e.g., 21 million passages in our experiments, described in Section 4.1) and $k$ is usually small, such as 20–100.

### 3.1. Overview

Our dense passage retriever (DPR) uses a dense encoder $E_P(\cdot)$ which maps any text passage to a $d$-dimensional real-valued vector and builds an index for all the $M$ passages that we will use for retrieval. At run-time, DPR applies a different encoder $E_Q(\cdot)$ that maps the input question to a $d$-dimensional vector, and retrieves $k$ passages of which vectors are the closest to the question vector. We define the similarity between the question and the passage using the dot product of their vectors:

$$\text{sim}(q, p) = E_Q(q)^\top E_P(p) \tag{1}$$

Although more expressive model forms for measuring the similarity between a question and a passage do exist, such as networks consisting of multiple layers of cross attentions, the similarity function needs to be decomposable so that the representations of the collection of passages can be pre-computed. Most decomposable similarity functions are some transformations of Euclidean distance (L2). For instance, cosine is equivalent to inner product for unit vectors and the Mahalanobis distance is equivalent to L2 distance in a transformed space. Inner product search has been widely used and studied, as well as its connection to cosine similarity and L2 distance (Mussmann and Ermon, 2016; Ram and Gray, 2012). As our ablation study finds other similarity functions perform comparably (Section 5.2; Appendix B), we thus choose the simpler inner product function and improve the dense passage retriever by learning better encoders.

**Encoders.** Although in principle the question and passage encoders can be implemented by any neural networks, in this work we use two independent BERT (Devlin et al., 2019) networks (base, uncased) and take the representation at the `[CLS]` token as the output, so $d = 768$.

**Inference.** During inference time, we apply the passage encoder $E_P$ to all the passages and index them using FAISS (Johnson et al., 2017) offline. FAISS is an extremely efficient, open-source library for similarity search and clustering of dense vectors, which can easily be applied to billions of vectors. Given a question $q$ at run-time, we derive its embedding $v_q = E_Q(q)$ and retrieve the top $k$ passages with embeddings closest to $v_q$.

### 3.2. Training

Training the encoders so that the dot-product similarity (Eq. (1)) becomes a good ranking function for retrieval is essentially a *metric learning* problem (Kulis, 2013). The goal is to create a vector space such that relevant pairs of questions and passages will have smaller distance (i.e., higher similarity) than the irrelevant ones, by learning a better embedding function.

Let $\mathcal{D} = \{\langle q_i, p_i^+, p_{i,1}^-, \ldots, p_{i,n}^-\rangle\}_{i=1}^{m}$ be the training data that consists of $m$ instances. Each instance contains one question $q_i$ and one relevant (positive) passage $p_i^+$, along with $n$ irrelevant (negative) passages $p_{i,j}^-$. We optimize the loss function as the negative log likelihood of the positive passage:

$$L(q_i, p_i^+, p_{i,1}^-, \ldots, p_{i,n}^-) = -\log \frac{e^{\text{sim}(q_i, p_i^+)}}{e^{\text{sim}(q_i, p_i^+)} + \sum_{j=1}^{n} e^{\text{sim}(q_i, p_{i,j}^-)}} \tag{2}$$

**Positive and negative passages.** For retrieval problems, it is often the case that positive examples are available explicitly, while negative examples need to be selected from an extremely large pool. For instance, passages relevant to a question may be given in a QA dataset, or can be found using the answer. All other passages in the collection, while not specified explicitly, can be viewed as irrelevant by default. In practice, how to select negative examples is often overlooked but could be decisive for learning a high-quality encoder. We consider three different types of negatives: (1) **Random**: any random passage from the corpus; (2) **BM25**: top passages returned by BM25 which don't contain the answer but match most question tokens; (3) **Gold**: positive passages paired with other questions which appear in the training set. We will discuss the impact of different types of negative passages and training schemes in Section 5.2. Our best model uses gold passages from the same mini-batch and one BM25 negative passage. In particular, re-using gold passages from the same batch as negatives can make the computation efficient while achieving great performance. We discuss this approach below.

**In-batch negatives.** Assume that we have $B$ questions in a mini-batch and each one is associated with a relevant passage. Let $Q$ and $P$ be the $(B \times d)$ matrix of question and passage embeddings in a batch of size $B$. $S = QP^\top$ is a $(B \times B)$ matrix of similarity scores, where each row of which corresponds to a question, paired with $B$ passages. In this way, we reuse computation and effectively train on $B^2$ $(q_i, p_j)$ question/passage pairs in each batch. Any $(q_i, p_j)$ pair is a positive example when $i = j$, and negative otherwise. This creates $B$ training instances in each batch, where there are $B - 1$ negative passages for each question.

The trick of in-batch negatives has been used in the full batch setting (Yih et al., 2011) and more recently for mini-batch (Henderson et al., 2017; Gillick et al., 2019). It has been shown to be an effective strategy for learning a dual-encoder model that boosts the number of training examples.

---

## 4. Experimental Setup

In this section, we describe the data we used for experiments and the basic setup.

### 4.1. Wikipedia Data Pre-processing

Following (Lee et al., 2019), we use the English Wikipedia dump from Dec. 20, 2018 as the source documents for answering questions. We first apply the pre-processing code released in DrQA (Chen et al., 2017) to extract the clean, text-portion of articles from the Wikipedia dump. This step removes semi-structured data, such as tables, infoboxes, lists, as well as the disambiguation pages. We then split each article into multiple, disjoint text blocks of 100 words as *passages*, serving as our basic retrieval units, following (Wang et al., 2019), which results in 21,015,324 passages in the end.⁴ Each passage is also prepended with the title of the Wikipedia article where the passage is from, along with an `[SEP]` token.

### 4.2. Question Answering Datasets

We use the same five QA datasets and training/dev/testing splitting method as in previous work (Lee et al., 2019). Below we briefly describe each dataset and refer readers to their paper for the details of data preparation.

**Natural Questions (NQ)** (Kwiatkowski et al., 2019) was designed for end-to-end question answering. The questions were mined from real Google search queries and the answers were spans in Wikipedia articles identified by annotators.

**TriviaQA** (Joshi et al., 2017) contains a set of trivia questions with answers that were originally scraped from the Web.

**WebQuestions (WQ)** (Berant et al., 2013) consists of questions selected using Google Suggest API, where the answers are entities in Freebase.

**CuratedTREC** (TREC) (Baudiš and Šedivý, 2015) sources questions from TREC QA tracks as well as various Web sources and is intended for open-domain QA from unstructured corpora.

**SQuAD v1.1** (Rajpurkar et al., 2016) is a popular benchmark dataset for reading comprehension. Annotators were presented with a Wikipedia paragraph, and asked to write questions that could be answered from the given text. Although SQuAD has been used previously for open-domain QA research, it is not ideal because many questions lack context in absence of the provided paragraph. We still include it in our experiments for providing a fair comparison to previous work and we will discuss more in Section 5.1.

**Selection of positive passages.** Because only pairs of questions and answers are provided in TREC, WebQuestions and TriviaQA⁵, we use the highest-ranked passage from BM25 that contains the answer as the positive passage. If none of the top 100 retrieved passages has the answer, the question will be discarded. For SQuAD and Natural Questions, since the original passages have been split and processed differently than our pool of candidate passages, we match and replace each gold passage with the corresponding passage in the candidate pool.⁶ We discard the questions when the matching is failed due to different Wikipedia versions or pre-processing. Table 1 shows the number of questions in training/dev/test sets for all the datasets and the actual questions used for training the retriever.

#### Table 1: Number of questions in each QA dataset.

| Dataset | Train (original) | Train (used) | Dev | Test |
|---------|:---------------:|:-----------:|:---:|:----:|
| Natural Questions | 79,168 | 58,880 | 8,757 | 3,610 |
| TriviaQA | 78,785 | 60,413 | 8,837 | 11,313 |
| WebQuestions | 3,417 | 2,474 | 361 | 2,032 |
| CuratedTREC | 1,353 | 1,125 | 133 | 694 |
| SQuAD | 78,713 | 70,096 | 8,886 | 10,570 |

The two columns of *Train* denote the original training examples in the dataset and the actual questions used for training DPR after filtering. See text for more details.

---

## 5. Experiments: Passage Retrieval

In this section, we evaluate the retrieval performance of our Dense Passage Retriever (DPR), along with analysis on how its output differs from traditional retrieval methods, the effects of different training schemes and the run-time efficiency.

The DPR model used in our main experiments is trained using the in-batch negative setting (Section 3.2) with a batch size of 128 and one additional BM25 negative passage per question. We trained the question and passage encoders for up to 40 epochs for large datasets (NQ, TriviaQA, SQuAD) and 100 epochs for small datasets (TREC, WQ), with a learning rate of $10^{-5}$ using Adam, linear scheduling with warm-up and dropout rate 0.1.

While it is good to have the flexibility to adapt the retriever to each dataset, it would also be desirable to obtain a single retriever that works well across the board. To this end, we train a *multi-dataset* encoder by combining training data from all datasets excluding SQuAD.⁷ In addition to DPR, we also present the results of BM25, the traditional retrieval method⁸ and BM25+DPR, using a linear combination of their scores as the new ranking function. Specifically, we obtain two initial sets of top-2000 passages based on BM25 and DPR, respectively, and rerank the union of them using $\text{BM25}(q,p) + \lambda \cdot \text{sim}(q,p)$ as the ranking function. We used $\lambda = 1.1$ based on the retrieval accuracy in the development set.

### 5.1. Main Results

Table 2 compares different passage retrieval systems on five QA datasets, using the top-$k$ accuracy ($k \in \{20, 100\}$). With the exception of SQuAD, DPR performs consistently better than BM25 on all datasets. The gap is especially large when $k$ is small (e.g., 78.4% vs. 59.1% for top-20 accuracy on Natural Questions). When training with multiple datasets, TREC, the smallest dataset of the five, benefits greatly from more training examples. In contrast, Natural Questions and WebQuestions improve modestly and TriviaQA degrades slightly. Results can be improved further in some cases by combining DPR with BM25 in both single- and multi-dataset settings.

We conjecture that the lower performance on SQuAD is due to two reasons. First, the annotators wrote questions after seeing the passage. As a result, there is a high lexical overlap between passages and questions, which gives BM25 a clear advantage. Second, the data was collected from only 500+ Wikipedia articles and thus the distribution of training examples is extremely biased, as argued previously by Lee et al. (2019).

#### Table 2: Top-20 & Top-100 retrieval accuracy on test sets.

Measured as the percentage of top 20/100 retrieved passages that contain the answer. *Single* and *Multi* denote that our Dense Passage Retriever (DPR) was trained using individual or combined training datasets (all the datasets excluding SQuAD). See text for more details.

| Training | Retriever | NQ Top-20 | TriviaQA Top-20 | WQ Top-20 | TREC Top-20 | SQuAD Top-20 | NQ Top-100 | TriviaQA Top-100 | WQ Top-100 | TREC Top-100 | SQuAD Top-100 |
|----------|-----------|:---------:|:---------------:|:---------:|:-----------:|:------------:|:----------:|:----------------:|:----------:|:------------:|:-------------:|
| None | BM25 | 59.1 | 66.9 | 55.0 | 70.9 | 68.8 | 73.7 | 76.7 | 71.1 | 84.1 | 80.0 |
| Single | DPR | **78.4** | **79.4** | **73.2** | **79.8** | 63.2 | **85.4** | **85.0** | **81.4** | **89.1** | 77.2 |
| Single | BM25 + DPR | 76.6 | 79.8 | 71.0 | 85.2 | **71.5** | 83.8 | 84.5 | 80.5 | 92.7 | **81.3** |
| Multi | DPR | **79.4** | 78.8 | **75.0** | **89.1** | 51.6 | **86.0** | **84.7** | **82.9** | **93.9** | 67.6 |
| Multi | BM25 + DPR | 78.0 | **79.9** | 74.7 | **88.5** | **66.2** | 83.9 | 84.4 | 82.3 | **94.1** | **78.6** |

> **Figure 1.** Retriever top-$k$ accuracy with different numbers of training examples used in our dense passage retriever vs BM25. The results are measured on the development set of Natural Questions. Our DPR trained using 1,000 examples already outperforms BM25. (X-axis: $k$: # of retrieved passages, 20–100; Y-axis: Top-k accuracy (%), 40–90. Curves for BM25, # Train: 1k, 10k, 20k, 40k, all (59k).)

### 5.2. Ablation Study on Model Training

To understand further how different model training options affect the results, we conduct several additional experiments and discuss our findings below.

**Sample efficiency.** We explore how many training examples are needed to achieve good passage retrieval performance. Figure 1 illustrates the top-$k$ retrieval accuracy with respect to different numbers of training examples, measured on the development set of Natural Questions. As is shown, a dense passage retriever trained using only 1,000 examples already outperforms BM25. This suggests that with a general pretrained language model, it is possible to train a high-quality dense retriever with a small number of question–passage pairs. Adding more training examples (from 1k to 59k) further improves the retrieval accuracy consistently.

**In-batch negative training.** We test different training schemes on the development set of Natural Questions and summarize the results in Table 3. The top block is the standard 1-of-$N$ training setting, where each question in the batch is paired with a positive passage and its own set of $n$ negative passages (Eq. (2)). We find that the choice of negatives — random, BM25 or gold passages (positive passages from other questions) — does not impact the top-$k$ accuracy much in this setting when $k \geq 20$.

The middle block is the in-batch negative training (Section 3.2) setting. We find that using a similar configuration (7 gold negative passages), in-batch negative training improves the results substantially. The key difference between the two is whether the gold negative passages come from the same batch or from the whole training set. Effectively, in-batch negative training is an easy and memory-efficient way to reuse the negative examples already in the batch rather than creating new ones. It produces more pairs and thus increases the number of training examples, which might contribute to the good model performance. As a result, accuracy consistently improves as the batch size grows.

Finally, we explore in-batch negative training with additional "hard" negative passages that have high BM25 scores given the question, but do not contain the answer string (the bottom block). These additional passages are used as negative passages for all questions in the same batch. We find that adding a single BM25 negative passage improves the result substantially while adding two does not help further.

#### Table 3: Comparison of different training schemes.

Measured as top-$k$ retrieval accuracy on Natural Questions (development set). #N: number of negative examples, IB: in-batch training. G.+BM25(1) and G.+BM25(2) denote in-batch training with 1 or 2 additional BM25 negatives, which serve as negative passages for all questions in the batch.

| Type | #N | IB | Top-5 | Top-20 | Top-100 |
|------|:--:|:--:|:-----:|:------:|:-------:|
| Random | 7 | 7 | 47.0 | 64.3 | 77.8 |
| BM25 | 7 | 7 | 50.0 | 63.3 | 74.8 |
| Gold | 7 | 7 | 42.6 | 63.1 | 78.3 |
| Gold | 7 | 3 | 51.1 | 69.1 | 80.8 |
| Gold | 31 | 3 | 52.1 | 70.8 | 82.1 |
| Gold | 127 | 3 | 55.8 | 73.0 | 83.1 |
| G.+BM25(1) | 31+32 | 3 | **65.0** | **77.3** | 84.4 |
| G.+BM25(2) | 31+64 | 3 | 64.5 | 76.4 | 84.0 |
| G.+BM25(1) | 127+128 | 3 | **65.8** | **78.0** | **84.9** |

**Impact of gold passages.** We use passages that match the gold contexts in the original datasets (when available) as positive examples (Section 4.2). Our experiments on Natural Questions show that switching to distantly-supervised passages (using the highest-ranked BM25 passage that contains the answer), has only a small impact: 1 point lower top-$k$ accuracy for retrieval. Appendix A contains more details.

**Similarity and loss.** Besides dot product, cosine and Euclidean L2 distance are also commonly used as decomposable similarity functions. We test these alternatives and find that L2 performs comparable to dot product, and both of them are superior to cosine. Similarly, in addition to negative log-likelihood, a popular option for ranking is triplet loss, which compares a positive passage and a negative one directly with respect to a question (Burges et al., 2005). Our experiments show that using triplet loss does not affect the results much. More details can be found in Appendix B.

**Cross-dataset generalization.** One interesting question regarding DPR's discriminative training is how much performance degradation it may suffer from a non-iid setting. In other words, can it still generalize well when directly applied to a different dataset without additional fine-tuning? To test the cross-dataset generalization, we train DPR on Natural Questions only and test it directly on the smaller WebQuestions and CuratedTREC datasets. We find that DPR generalizes well, with 3-5 points loss from the best performing fine-tuned model in top-20 retrieval accuracy (69.9/86.3 vs. 75.0/89.1 for WebQuestions and TREC, respectively), while still greatly outperforming the BM25 baseline (55.0/70.9).

### 5.3. Qualitative Analysis

Although DPR performs better than BM25 in general, passages retrieved by these two methods differ qualitatively. Term-matching methods like BM25 are sensitive to highly selective keywords and phrases, while DPR captures lexical variations or semantic relationships better. See Appendix C for examples and more discussion.

### 5.4. Run-time Efficiency

The main reason that we require a retrieval component for open-domain QA is to reduce the number of candidate passages that the reader needs to consider, which is crucial for answering user's questions in real-time. We profiled the passage retrieval speed on a server with Intel Xeon CPU E5-2698 v4 @ 2.20GHz and 512GB memory. With the help of FAISS in-memory index for real-valued vectors⁹, DPR can be made incredibly efficient, processing **995.0 questions per second**, returning top 100 passages per question. In contrast, BM25/Lucene (implemented in Java, using file index) processes 23.7 questions per second per CPU thread.

On the other hand, the time required for building an index for dense vectors is much longer. Computing dense embeddings on 21-million passages is resource intensive, but can be easily parallelized, taking roughly 8.8 hours on 8 GPUs. However, building the FAISS index on 21-million vectors on a single server takes 8.5 hours. In comparison, building an inverted index using Lucene is much cheaper and takes only about 30 minutes in total.

---

## 6. Experiments: Question Answering

In this section, we experiment with how different passage retrievers affect the final QA accuracy.

### 6.1. End-to-end QA System

We implement an end-to-end question answering system in which we can plug different retriever systems directly. Besides the retriever, our QA system consists of a neural *reader* that outputs the answer to the question. Given the top $k$ retrieved passages (up to 100 in our experiments), the reader assigns a passage selection score to each passage. In addition, it extracts an answer span from each passage and assigns a span score. The best span from the passage with the highest passage selection score is chosen as the final answer.

The passage selection model serves as a reranker through cross-attention between the question and the passage. Although cross-attention is not feasible for retrieving relevant passages in a large corpus due to its non-decomposable nature, it has more capacity than the dual-encoder model $\text{sim}(q, p)$ as in Eq. (1). Applying it to selecting the passage from a small number of retrieved candidates has been shown to work well (Wang et al., 2019, 2018; Lin et al., 2018).

Specifically, let $P_i \in \mathbb{R}^{L \times h}$ ($1 \leq i \leq k$) be a BERT (base, uncased in our experiments) representation for the $i$-th passage, where $L$ is the maximum length of the passage and $h$ the hidden dimension. The probabilities of a token being the starting/ending positions of an answer span and a passage being selected are defined as:

$$P_{\text{start},i}(s) = \text{softmax}(P_i w_{\text{start}})_s \tag{3}$$

$$P_{\text{end},i}(t) = \text{softmax}(P_i w_{\text{end}})_t \tag{4}$$

$$P_{\text{selected}}(i) = \text{softmax}(\hat{P}^\top w_{\text{selected}})_i \tag{5}$$

where $\hat{P} = [P_1^{[\text{CLS}]}, \ldots, P_k^{[\text{CLS}]}] \in \mathbb{R}^{h \times k}$ and $w_{\text{start}}, w_{\text{end}}, w_{\text{selected}} \in \mathbb{R}^h$ are learnable vectors.

We compute a span score of the $s$-th to $t$-th words from the $i$-th passage as $P_{\text{start},i}(s) \times P_{\text{end},i}(t)$, and a passage selection score of the $i$-th passage as $P_{\text{selected}}(i)$.

During training, we sample one positive and $\tilde{m} - 1$ negative passages from the top 100 passages returned by the retrieval system (BM25 or DPR) for each question. $\tilde{m}$ is a hyper-parameter and we use $\tilde{m} = 24$ in all the experiments. The training objective is to maximize the marginal log-likelihood of all the correct answer spans in the positive passage (the answer string may appear multiple times in one passage), combined with the log-likelihood of the positive passage being selected. We use the batch size of 16 for large (NQ, TriviaQA, SQuAD) and 4 for small (TREC, WQ) datasets, and tune $k$ on the development set. For experiments on small datasets under the *Multi* setting, in which using other datasets is allowed, we fine-tune the reader trained on Natural Questions to the target dataset. All experiments were done on eight 32GB GPUs.

### 6.2. Results

Table 4 summarizes our final end-to-end QA results, measured by *exact match* with the reference answer after minor normalization as in (Chen et al., 2017; Lee et al., 2019). From the table, we can see that higher retriever accuracy typically leads to better final QA results: in all cases except SQuAD, answers extracted from the passages retrieved by DPR are more likely to be correct, compared to those from BM25.

For large datasets like NQ and TriviaQA, models trained using multiple datasets (*Multi*) perform comparably to those trained using the individual training set (*Single*). Conversely, on smaller datasets like WQ and TREC, the multi-dataset setting has a clear advantage. Overall, our DPR-based models outperform the previous state-of-the-art results on four out of the five datasets, with 1% to 12% absolute differences in exact match accuracy.

It is interesting to contrast our results to those of ORQA (Lee et al., 2019) and also the concurrently developed approach, REALM (Guu et al., 2020). While both methods include additional pre-training tasks and employ an expensive end-to-end training regime, DPR manages to outperform them on both NQ and TriviaQA, simply by focusing on learning a strong passage retrieval model using pairs of questions and answers. The additional pre-training tasks are likely more useful only when the target training sets are small. Although the results of DPR on WQ and TREC in the single-dataset setting are less competitive, adding more question–answer pairs helps boost the performance, achieving the new state of the art.

#### Table 4: End-to-end QA (Exact Match) Accuracy.

The first block of results are copied from their cited papers. REALM$_{\text{Wiki}}$ and REALM$_{\text{News}}$ are the same model but pretrained on Wikipedia and CC-News, respectively. *Single* and *Multi* denote that our Dense Passage Retriever (DPR) is trained using individual or combined training datasets (all except SQuAD). For WQ and TREC in the *Multi* setting, we fine-tune the reader trained on NQ.

| Training | Model | NQ | TriviaQA | WQ | TREC | SQuAD |
|----------|-------|:--:|:--------:|:--:|:----:|:-----:|
| Single | BM25+BERT (Lee et al., 2019) | 26.5 | 47.1 | 17.7 | 21.3 | 33.2 |
| Single | ORQA (Lee et al., 2019) | 33.3 | 45.0 | 36.4 | 30.1 | 20.2 |
| Single | HardEM (Min et al., 2019a) | 28.1 | 50.9 | - | - | - |
| Single | GraphRetriever (Min et al., 2019b) | 34.5 | 56.0 | 36.4 | - | - |
| Single | PathRetriever (Asai et al., 2020) | 32.6 | - | - | - | 56.5 |
| Single | REALM$_{\text{Wiki}}$ (Guu et al., 2020) | 39.2 | - | 40.2 | 46.8 | - |
| Single | REALM$_{\text{News}}$ (Guu et al., 2020) | 40.4 | - | 40.7 | 42.9 | - |
| Single | BM25 | 32.6 | 52.4 | 29.9 | 24.9 | 38.1 |
| Single | **DPR** | **41.5** | **56.8** | **34.6** | 25.9 | 29.8 |
| Single | BM25+DPR | 39.0 | 57.0 | 35.2 | 28.0 | **36.7** |
| Multi | **DPR** | **41.5** | **56.8** | **42.4** | **49.4** | 24.1 |
| Multi | BM25+DPR | 38.8 | **57.9** | 41.1 | **50.6** | **35.8** |

To compare our pipeline training approach with joint learning, we run an ablation on Natural Questions where the retriever and reader are jointly trained, following Lee et al. (2019). This approach obtains a score of 39.8 EM, which suggests that our strategy of training a strong retriever and reader in isolation can leverage effectively available supervision, while outperforming a comparable joint training approach with a simpler design (Appendix D).

One thing worth noticing is that our reader does consider more passages compared to ORQA, although it is not completely clear how much more time it takes for inference. While DPR processes up to 100 passages for each question, the reader is able to fit all of them into one batch on a single 32GB GPU, thus the latency remains almost identical to the single passage case (around 20ms). The exact impact on throughput is harder to measure: ORQA uses 2-3x longer passages compared to DPR (288 word pieces compared to our 100 tokens) and the computational complexity is super-linear in passage length. We also note that we found $k = 50$ to be optimal for NQ, and $k = 10$ leads to only marginal loss in exact match accuracy (40.8 vs. 41.5 EM on NQ), which should be roughly comparable to ORQA's 5-passage setup.

---

## 7. Related Work

Passage retrieval has been an important component for open-domain QA (Voorhees, 1999). It not only effectively reduces the search space for answer extraction, but also identifies the support context for users to verify the answer. Strong sparse vector space models like TF-IDF or BM25 have been used as the standard method applied broadly to various QA tasks (e.g., Chen et al., 2017; Yang et al., 2019a,b; Nie et al., 2019; Min et al., 2019a; Wolfson et al., 2020). Augmenting text-based retrieval with external structured information, such as knowledge graph and Wikipedia hyperlinks, has also been explored recently (Min et al., 2019b; Asai et al., 2020).

The use of dense vector representations for retrieval has a long history since Latent Semantic Analysis (Deerwester et al., 1990). Using labeled pairs of queries and documents, discriminatively trained dense encoders have become popular recently (Yih et al., 2011; Huang et al., 2013; Gillick et al., 2019), with applications to cross-lingual document retrieval, ad relevance prediction, Web search and entity retrieval. Such approaches complement the sparse vector methods as they can potentially give high similarity scores to semantically relevant text pairs, even without exact token matching. The dense representation alone, however, is typically inferior to the sparse one. While not the focus of this work, dense representations from pretrained models, along with cross-attention mechanisms, have also been shown effective in passage or dialogue re-ranking tasks (Nogueira and Cho, 2019; Humeau et al., 2020). Finally, a concurrent work (Khattab and Zaharia, 2020) demonstrates the feasibility of full dense retrieval in IR tasks. Instead of employing the dual-encoder framework, they introduced a late-interaction operator on top of the BERT encoders.

Dense retrieval for open-domain QA has been explored by Das et al. (2019), who propose to retrieve relevant passages iteratively using reformulated question vectors. As an alternative approach that skips passage retrieval, Seo et al. (2019) propose to encode candidate answer phrases as vectors and directly retrieve the answers to the input questions efficiently. Using additional pre-training with the objective that matches surrogates of questions and relevant passages, Lee et al. (2019) jointly train the question encoder and reader. Their approach outperforms the BM25 plus reader paradigm on multiple open-domain QA datasets in QA accuracy, and is further extended by REALM (Guu et al., 2020), which includes tuning the passage encoder asynchronously by re-indexing the passages during training. The pre-training objective has also recently been improved by Xiong et al. (2020b). In contrast, our model provides a simple and yet effective solution that shows stronger empirical performance, without relying on additional pre-training or complex joint training schemes.

DPR has also been used as an important module in very recent work. For instance, extending the idea of leveraging hard negatives, Xiong et al. (2020a) use the retrieval model trained in the previous iteration to discover new negatives and construct a different set of examples in each training iteration. Starting from our trained DPR model, they show that the retrieval performance can be further improved. Recent work (Izacard and Grave, 2020; Lewis et al., 2020b) have also shown that DPR can be combined with generation models such as BART (Lewis et al., 2020a) and T5 (Raffel et al., 2019), achieving good performance on open-domain QA and other knowledge-intensive tasks.

---

## 8. Conclusion

In this work, we demonstrated that dense retrieval can outperform and potentially replace the traditional sparse retrieval component in open-domain question answering. While a simple dual-encoder approach can be made to work surprisingly well, we showed that there are some critical ingredients to training a dense retriever successfully. Moreover, our empirical analysis and ablation studies indicate that more complex model frameworks or similarity functions do not necessarily provide additional values. As a result of improved retrieval performance, we obtained new state-of-the-art results on multiple open-domain question answering benchmarks.

---

## Acknowledgments

We thank the anonymous reviewers for their helpful comments and suggestions.

---

## References

- Akari Asai, Kazuma Hashimoto, Hannaneh Hajishirzi, Richard Socher, and Caiming Xiong. 2020. Learning to retrieve reasoning paths over Wikipedia graph for question answering. In *International Conference on Learning Representations (ICLR)*.
- Petr Baudiš and Jan Šedivý. 2015. Modeling of the question answering task in the YodaQA system. In *International Conference of the Cross-Language Evaluation Forum for European Languages*, pages 222–228. Springer.
- Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic parsing on Freebase from question-answer pairs. In *Empirical Methods in Natural Language Processing (EMNLP)*.
- Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, and Roopak Shah. 1994. Signature verification using a "Siamese" time delay neural network. In *NIPS*, pages 737–744.
- Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005. Learning to rank using gradient descent. In *Proceedings of the 22nd international conference on Machine learning*, pages 89–96.
- Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer open-domain questions. In *Association for Computational Linguistics (ACL)*, pages 1870–1879.
- Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, and Andrew McCallum. 2019. Multi-step retriever-reader interaction for scalable open-domain question answering. In *International Conference on Learning Representations (ICLR)*.
- Scott Deerwester, Susan T Dumais, George W Furnas, Thomas K Landauer, and Richard Harshman. 1990. Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6):391–407.
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In *North American Association for Computational Linguistics (NAACL)*.
- David A Ferrucci. 2012. Introduction to "This is Watson". *IBM Journal of Research and Development*, 56(3.4):1–1.
- Daniel Gillick, Sayali Kulkarni, Larry Lansing, Alessandro Presta, Jason Baldridge, Eugene Ie, and Diego Garcia-Olano. 2019. Learning dense representations for entity retrieval. In *Computational Natural Language Learning (CoNLL)*.
- Ruiqi Guo, Sanjiv Kumar, Krzysztof Choromanski, and David Simcha. 2016. Quantization based fast inner product search. In *Artificial Intelligence and Statistics*, pages 482–490.
- Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: Retrieval-augmented language model pre-training. *ArXiv*, abs/2002.08909.
- Matthew Henderson, Rami Al-Rfou, Brian Strope, Yunhsuan Sung, László Lukács, Ruiqi Guo, Sanjiv Kumar, Balint Miklos, and Ray Kurzweil. 2017. Efficient natural language response suggestion for smart reply. *ArXiv*, abs/1705.00652.
- Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for Web search using clickthrough data. In *ACM International Conference on Information and Knowledge Management (CIKM)*, pages 2333–2338.
- Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In *International Conference on Learning Representations (ICLR)*.
- Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. *ArXiv*, abs/2007.01282.
- Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs. *ArXiv*, abs/1702.08734.
- Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In *Association for Computational Linguistics (ACL)*, pages 1601–1611.
- Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In *ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)*, pages 39–48.
- Brian Kulis. 2013. Metric learning: A survey. *Foundations and Trends in Machine Learning*, 5(4):287–364.
- Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. *Transactions of the Association of Computational Linguistics (TACL)*.
- Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In *Association for Computational Linguistics (ACL)*, pages 6086–6096.
- Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020a. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In *Association for Computational Linguistics (ACL)*, pages 7871–7880.
- Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020b. Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems (NeurIPS)*.
- Yankai Lin, Haozhe Ji, Zhiyuan Liu, and Maosong Sun. 2018. Denoising distantly supervised open-domain question answering. In *Association for Computational Linguistics (ACL)*, pages 1736–1745.
- Sewon Min, Danqi Chen, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2019a. A discrete hard EM approach for weakly supervised question answering. In *Empirical Methods in Natural Language Processing (EMNLP)*.
- Sewon Min, Danqi Chen, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2019b. Knowledge guided text retrieval and reading for open domain question answering. *ArXiv*, abs/1911.03868.
- Dan Moldovan, Marius Pașca, Sanda Harabagiu, and Mihai Surdeanu. 2003. Performance issues and error analysis in an open-domain question answering system. *ACM Transactions on Information Systems (TOIS)*, 21(2):133–154.
- Stephen Mussmann and Stefano Ermon. 2016. Learning and inference via maximum inner product search. In *International Conference on Machine Learning (ICML)*, pages 2587–2596.
- Yixin Nie, Songhe Wang, and Mohit Bansal. 2019. Revealing the importance of semantic retrieval for machine reading at scale. In *Empirical Methods in Natural Language Processing (EMNLP)*.
- Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with BERT. *ArXiv*, abs/1901.04085.
- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. *ArXiv*, abs/1910.10683.
- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In *Empirical Methods in Natural Language Processing (EMNLP)*, pages 2383–2392.
- Parikshit Ram and Alexander G Gray. 2012. Maximum inner-product search using cone trees. In *Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 931–939.
- Adam Roberts, Colin Raffel, and Noam Shazeer. 2020. How much knowledge can you pack into the parameters of a language model? *ArXiv*, abs/2002.08910.
- Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4):333–389.
- Minjoon Seo, Jinhyuk Lee, Tom Kwiatkowski, Ankur Parikh, Ali Farhadi, and Hannaneh Hajishirzi. 2019. Real-time open-domain question answering with dense-sparse phrase index. In *Association for Computational Linguistics (ACL)*.
- Anshumali Shrivastava and Ping Li. 2014. Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS). In *Advances in Neural Information Processing Systems (NIPS)*, pages 2321–2329.
- Ellen M Voorhees. 1999. The TREC-8 question answering track report. In *TREC*, volume 99, pages 77–82.
- Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang, Tim Klinger, Wei Zhang, Shiyu Chang, Gerry Tesauro, Bowen Zhou, and Jing Jiang. 2018. R³: Reinforced ranker-reader for open-domain question answering. In *Conference on Artificial Intelligence (AAAI)*.
- Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallapati, and Bing Xiang. 2019. Multi-passage BERT: A globally normalized BERT model for open-domain question answering. In *Empirical Methods in Natural Language Processing (EMNLP)*.
- Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, and Jonathan Berant. 2020. Break it down: A question understanding benchmark. *Transactions of the Association of Computational Linguistics (TACL)*.
- Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020a. Approximate nearest neighbor negative contrastive learning for dense text retrieval. *ArXiv*, abs/2007.00808.
- Wenhan Xiong, Hankang Wang, and William Yang Wang. 2020b. Progressively pretrained dense corpus index for open-domain question answering. *ArXiv*, abs/2005.00038.
- Wei Yang, Yuqing Xie, Aileen Lin, Xingyu Li, Luchen Tan, Kun Xiong, Ming Li, and Jimmy Lin. 2019a. End-to-end open-domain question answering with BERTserini. In *North American Association for Computational Linguistics (NAACL)*, pages 72–77.
- Wei Yang, Yuqing Xie, Luchen Tan, Kun Xiong, Ming Li, and Jimmy Lin. 2019b. Data augmentation for BERT fine-tuning in open-domain question answering. *ArXiv*, abs/1904.06652.
- Wen-tau Yih, Kristina Toutanova, John C Platt, and Christopher Meek. 2011. Learning discriminative projections for text similarity measures. In *Computational Natural Language Learning (CoNLL)*, pages 247–256.

---

## Appendix A. Distant Supervision

When training our final DPR model using Natural Questions, we use the passages in our collection that best match the gold context as the positive passages. As some QA datasets contain only the question and answer pairs, it is thus interesting to see when using the passages that contain the answers as positives (i.e., the distant supervision setting), whether there is a significant performance degradation. Using the question and answer together as the query, we run Lucene-BM25 and pick the top passage that contains the answer as the positive passage. Table 5 shows the performance of DPR when trained using the original setting and the distant supervision setting.

#### Table 5: Retrieval accuracy on the development set of Natural Questions.

Trained on passages that match the gold context (Gold) or the top BM25 passage that contains the answer (Dist. Sup.).

| | Top-1 | Top-5 | Top-20 | Top-100 |
|---|:-----:|:-----:|:------:|:-------:|
| Gold | 44.9 | 66.8 | 78.1 | 85.0 |
| Dist. Sup. | 43.9 | 65.3 | 77.1 | 84.4 |

---

## Appendix B. Alternative Similarity Functions & Triplet Loss

In addition to dot product (DP) and negative log-likelihood based on softmax (NLL), we also experiment with Euclidean distance (L2) and the triplet loss. We negate L2 similarity scores before applying softmax and change signs of question-to-positive and question-to-negative similarities when applying the triplet loss on dot product scores. The margin value of the triplet loss is set to 1. Table 6 summarizes the results. All these additional experiments are conducted using the same hyper-parameters tuned for the baseline (DP, NLL).

Note that the retrieval accuracy for our "baseline" settings reported in Table 5 (Gold) and Table 6 (DP, NLL) is slightly better than those reported in Table 3. This is due to a better hyper-parameter setting used in these analysis experiments, which is documented in our code release.

#### Table 6: Retrieval Top-$k$ accuracy on the development set of Natural Questions using different similarity and loss functions.

| Sim | Loss | Top-1 | Top-5 | Top-20 | Top-100 |
|-----|------|:-----:|:-----:|:------:|:-------:|
| DP | NLL | 44.9 | 66.8 | 78.1 | 85.0 |
| DP | Triplet | 41.6 | 65.0 | 77.2 | 84.5 |
| L2 | NLL | 43.5 | 64.7 | 76.1 | 83.1 |
| L2 | Triplet | 42.2 | 66.0 | 78.1 | 84.9 |

---

## Appendix C. Qualitative Analysis

Although DPR performs better than BM25 in general, the retrieved passages of these two retrievers actually differ qualitatively. Methods like BM25 are sensitive to highly selective keywords and phrases, but cannot capture lexical variations or semantic relationships well. In contrast, DPR excels at semantic representation, but might lack sufficient capacity to represent salient phrases which appear rarely. Table 7 illustrates this phenomenon with two examples. In the first example, the top scoring passage from BM25 is irrelevant, even though keywords such as *England* and *Ireland* appear multiple times. In comparison, DPR is able to return the correct answer, presumably by matching *"body of water"* with semantic neighbors such as *sea* and *channel*, even though no lexical overlap exists. The second example is one where BM25 does better. The salient phrase *"Thoros of Myr"* is critical, and DPR is unable to capture it.

#### Table 7: Examples of passages returned from BM25 and DPR.

Correct answers are written in blue and the content words in the question are written in bold.

| Question | Passage received by BM25 | Passage retrieved by DPR |
|----------|--------------------------|--------------------------|
| What is the **body of water** between **England** and **Ireland**? | **Title: British Cycling.** . . . *England* is not recognised as a region by the UCI, and there is no English cycling team outside the Commonwealth Games. For those occasions, British Cycling selects and supports the *England* team. Cycling is represented on the Isle of Man by the Isle of Man Cycling Association. Cycling in Northern *Ireland* is organised under Cycling Ulster, part of the all-Ireland governing *body* Cycling *Ireland*. Until 2006, a rival governing *body* existed, . . . | **Title: Irish Sea.** . . . Annual traffic between Great Britain and *Ireland* amounts to over 12 million passengers and of traded goods. The *Irish Sea* is connected to the North Atlantic at both its northern and southern ends. To the north, the connection is through the North Channel between Scotland and Northern *Ireland* and the Malin Sea. The southern end is linked to the Atlantic through the St George's Channel between *Ireland* and Pembrokeshire, and the Celtic Sea. . . . |
| Who plays **Thoros of Myr** in **Game of Thrones**? | **Title: No One (Game of Thrones).** . . . He may be "no one," but there's still enough of a person left in him to respect, and admire who this girl is and what she's become. Arya finally tells us something that we've kind of known all along, that she's not no one, she's Arya Stark of Winterfell." "No One" saw the reintroduction of Richard Dormer and Paul Kaye, who portrayed Beric Dondarrion and **Thoros of Myr**, respectively, in the third season, . . . | **Title: Pål Sverre Hagen.** Pål Sverre Valheim Hagen (born 6 November 1980) is a Norwegian stage and screen actor. He appeared in the Norwegian film "Max Manus" and played Thor Heyerdahl in the Oscar-nominated 2012 film "Kon-Tiki". P. Hagen was born in Stavanger, Norway, the son of Roar Hagen, a Norwegian cartoonist who has long been associated with Norway's largest daily, "VG". He lived in Jæten, a neighborhood in the city of Stavanger in south-western Norway. . . . |

---

## Appendix D. Joint Training of Retriever and Reader

We fix the passage encoder in our joint-training scheme while allowing only the question encoder to receive backpropagation signal from the combined (retriever + reader) loss function. This allows us to leverage the HNSW-based FAISS index for efficient low-latency retrieving, without re-indexing the passages during model updates. Our loss function largely follows ORQA's approach, which uses log probabilities of positive passages selected from the retriever model, and correct spans and passages selected from the reader model. Since the passage encoder is fixed, we could use larger amount of retrieved passages when calculating the retriever loss. Specifically, we get top 100 passages for each question in a mini-batch and use the method similar to in-batch negative training: all retrieved passages' vectors participate in the loss calculation for *all* questions in a batch. Our training batch size is set to 16, which effectively gives 1,600 passages per question to calculate retriever loss. The reader still uses 24 passages per question, which are selected from the top 5 positive and top 30 negative passages (from the set of top 100 passages retrieved from the same question). The question encoder's initial state is taken from a DPR model previously trained on the NQ dataset. The reader's initial state is a BERT-base model. In terms of the end-to-end QA results, our joint-training scheme does not provide better results compared to the usual retriever/reader training pipeline, resulting in the same 39.8 exact match score on NQ dev as in our regular reader model training.

---

## Footnotes

1. For instance, the exact match score on SQuAD v1.1 drops from above 80% to less than 40% (Yang et al., 2019a).
2. The ideal size and boundary of a text passage are functions of both the retriever and reader. We also experimented with natural paragraphs in our preliminary trials and found that using fixed-length passages performs better in both retrieval and final QA accuracy, as observed by Wang et al. (2019).
3. Exceptions include (Seo et al., 2019) and (Roberts et al., 2020), which *retrieves* and *generates* the answers, respectively.
4. However, Wang et al. (2019) also propose splitting documents into overlapping passages, which we do not find advantageous compared to the non-overlapping version.
5. We use the unfiltered TriviaQA version and discard the noisy evidence documents mined from Bing.
6. The improvement of using gold contexts over passages that contain answers is small. See Section 5.2 and Appendix A.
7. SQuAD is limited to a small set of Wikipedia documents and thus introduces unwanted bias. We will discuss this issue more in Section 5.1.
8. Lucene implementation. BM25 parameters $b = 0.4$ (document length normalization) and $k_1 = 0.9$ (term frequency scaling) are tuned using development sets.
9. FAISS configuration: we used HNSW index type on CPU, neighbors to store per node = 512, construction time search depth = 200, search depth = 128.

---

The code and trained models have been released at [https://github.com/facebookresearch/DPR](https://github.com/facebookresearch/DPR).

read the article and return me articles's text as a Markdown file.
markdown

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Patrick Lewis**†‡, **Ethan Perez**§, **Aleksandra Piktus**†, **Fabio Petroni**†, **Vladimir Karpukhin**†, **Naman Goyal**†, **Heinrich Küttler**†, **Mike Lewis**†, **Wen-tau Yih**†, **Tim Rocktäschel**†‡, **Sebastian Riedel**†‡, **Douwe Kiela**†

† Facebook AI Research
‡ University College London
§ New York University

plewis@fb.com

arXiv:2005.11401v4 [cs.CL] 12 Apr 2021

---

## Abstract

Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for **retrieval-augmented generation (RAG)** — models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, and another which can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state of the art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.

---

## 1. Introduction

Pre-trained neural language models have been shown to learn a substantial amount of in-depth knowledge from data [47]. They can do so without any access to an external memory, as a parameterized implicit knowledge base [51, 52]. While this development is exciting, such models do have downsides: They cannot easily expand or revise their memory, can't straightforwardly provide insight into their predictions, and may produce "hallucinations" [38]. Hybrid models that combine parametric memory with non-parametric (i.e., retrieval-based) memories [20, 26, 48] can address some of these issues because knowledge can be directly revised and expanded, and accessed knowledge can be inspected and interpreted. REALM [20] and ORQA [31], two recently introduced models that combine masked language models [8] with a differentiable retriever, have shown promising results, but have only explored open-domain extractive question answering. Here, we bring hybrid parametric and non-parametric memory to the "workhorse of NLP," i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose fine-tuning approach which we refer to as **retrieval-augmented generation (RAG)**.

> **Figure 1.** Overview of our approach. We combine a pre-trained retriever (*Query Encoder* + *Document Index*) with a pre-trained seq2seq model (*Generator*) and fine-tune end-to-end. For query $x$, we use Maximum Inner Product Search (MIPS) to find the top-$K$ documents $z_i$. For final prediction $y$, we treat $z$ as a latent variable and marginalize over seq2seq predictions given different documents.

We build RAG models where the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We combine these components in a probabilistic model trained end-to-end (Fig. 1). The retriever (Dense Passage Retriever [26], henceforth DPR) provides latent documents conditioned on the input, and the seq2seq model (BART [32]) then conditions on these latent documents together with the input to generate the output. We marginalize the latent documents with a top-$K$ approximation, either on a per-output basis (assuming the same document is responsible for all tokens) or a per-token basis (where different documents are responsible for different tokens). Like T5 [51] or BART, RAG can be fine-tuned on any seq2seq task, whereby both the generator and retriever are jointly learned.

There has been extensive previous work proposing architectures to enrich systems with non-parametric memory which are trained from scratch for specific tasks, e.g. memory networks [64, 55], stack-augmented networks [25] and memory layers [30]. In contrast, we explore a setting where both parametric and non-parametric memory components are pre-trained and pre-loaded with extensive knowledge. Crucially, by using pre-trained access mechanisms, the ability to access knowledge is present without additional training.

Our results highlight the benefits of combining parametric and non-parametric memory with generation for *knowledge-intensive tasks*—tasks that humans could not reasonably be expected to perform without access to an external knowledge source. Our RAG models achieve state-of-the-art results on open Natural Questions [29], WebQuestions [3] and CuratedTrec [2] and strongly outperform recent approaches that use specialised pre-training objectives on TriviaQA [24]. Despite these being extractive tasks, we find that unconstrained generation outperforms previous extractive approaches. For knowledge-intensive generation, we experiment with MS-MARCO [1] and Jeopardy question generation, and we find that our models generate responses that are more factual, specific, and diverse than a BART baseline. For FEVER [56] fact verification, we achieve results within 4.3% of state-of-the-art pipeline models which use strong retrieval supervision. Finally, we demonstrate that the non-parametric memory can be replaced to update the models' knowledge as the world changes.

---

## 2. Methods

We explore RAG models, which use the input sequence $x$ to retrieve text documents $z$ and use them as additional context when generating the target sequence $y$. As shown in Figure 1, our models leverage two components: (i) a retriever $p_\eta(z|x)$ with parameters $\eta$ that returns (top-$K$ truncated) distributions over text passages given a query $x$ and (ii) a generator $p_\theta(y_i | x, z, y_{1:i-1})$ parametrized by $\theta$ that generates a current token based on a context of the previous $i-1$ tokens $y_{1:i-1}$, the original input $x$ and a retrieved passage $z$.

To train the retriever and generator end-to-end, we treat the retrieved document as a latent variable. We propose two models that marginalize over the latent documents in different ways to produce a distribution over generated text. In one approach, **RAG-Sequence**, the model uses the same document to predict each target token. The second approach, **RAG-Token**, can predict each target token based on a different document. In the following, we formally introduce both models and then describe the $p_\eta$ and $p_\theta$ components, as well as the training and decoding procedure.

### 2.1. Models

**RAG-Sequence Model.** The RAG-Sequence model uses the same retrieved document to generate the complete *sequence*. Technically, it treats the retrieved document as a single latent variable that is marginalized to get the seq2seq probability $p(y|x)$ via a top-$K$ approximation. Concretely, the top $K$ documents are retrieved using the retriever, and the generator produces the output sequence probability for each document, which are then marginalized:

$$p_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x) \, p_\theta(y|x, z) = \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x) \prod_i p_\theta(y_i | x, z, y_{1:i-1})$$

**RAG-Token Model.** In the RAG-Token model we can draw a different latent document for each target *token* and marginalize accordingly. This allows the generator to choose content from several documents when producing an answer. Concretely, the top $K$ documents are retrieved using the retriever, and then the generator produces a distribution for the next output token for each document, before marginalizing, and repeating the process with the following output token. Formally, we define:

$$p_{\text{RAG-Token}}(y|x) \approx \prod_i \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x) \, p_\theta(y_i | x, z, y_{1:i-1})$$

Finally, we note that RAG can be used for sequence classification tasks by considering the target class as a target sequence of length one, in which case RAG-Sequence and RAG-Token are equivalent.

### 2.2. Retriever: DPR

The retrieval component $p_\eta(z|x)$ is based on DPR [26]. DPR follows a bi-encoder architecture:

$$p_\eta(z|x) \propto \exp(d(z)^\top q(x)), \quad d(z) = \text{BERT}_d(z), \quad q(x) = \text{BERT}_q(x)$$

where $d(z)$ is a dense representation of a document produced by a $\text{BERT}_{\text{BASE}}$ document encoder [8], and $q(x)$ a query representation produced by a *query encoder*, also based on $\text{BERT}_{\text{BASE}}$. Calculating $\text{top-}k(p_\eta(\cdot|x))$, the list of $k$ documents $z$ with highest prior probability $p_\eta(z|x)$, is a Maximum Inner Product Search (MIPS) problem, which can be approximately solved in sub-linear time [23]. We use a pre-trained bi-encoder from DPR to initialize our retriever and to build the document index. This retriever was trained to retrieve documents which contain answers to TriviaQA [24] questions and Natural Questions [29]. We refer to the document index as the *non-parametric memory*.

### 2.3. Generator: BART

The generator component $p_\theta(y_i | x, z, y_{1:i-1})$ could be modelled using any encoder-decoder. We use BART-large [32], a pre-trained seq2seq transformer [58] with 400M parameters. To combine the input $x$ with the retrieved content $z$ when generating from BART, we simply concatenate them. BART was pre-trained using a denoising objective and a variety of different noising functions. It has obtained state-of-the-art results on a diverse set of generation tasks and outperforms comparably-sized T5 models [32]. We refer to the BART generator parameters $\theta$ as the *parametric memory* henceforth.

### 2.4. Training

We jointly train the retriever and generator components without any direct supervision on what document should be retrieved. Given a fine-tuning training corpus of input/output pairs $(x_j, y_j)$, we minimize the negative marginal log-likelihood of each target, $\sum_j -\log p(y_j | x_j)$ using stochastic gradient descent with Adam [28]. Updating the document encoder $\text{BERT}_d$ during training is costly as it requires the document index to be periodically updated as REALM does during pre-training [20]. We do not find this step necessary for strong performance, and keep the document encoder (and index) fixed, only fine-tuning the query encoder $\text{BERT}_q$ and the BART generator.

### 2.5. Decoding

At test time, RAG-Sequence and RAG-Token require different ways to approximate $\arg\max_y p(y|x)$.

**RAG-Token.** The RAG-Token model can be seen as a standard, autoregressive seq2seq generator with transition probability:

$$p'_\theta(y_i | x, y_{1:i-1}) = \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z_i|x) \, p_\theta(y_i | x, z_i, y_{1:i-1})$$

To decode, we can plug $p'_\theta(y_i | x, y_{1:i-1})$ into a standard beam decoder.

**RAG-Sequence.** For RAG-Sequence, the likelihood $p(y|x)$ does not break into a conventional per-token likelihood, hence we cannot solve it with a single beam search. Instead, we run beam search for each document $z$, scoring each hypothesis using $p_\theta(y_i | x, z, y_{1:i-1})$. This yields a set of hypotheses $\mathcal{Y}$, some of which may not have appeared in the beams of all documents. To estimate the probability of a hypothesis $y$ we run an additional forward pass for each document $z$ for which $y$ does not appear in the beam, multiply generator probability with $p_\eta(z|x)$ and then sum the probabilities across beams for the marginals. We refer to this decoding procedure as "**Thorough Decoding**." For longer output sequences, $|\mathcal{Y}|$ can become large, requiring many forward passes. For more efficient decoding, we can make a further approximation that $p_\theta(y | x, z_i) \approx 0$ where $y$ was not generated during beam search from $x, z_i$. This avoids the need to run additional forward passes once the candidate set $\mathcal{Y}$ has been generated. We refer to this decoding procedure as "**Fast Decoding**."

---

## 3. Experiments

We experiment with RAG in a wide range of knowledge-intensive tasks. For all experiments, we use a single Wikipedia dump for our non-parametric knowledge source. Following Lee et al. [31] and Karpukhin et al. [26], we use the December 2018 dump. Each Wikipedia article is split into disjoint 100-word chunks, to make a total of 21M documents. We use the document encoder to compute an embedding for each document, and build a single MIPS index using FAISS [23] with a Hierarchical Navigable Small World approximation for fast retrieval [37]. During training, we retrieve the top $k$ documents for each query. We consider $k \in \{5, 10\}$ for training and set $k$ for test time using dev data. We now discuss experimental details for each task.

### 3.1. Open-domain Question Answering

Open-domain question answering (QA) is an important real-world application and common testbed for knowledge-intensive tasks [20]. We treat questions and answers as input-output text pairs $(x, y)$ and train RAG by directly minimizing the negative log-likelihood of answers. We compare RAG to the popular extractive QA paradigm [5, 7, 31, 26], where answers are extracted spans from retrieved documents, relying primarily on non-parametric knowledge. We also compare to "Closed-Book QA" approaches [52], which, like RAG, generate answers, but which do not exploit retrieval, instead relying purely on parametric knowledge. We consider four popular open-domain QA datasets: Natural Questions (NQ) [29], TriviaQA (TQA) [24], WebQuestions (WQ) [3] and CuratedTrec (CT) [2]. As CT and WQ are small, we follow DPR [26] by initializing CT and WQ models with our NQ RAG model. We use the same train/dev/test splits as prior work [31, 26] and report Exact Match (EM) scores. For TQA, to compare with T5 [52], we also evaluate on the TQA Wiki test set.

### 3.2. Abstractive Question Answering

RAG models can go beyond simple extractive QA and answer questions with free-form, abstractive text generation. To test RAG's natural language generation (NLG) in a knowledge-intensive setting, we use the MS-MARCO NLG task v2.1 [43]. The task consists of questions, ten gold passages retrieved from a search engine for each question, and a full sentence answer annotated from the retrieved passages. We do not use the supplied passages, only the questions and answers, to treat MS-MARCO as an open-domain abstractive QA task. MS-MARCO has some questions that cannot be answered in a way that matches the reference answer without access to the gold passages, such as "What is the weather in Volcano, CA?" so performance will be lower without using gold passages. We also note that some MS-MARCO questions cannot be answered using Wikipedia alone. Here, RAG can rely on parametric knowledge to generate reasonable responses.

### 3.3. Jeopardy Question Generation

To evaluate RAG's generation abilities in a non-QA setting, we study open-domain question generation. Rather than use questions from standard open-domain QA tasks, which typically consist of short, simple questions, we propose the more demanding task of generating Jeopardy questions. Jeopardy is an unusual format that consists of trying to guess an entity from a fact about that entity. For example, "The World Cup" is the answer to the question "In 1986 Mexico scored as the first country to host this international sports competition twice." As Jeopardy questions are precise, factual statements, generating Jeopardy questions conditioned on their answer entities constitutes a challenging knowledge-intensive generation task.

We use the splits from SearchQA [10], with 100K train, 14K dev, and 27K test examples. As this is a new task, we train a BART model for comparison. Following [67], we evaluate using the SQuAD-tuned Q-BLEU-1 metric [42]. Q-BLEU is a variant of BLEU with a higher weight for matching entities and has higher correlation with human judgment for question generation than standard metrics. We also perform two human evaluations, one to assess generation factuality, and one for specificity. We define factuality as whether a statement can be corroborated by trusted external sources, and specificity as high mutual dependence between the input and output [33]. We follow best practice and use pairwise comparative evaluation [34]. Evaluators are shown an answer and two generated questions, one from BART and one from RAG. They are then asked to pick one of four options—question A is better, question B is better, both are good, or neither is good.

### 3.4. Fact Verification

FEVER [56] requires classifying whether a natural language claim is supported or refuted by Wikipedia, or whether there is not enough information to decide. The task requires retrieving evidence from Wikipedia relating to the claim and then reasoning over this evidence to classify whether the claim is true, false, or unverifiable from Wikipedia alone. FEVER is a retrieval problem coupled with a challenging entailment reasoning task. It also provides an appropriate testbed for exploring the RAG models' ability to handle classification rather than generation. We map FEVER class labels (supports, refutes, or not enough info) to single output tokens and directly train with claim-class pairs. Crucially, unlike most other approaches to FEVER, we do not use supervision on retrieved evidence. In many real-world applications, retrieval supervision signals aren't available, and models that do not require such supervision will be applicable to a wider range of tasks. We explore two variants: the standard 3-way classification task (supports/refutes/not enough info) and the 2-way (supports/refutes) task studied in Thorne and Vlachos [57]. In both cases we report label accuracy.

---

## 4. Results

### 4.1. Open-domain Question Answering

Table 1 shows results for RAG along with state-of-the-art models. On all four open-domain QA tasks, RAG sets a new state of the art (only on the T5-comparable split for TQA). RAG combines the generation flexibility of the "closed-book" (parametric only) approaches and the performance of "open-book" retrieval-based approaches. Unlike REALM and T5+SSM, RAG enjoys strong results without expensive, specialized "salient span masking" pre-training [20]. It is worth noting that RAG's retriever is initialized using DPR's retriever, which uses retrieval supervision on Natural Questions and TriviaQA. RAG compares favourably to the DPR QA system, which uses a BERT-based "cross-encoder" to re-rank documents, along with an extractive reader. RAG demonstrates that neither a re-ranker nor extractive reader is necessary for state-of-the-art performance.

There are several advantages to generating answers even when it is possible to extract them. Documents with clues about the answer but do not contain the answer verbatim can still contribute towards a correct answer being generated, which is not possible with standard extractive approaches, leading to more effective marginalization over documents. Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, where an extractive model would score 0%.

#### Table 1: Open-Domain QA Test Scores.

For TQA, left column uses the standard test set for Open-Domain QA, right column uses the TQA-Wiki test set. See Appendix D for further details.

| Model | NQ | TQA | WQ | CT |
|-------|:--:|:---:|:--:|:--:|
| **Closed Book** | | | | |
| T5-11B [52] | 34.5 | - / 50.1 | 37.4 | - |
| T5-11B+SSM [52] | 36.6 | - / 60.5 | 44.7 | - |
| **Open Book** | | | | |
| REALM [20] | 40.4 | - / - | 40.7 | 46.8 |
| DPR [26] | 41.5 | 57.9 / - | 41.1 | 50.6 |
| **RAG-Token** | **44.1** | **55.2 / 66.1** | **45.5** | 50.0 |
| **RAG-Seq.** | **44.5** | **56.8 / 68.0** | **45.2** | **52.2** |

### 4.2. Abstractive Question Answering

As shown in Table 2, RAG-Sequence outperforms BART on Open MS-MARCO NLG by 2.6 Bleu points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is impressive given that (i) those models access gold passages with specific information required to generate the reference answer, (ii) many questions are unanswerable without the gold passages, and (iii) not all questions are answerable from Wikipedia alone. Table 3 shows some generated answers from our models. Qualitatively, we find that RAG models hallucinate less and generate factually correct text more often than BART. Later, we also show that RAG generations are more diverse than BART generations (see §4.5).

#### Table 2: Generation and classification Test Scores.

MS-MARCO SotA is [4], FEVER-3 is [68] and FEVER-2 is [57]. *Uses gold context/evidence. Best model without gold access underlined.

| Model | Jeopardy B-1 | Jeopardy QB-1 | MSMARCO R-L | MSMARCO B-1 | FVR3 Label Acc. | FVR2 Label Acc. |
|-------|:-----------:|:------------:|:-----------:|:-----------:|:--------------:|:--------------:|
| SotA | - | - | 49.8* | 49.9* | 76.8 | 92.2* |
| BART | 15.1 | 19.7 | 38.2 | 41.6 | 64.0 | 81.1 |
| RAG-Tok. | 17.3 | 22.2 | 40.1 | 41.5 | 72.5 | 89.5 |
| RAG-Seq. | 14.7 | 21.4 | **40.8** | **44.2** | | |

### 4.3. Jeopardy Question Generation

Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. Table 4 shows human evaluation results, over 452 pairs of generations from BART and RAG-Token. Evaluators indicated that BART was more factual than RAG in only 7.1% of cases, while RAG was more factual in 42.7% of cases, and both RAG and BART were factual in a further 17% of cases, clearly demonstrating the effectiveness of RAG on the task over a state-of-the-art generation model. Evaluators also find RAG generations to be more specific by a large margin. Table 3 shows typical generations from each model.

#### Table 4: Human assessments for the Jeopardy Question Generation Task.

| | Factuality | Specificity |
|---|:----------:|:-----------:|
| BART better | 7.1% | 16.8% |
| RAG better | 42.7% | 37.4% |
| Both good | 11.7% | 11.8% |
| Both poor | 17.7% | 6.9% |
| No majority | 20.8% | 20.1% |

Jeopardy questions often contain two separate pieces of information, and RAG-Token may perform best because it can generate responses that combine content from several documents. Figure 2 shows an example. When generating "Sun", the posterior is high for document 2 which mentions "The Sun Also Rises". Similarly, document 1 dominates the posterior when "A Farewell to Arms" is generated. Intriguingly, after the first token of each book is generated, the document posterior flattens. This observation suggests that the generator can complete the titles without depending on specific documents. In other words, the model's parametric knowledge is sufficient to complete the titles. We find evidence for this hypothesis by feeding the BART-only baseline with the partial decoding "The Sun". BART completes the generation *"The Sun Also Rises" is a novel by this author of "The Sun Also Rises"* indicating the title "The Sun Also Rises" is stored in BART's parameters. Similarly, BART will complete the partial decoding *"The Sun Also Rises" is a novel by this author of "A*" with *"The Sun Also Rises" is a novel by this author of "A Farewell to Arms"*. This example shows how parametric and non-parametric memories *work together*—the non-parametric component helps to guide the generation, drawing out specific knowledge stored in the parametric memory.

> **Figure 2.** RAG-Token document posterior $p(z_i | x, y_i, y_{-i})$ for each generated token for input "Hemingway" for Jeopardy generation with 5 retrieved documents. The posterior for document 1 is high when generating "A Farewell to Arms" and for document 2 when generating "The Sun Also Rises".

#### Table 3: Examples from generation tasks.

RAG models generate more specific and factually accurate responses. '?' indicates factually incorrect responses, * indicates partially correct responses.

| Task | Input | Model | Generation |
|------|-------|-------|-----------|
| MS-MARCO | define middle ear | BART | ? The middle ear is the part of the ear between the middle ear and the nose. |
| | | RAG-T | The middle ear is the portion of the ear internal to the eardrum. |
| | | RAG-S | The middle ear includes the tympanic cavity and the three ossicles. |
| | what currency needed in scotland | BART | The currency needed in Scotland is Pound sterling. |
| | | RAG-T | Pound is the currency needed in Scotland. |
| | | RAG-S | The currency needed in Scotland is the pound sterling. |
| Jeopardy QGen | Washington | BART | ? This state has the largest number of counties in the U.S. |
| | | RAG-T | It's the only U.S. state named for a U.S. president |
| | | RAG-S | It's the state where you'll find Mount Rainier National Park |
| | The Divine Comedy | BART | * This epic poem by Dante is divided into 3 parts: the Inferno, the Purgatorio & the Purgatorio |
| | | RAG-T | Dante's "Inferno" is the first part of this epic poem |
| | | RAG-S | This 14th century work is divided into 3 sections: "Inferno", "Purgatorio" & "Paradiso" |

### 4.4. Fact Verification

Table 2 shows our results on FEVER. For 3-way classification, RAG scores are within 4.3% of state-of-the-art models, which are complex pipeline systems with domain-specific architectures and substantial engineering, trained using intermediate retrieval supervision, which RAG does not require.

For 2-way classification, we compare against Thorne and Vlachos [57], who train RoBERTa [35] to classify the claim as true or false given the gold evidence sentence. RAG achieves an accuracy within 2.7% of this model, despite being supplied with only the claim and retrieving its own evidence. We also analyze whether documents retrieved by RAG correspond to documents annotated as gold evidence in FEVER. We calculate the overlap in article titles between the top $k$ documents retrieved by RAG and gold evidence annotations. We find that the top retrieved document is from a gold article in 71% of cases, and a gold article is present in the top 10 retrieved articles in 90% of cases.

### 4.5. Additional Results

**Generation Diversity.** Section 4.3 shows that RAG models are more factual and specific than BART for Jeopardy question generation. Following recent work on diversity-promoting decoding [33, 59, 39], we also investigate generation diversity by calculating the ratio of distinct n-grams to total n-grams generated by different models. Table 5 shows that RAG-Sequence's generations are more diverse than RAG-Token's, and both are significantly more diverse than BART without needing any diversity-promoting decoding.

#### Table 5: Ratio of distinct to total tri-grams for generation tasks.

| | MSMARCO | Jeopardy QGen |
|---|:-------:|:-------------:|
| Gold | 89.6% | 90.0% |
| BART | 70.7% | 32.4% |
| RAG-Token | 77.8% | 46.8% |
| RAG-Seq. | 83.5% | 53.8% |

**Retrieval Ablations.** A key feature of RAG is learning to retrieve relevant information for the task. To assess the effectiveness of the retrieval mechanism, we run ablations where we freeze the retriever during training. As shown in Table 6, learned retrieval improves results for all tasks.

#### Table 6: Ablations on the dev set.

As FEVER is a classification task, both RAG models are equivalent.

| Model | NQ EM | TQA EM | WQ EM | CT EM | Jeopardy B-1 | Jeopardy QB-1 | MSMarco R-L | MSMarco B-1 | FVR-3 Label Acc. | FVR-2 Label Acc. |
|-------|:-----:|:------:|:-----:|:-----:|:-----------:|:------------:|:-----------:|:-----------:|:--------------:|:--------------:|
| RAG-Token-BM25 | 29.7 | 41.5 | 32.1 | 33.1 | 17.5 | 22.3 | 55.5 | 48.4 | 75.1 | 91.6 |
| RAG-Sequence-BM25 | 31.8 | 44.1 | 36.6 | 33.8 | 11.1 | 19.5 | 56.5 | 46.9 | | |
| RAG-Token-Frozen | 37.8 | 50.1 | 37.1 | 51.1 | 16.7 | 21.7 | 55.9 | 49.4 | 72.9 | 89.4 |
| RAG-Sequence-Frozen | 41.2 | 52.1 | 41.8 | 52.6 | 11.8 | 19.6 | 56.7 | 47.3 | | |
| RAG-Token | 43.5 | 54.8 | 46.5 | 51.9 | 17.9 | 22.6 | 56.2 | 49.4 | 74.5 | 90.6 |
| RAG-Sequence | 44.0 | 55.8 | 44.9 | 53.4 | 15.3 | 21.5 | 57.2 | 47.5 | | |

We compare RAG's dense retriever to a word overlap-based BM25 retriever [53]. Here, we replace RAG's retriever with a fixed BM25 system, and use BM25 retrieval scores as logits when calculating $p(z|x)$. Table 6 shows the results. For FEVER, BM25 performs best, perhaps since FEVER claims are heavily entity-centric and thus well-suited for word overlap-based retrieval. Differentiable retrieval improves results on all other tasks, especially for Open-Domain QA, where it is crucial.

**Index hot-swapping.** An advantage of non-parametric memory models like RAG is that knowledge can be easily updated at test time. Parametric-only models like T5 or BART need further training to update their behavior as the world changes. To demonstrate, we build an index using the DrQA [5] Wikipedia dump from December 2016 and compare outputs from RAG using this index to the newer index from our main results (December 2018). We prepare a list of 82 world leaders who had changed between these dates and use a template "Who is {position}?" (e.g. "Who is the President of Peru?") to query our NQ RAG model with each index. RAG answers 70% correctly using the 2016 index for 2016 world leaders and 68% using the 2018 index for 2018 world leaders. Accuracy with mismatched indices is low (12% with the 2018 index and 2016 leaders, 4% with the 2016 index and 2018 leaders). This shows we can update RAG's world knowledge by simply replacing its non-parametric memory.

**Effect of Retrieving more documents.** Models are trained with either 5 or 10 retrieved latent documents, and we do not observe significant differences in performance between them. We have the flexibility to adjust the number of retrieved documents at test time, which can affect performance and runtime. Figure 3 (left) shows that retrieving more documents at test time monotonically improves Open-domain QA results for RAG-Sequence, but performance peaks for RAG-Token at 10 retrieved documents. Figure 3 (right) shows that retrieving more documents leads to higher Rouge-L for RAG-Token at the expense of Bleu-1, but the effect is less pronounced for RAG-Sequence.

> **Figure 3.** Left: NQ performance as more documents are retrieved (X-axis: K Retrieved Docs 10–50, Y-axis: NQ Exact Match 39–44. Curves: RAG-Tok, RAG-Seq). Center: Retrieval recall performance in NQ (X-axis: K Retrieved Docs 10–50, Y-axis: NQ Answer Recall @ K 40–80. Curves: RAG-Tok, RAG-Seq, Fixed DPR, BM25). Right: MS-MARCO Bleu-1 and Rouge-L as more documents are retrieved (X-axis: K Retrieved Docs 10–50, Y-axis: Bleu-1 / Rouge-L score 48–56. Curves: RAG-Tok R-L, RAG-Tok B-1, RAG-Seq R-L, RAG-Seq B-1).

---

## 5. Related Work

**Single-Task Retrieval.** Prior work has shown that retrieval improves performance across a variety of NLP tasks when considered in isolation. Such tasks include open-domain question answering [5, 29], fact checking [56], fact completion [48], long-form question answering [12], Wikipedia article generation [36], dialogue [41, 65, 9, 13], translation [17], and language modeling [19, 27]. Our work unifies previous successes in incorporating retrieval into individual tasks, showing that a single retrieval-based architecture is capable of achieving strong performance across several tasks.

**General-Purpose Architectures for NLP.** Prior work on general-purpose architectures for NLP tasks has shown great success without the use of retrieval. A single, pre-trained language model has been shown to achieve strong performance on various classification tasks in the GLUE benchmarks [60, 61] after fine-tuning [49, 8]. GPT-2 [50] later showed that a single, left-to-right, pre-trained language model could achieve strong performance across both discriminative and generative tasks. For further improvement, BART [32] and T5 [51, 52] propose a single, pre-trained encoder-decoder model that leverages bi-directional attention to achieve stronger performance on discriminative and generative tasks. Our work aims to expand the space of possible tasks with a single, unified architecture, by learning a retrieval module to augment pre-trained, generative language models.

**Learned Retrieval.** There is significant work on learning to retrieve documents in information retrieval, more recently with pre-trained, neural language models [44, 26] similar to ours. Some work optimizes the retrieval module to aid in a specific, downstream task such as question answering, using search [46], reinforcement learning [6, 63, 62], or a latent variable approach [31, 20] as in our work. These successes leverage different retrieval-based architectures and optimization techniques to achieve strong performance on a single task, while we show that a single retrieval-based architecture can be fine-tuned for strong performance on a variety of tasks.

**Memory-based Architectures.** Our document index can be seen as a large external memory for neural networks to attend to, analogous to memory networks [64, 55]. Concurrent work [14] learns to retrieve a trained embedding for each entity in the input, rather than to retrieve raw text as in our work. Other work improves the ability of dialog models to generate factual text by attending over fact embeddings [15, 13]. A key feature of our memory is that it is comprised of raw text rather than distributed representations, which makes the memory both (i) human-readable, lending a form of interpretability to our model, and (ii) human-writable, enabling us to dynamically update the model's memory by editing the document index. This approach has also been used in knowledge-intensive dialog, where generators have been conditioned on retrieved text directly, albeit obtained via TF-IDF rather than end-to-end learnt retrieval [9].

**Retrieve-and-Edit approaches.** Our method shares some similarities with retrieve-and-edit style approaches, where a similar training input-output pair is retrieved for a given input, and then edited to provide a final output. These approaches have proved successful in a number of domains including Machine Translation [18, 22] and Semantic Parsing [21]. Our approach does have several differences, including less emphasis on lightly editing a retrieved item, but on aggregating content from several pieces of retrieved content, as well as learning latent retrieval, and retrieving evidence documents rather than related training pairs. This said, RAG techniques may work well in these settings, and could represent promising future work.

---

## 6. Discussion

In this work, we presented hybrid generation models with access to parametric and non-parametric memory. We showed that our RAG models obtain state of the art results on open-domain QA. We found that people prefer RAG's generation over purely parametric BART, finding RAG more factual and specific. We conducted a thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated how the retrieval index can be hot-swapped to update the model without requiring any retraining. In future work, it may be fruitful to investigate if the two components can be jointly pre-trained from scratch, either with a denoising objective similar to BART or some another objective. Our work opens up new research directions on how parametric and non-parametric memories interact and how to most effectively combine them, showing promise in being applied to a wide variety of NLP tasks.

---

## Broader Impact

This work offers several positive societal benefits over previous work: the fact that it is more strongly grounded in real factual knowledge (in this case Wikipedia) makes it "hallucinate" less with generations that are more factual, and offers more control and interpretability. RAG could be employed in a wide variety of scenarios with direct benefit to society, for example by endowing it with a medical index and asking it open-domain questions on that topic, or by helping people be more effective at their jobs.

With these advantages also come potential downsides: Wikipedia, or any potential external knowledge source, will probably never be entirely factual and completely devoid of bias. Since RAG can be employed as a language model, similar concerns as for GPT-2 [50] are valid here, although arguably to a lesser extent, including that it might be used to generate abuse, faked or misleading content in the news or on social media; to impersonate others; or to automate the production of spam/phishing content [54]. Advanced language models may also lead to the automation of various jobs in the coming decades [16]. In order to mitigate these risks, AI systems could be employed to fight against misleading content and automated spam/phishing.

---

## Acknowledgments

The authors would like to thank the reviewers for their thoughtful and constructive feedback on this paper, as well as HuggingFace for their help in open-sourcing code to run RAG models. The authors would also like to thank Kyunghyun Cho and Sewon Min for productive discussions and advice. EP thanks supports from the NSF Graduate Research Fellowship. PL is supported by the FAIR PhD program.

---

## References

[1] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv:1611.09268 [cs], November 2016.

[2] Petr Baudiš and Jan Šedivý. Modeling of the question answering task in the yodaqa system. In *International Conference of the Cross-Language Evaluation Forum for European Languages*, pages 222–228. Springer, 2015.

[3] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic Parsing on Freebase from Question-Answer Pairs. In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pages 1533–1544, Seattle, Washington, USA, October 2013. Association for Computational Linguistics.

[4] Bin Bi, Chenliang Li, Chen Wu, Ming Yan, and Wei Wang. PALM: Pre-training an autoencoding & autoregressive language model for context-conditioned generation. ArXiv, abs/2004.07159, 2020.

[5] Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading Wikipedia to Answer Open-Domain Questions. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1870–1879, Vancouver, Canada, July 2017.

[6] Eunsol Choi, Daniel Hewlett, Jakob Uszkoreit, Illia Polosukhin, Alexandre Lacoste, and Jonathan Berant. Coarse-to-fine question answering for long documents. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 209–220, Vancouver, Canada, July 2017.

[7] Christopher Clark and Matt Gardner. Simple and Effective Multi-Paragraph Reading Comprehension. arXiv:1710.10723 [cs], October 2017.

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota, June 2019.

[9] Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. Wizard of Wikipedia: Knowledge-powered conversational agents. In *International Conference on Learning Representations*, 2019.

[10] Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and Kyunghyun Cho. SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine. arXiv:1704.05179 [cs], April 2017.

[11] Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 889–898, Melbourne, Australia, July 2018.

[12] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. ELI5: Long form question answering. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 3558–3567, Florence, Italy, July 2019.

[13] Angela Fan, Claire Gardent, Chloe Braud, and Antoine Bordes. Augmenting transformers with KNN-based composite memory, 2020.

[14] Thibault Févry, Livio Baldini Soares, Nicholas FitzGerald, Eunsol Choi, and Tom Kwiatkowski. Entities as experts: Sparse memory access with entity supervision. ArXiv, abs/2004.07202, 2020.

[15] Marjan Ghazvininejad, Chris Brockett, Ming-Wei Chang, Bill Dolan, Jianfeng Gao, Wen tau Yih, and Michel Galley. A knowledge-grounded neural conversation model. In *AAAI Conference on Artificial Intelligence*, 2018.

[16] Katja Grace, John Salvatier, Allan Dafoe, Baobao Zhang, and Owain Evans. When will AI exceed human performance? evidence from AI experts. CoRR, abs/1705.08807, 2017.

[17] Jiatao Gu, Yong Wang, Kyunghyun Cho, and Victor O.K. Li. Search engine guided neural machine translation. In *AAAI Conference on Artificial Intelligence*, 2018.

[18] Jiatao Gu, Yong Wang, Kyunghyun Cho, and Victor O.K. Li. Search engine guided neural machine translation. In *32nd AAAI Conference on Artificial Intelligence, AAAI 2018*, pages 5133–5140. AAAI press, 2018.

[19] Kelvin Guu, Tatsunori B. Hashimoto, Yonatan Oren, and Percy Liang. Generating sentences by editing prototypes. *Transactions of the Association for Computational Linguistics*, 6:437–450, 2018.

[20] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-augmented language model pre-training. ArXiv, abs/2002.08909, 2020.

[21] Tatsunori B Hashimoto, Kelvin Guu, Yonatan Oren, and Percy S Liang. A retrieve-and-edit framework for predicting structured outputs. In *Advances in Neural Information Processing Systems 31*, pages 10052–10062. Curran Associates, Inc., 2018.

[22] Nabil Hossain, Marjan Ghazvininejad, and Luke Zettlemoyer. Simple and effective retrieve-edit-rerank text generation. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 2532–2538, Online, July 2020.

[23] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. arXiv preprint arXiv:1702.08734, 2017.

[24] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, Vancouver, Canada, July 2017.

[25] Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented recurrent nets. In *NIPS'15*, page 190–198, Cambridge, MA, USA, 2015. MIT Press.

[26] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906, 2020.

[27] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. In *International Conference on Learning Representations*, 2020.

[28] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *3rd International Conference on Learning Representations, ICLR 2015*, San Diego, CA, USA, May 7-9, 2015.

[29] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural Questions: a Benchmark for Question Answering Research. *Transactions of the Association of Computational Linguistics*, 2019.

[30] Guillaume Lample, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer, and Herve Jegou. Large memory layers with product keys. In *Advances in Neural Information Processing Systems 32*, pages 8548–8559. Curran Associates, Inc., 2019.

[31] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 6086–6096, Florence, Italy, July 2019.

[32] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.

[33] Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. A diversity-promoting objective function for neural conversation models. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 110–119, San Diego, California, June 2016.

[34] Margaret Li, Jason Weston, and Stephen Roller. Acute-eval: Improved dialogue evaluation with optimized questions and multi-turn comparisons. ArXiv, abs/1909.03087, 2019.

[35] Hairong Liu, Mingbo Ma, Liang Huang, Hao Xiong, and Zhongjun He. Robust neural machine translation with joint textual and phonetic embedding. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 3044–3049, Florence, Italy, July 2019.

[36] Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. Generating Wikipedia by summarizing long sequences. In *International Conference on Learning Representations*, 2018.

[37] Yury A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42:824–836, 2016.

[38] Gary Marcus. The next decade in ai: four steps towards robust artificial intelligence. arXiv preprint arXiv:2002.06177, 2020.

[39] Luca Massarelli, Fabio Petroni, Aleksandra Piktus, Myle Ott, Tim Rocktäschel, Vassilis Plachouras, Fabrizio Silvestri, and Sebastian Riedel. How decoding strategies affect the verifiability of generated text. arXiv preprint arXiv:1911.03587, 2019.

[40] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training. In *ICLR*, 2018.

[41] Nikita Moghe, Siddhartha Arora, Suman Banerjee, and Mitesh M. Khapra. Towards exploiting background knowledge for building conversation systems. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2322–2332, Brussels, Belgium, October-November 2018.

[42] Preksha Nema and Mitesh M. Khapra. Towards a better metric for evaluating question generation systems. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 3950–3959, Brussels, Belgium, October-November 2018.

[43] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. MS MARCO: A human generated machine reading comprehension dataset. In *Proceedings of the Workshop on Cognitive Computation: Integrating neural and symbolic approaches 2016 co-located with the 30th Annual Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain, December 9, 2016*, volume 1773 of CEUR Workshop Proceedings. CEUR-WS.org, 2016.

[44] Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085, 2019.

[45] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)*, pages 48–53, Minneapolis, Minnesota, June 2019.

[46] Ethan Perez, Siddharth Karamcheti, Rob Fergus, Jason Weston, Douwe Kiela, and Kyunghyun Cho. Finding generalizable evidence by learning to convince Q&A models. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 2402–2411, Hong Kong, China, November 2019.

[47] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. Language models as knowledge bases? In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 2463–2473, Hong Kong, China, November 2019.

[48] Fabio Petroni, Patrick Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel. How context affects language models' factual predictions. In *Automated Knowledge Base Construction*, 2020.

[49] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving Language Understanding by Generative Pre-Training, 2018.

[50] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019.

[51] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019.

[52] Adam Roberts, Colin Raffel, and Noam Shazeer. How much knowledge can you pack into the parameters of a language model? arXiv e-prints, 2020.

[53] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond. *Found. Trends Inf. Retr.*, 3(4):333–389, April 2009.

[54] Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herbert-Voss, Jeff Wu, Alec Radford, and Jian-Bing Wang. Release strategies and the social impacts of language models. ArXiv, abs/1908.09203, 2019.

[55] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In *Advances in Neural Information Processing Systems 28*, pages 2440–2448. Curran Associates, Inc., 2015.

[56] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERification. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 809–819, New Orleans, Louisiana, June 2018.

[57] James H. Thorne and Andreas Vlachos. Avoiding catastrophic forgetting in mitigating model biases in sentence-pair classification with elastic weight consolidation. ArXiv, abs/2004.14366, 2020.

[58] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems 30*, pages 5998–6008. Curran Associates, Inc., 2017.

[59] Ashwin Vijayakumar, Michael Cogswell, Ramprasaath Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. Diverse beam search for improved description of complex scenes. *AAAI Conference on Artificial Intelligence*, 2018.

[60] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In *Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*, pages 353–355, Brussels, Belgium, November 2018.

[61] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems. In *Advances in Neural Information Processing Systems 32*, pages 3261–3275. Curran Associates, Inc., 2019.

[62] Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang, Tim Klinger, Wei Zhang, Shiyu Chang, Gerry Tesauro, Bowen Zhou, and Jing Jiang. R³: Reinforced ranker-reader for open-domain question answering. In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)*, pages 5981–5988. AAAI Press, 2018.

[63] Shuohang Wang, Mo Yu, Jing Jiang, Wei Zhang, Xiaoxiao Guo, Shiyu Chang, Zhiguo Wang, Tim Klinger, Gerald Tesauro, and Murray Campbell. Evidence aggregation for answer re-ranking in open-domain question answering. In *ICLR*, 2018.

[64] Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. In *3rd International Conference on Learning Representations, ICLR 2015*, San Diego, CA, USA, May 7-9, 2015.

[65] Jason Weston, Emily Dinan, and Alexander Miller. Retrieve and refine: Improved sequence generation models for dialogue. In *Proceedings of the 2018 EMNLP Workshop SCAI: The 2nd International Workshop on Search-Oriented Conversational AI*, pages 87–92, Brussels, Belgium, October 2018.

[66] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface's transformers: State-of-the-art natural language processing. ArXiv, abs/1910.03771, 2019.

[67] Shiyue Zhang and Mohit Bansal. Addressing semantic drift in question generation for semi-supervised question answering. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 2495–2509, Hong Kong, China, November 2019.

[68] Wanjun Zhong, Jingjing Xu, Duyu Tang, Zenan Xu, Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin. Reasoning over semantic-level graph for fact checking. ArXiv, abs/1909.03745, 2019.

---

## Appendices for Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

### A. Implementation Details

For Open-domain QA we report test numbers using 15 retrieved documents for RAG-Token models. For RAG-Sequence models, we report test results using 50 retrieved documents, and we use the Thorough Decoding approach since answers are generally short. We use greedy decoding for QA as we did not find beam search improved results. For Open-MSMarco and Jeopardy question generation, we report test numbers using ten retrieved documents for both RAG-Token and RAG-Sequence, and we also train a BART-large model as a baseline. We use a beam size of four, and use the Fast Decoding approach for RAG-Sequence models, as Thorough Decoding did not improve performance.

### B. Human Evaluation

> **Figure 4.** Annotation interface for human evaluation of factuality. A pop-out for detailed instructions and a worked example appear when clicking "view tool guide".

Figure 4 shows the user interface for human evaluation. To avoid any biases for screen position, which model corresponded to sentence A and sentence B was randomly selected for each example. Annotators were encouraged to research the topic using the internet, and were given detailed instructions and worked examples in a full instructions tab. We included some gold sentences in order to assess the accuracy of the annotators. Two annotators did not perform well on these examples and their annotations were removed from the results.

### C. Training Setup Details

We train all RAG models and BART baselines using Fairseq [45].¹ We train with mixed precision floating point arithmetic [40], distributing training across 8, 32GB NVIDIA V100 GPUs, though training and inference can be run on one GPU. We find that doing Maximum Inner Product Search with FAISS is sufficiently fast on CPU, so we store document index vectors on CPU, requiring ∼100 GB of CPU memory for all of Wikipedia. After submission, we have ported our code to HuggingFace Transformers [66]², which achieves equivalent performance to the previous version but is a cleaner and easier to use implementation. This version is also open-sourced. We also compress the document index using FAISS's compression tools, reducing the CPU memory requirement to 36GB. Scripts to run experiments with RAG can be found at [https://github.com/huggingface/transformers/blob/master/examples/rag/README.md](https://github.com/huggingface/transformers/blob/master/examples/rag/README.md) and an interactive demo of a RAG model can be found at [https://huggingface.co/rag/](https://huggingface.co/rag/).

### D. Further Details on Open-Domain QA

For open-domain QA, multiple answer annotations are often available for a given question. These answer annotations are exploited by extractive models during training as typically all the answer annotations are used to find matches within documents when preparing training data. For RAG, we also make use of multiple annotation examples for Natural Questions and WebQuestions by training the model with each $(q, a)$ pair separately, leading to a small increase in accuracy. For TriviaQA, there are often many valid answers to a given question, some of which are not suitable training targets, such as emoji or spelling variants. For TriviaQA, we filter out answer candidates if they do not occur in top 1000 documents for the query.

**CuratedTrec preprocessing.** The answers for CuratedTrec are given in the form of regular expressions, which has been suggested as a reason why it is unsuitable for answer-generation models [20]. To overcome this, we use a pre-processing step where we first retrieve the top 1000 documents for each query, and use the answer that most frequently matches the regex pattern as the supervision target. If no matches are found, we resort to a simple heuristic: generate all possible permutations for each regex, replacing non-deterministic symbols in the regex nested tree structure with a whitespace.

**TriviaQA Evaluation setups.** The open-domain QA community customarily uses public development datasets as test datasets, as test data for QA datasets is often restricted and dedicated to reading comprehension purposes. We report our results using the datasets splits used in DPR [26], which are consistent with common practice in Open-domain QA. For TriviaQA, this test dataset is the public TriviaQA Web Development split. Roberts et al. [52] used the TriviaQA official Wikipedia test set instead. Févry et al. [14] follow this convention in order to compare with Roberts et al. [52] (See appendix of [14]). We report results on both test sets to enable fair comparison to both approaches. We find that our performance is much higher using the official Wiki test set, rather than the more conventional open-domain test set, which we attribute to the official Wiki test set questions being simpler to answer from Wikipedia.

### E. Further Details on FEVER

For FEVER classification, we follow the practice from [32], and first re-generate the claim, and then classify using the representation of the final hidden state, before finally marginalizing across documents to obtain the class probabilities. The FEVER task traditionally has two sub-tasks. The first is to classify the claim as either "Supported", "Refuted" or "Not Enough Info", which is the task we explore in the main paper. FEVER's other sub-task involves extracting sentences from Wikipedia as evidence supporting the classification prediction. As FEVER uses a different Wikipedia dump to us, directly tackling this task is not straightforward. We hope to address this in future work.

### F. Null Document Probabilities

We experimented with adding "Null document" mechanism to RAG, similar to REALM [20] in order to model cases where no useful information could be retrieved for a given input. Here, if $k$ documents were retrieved, we would additionally "retrieve" an empty document and predict a logit for the null document, before marginalizing over $k + 1$ predictions. We explored modelling this null document logit by learning (i) a document embedding for the null document, (ii) a static learnt bias term, or (iii) a neural network to predict the logit. We did not find that these improved performance, so in the interests of simplicity, we omit them. For Open MS-MARCO, where useful retrieved documents cannot always be retrieved, we observe that the model learns to always retrieve a particular set of documents for questions that are less likely to benefit from retrieval, suggesting that null document mechanisms may not be necessary for RAG.

### G. Parameters

Our RAG models contain the trainable parameters for the BERT-base query and document encoder of DPR, with 110M parameters each (although we do not train the document encoder ourselves) and 406M trainable parameters from BART-large, making a total of 626M trainable parameters. The best performing "closed-book" (parametric only) open-domain QA model is T5-11B with 11 Billion trainable parameters. The T5 model with the closest number of parameters to our models is T5-large (770M parameters), which achieves a score of 28.9 EM on Natural Questions [52], substantially below the 44.5 that RAG-Sequence achieves, indicating that hybrid parametric/non-parametric models require far fewer trainable parameters for strong open-domain QA performance. The non-parametric memory index does not consist of trainable parameters, but does consist of 21M 728 dimensional vectors, consisting of 15.3B values. These can easily be stored at 8-bit floating point precision to manage memory and disk footprints.

#### Table 7: Number of instances in the datasets used.

\*A hidden subset of this data is used for evaluation.

| Task | Train | Development | Test |
|------|:-----:|:-----------:|:----:|
| Natural Questions | 79,169 | 8,758 | 3,611 |
| TriviaQA | 78,786 | 8,838 | 11,314 |
| WebQuestions | 3,418 | 362 | 2,033 |
| CuratedTrec | 635 | 134 | 635 |
| Jeopardy Question Generation | 97,392 | 13,714 | 26,849 |
| MS-MARCO | 153,726 | 12,468 | 101,093\* |
| FEVER-3-way | 145,450 | 10,000 | 10,000 |
| FEVER-2-way | 96,966 | 6,666 | 6,666 |

### H. Retrieval Collapse

In preliminary experiments, we observed that for some tasks such as story generation [11], the retrieval component would "collapse" and learn to retrieve the same documents regardless of the input. In these cases, once retrieval had collapsed, the generator would learn to ignore the documents, and the RAG model would perform equivalently to BART. The collapse could be due to a less-explicit requirement for factual knowledge in some tasks, or the longer target sequences, which could result in less informative gradients for the retriever. Perez et al. [46] also found spurious retrieval results when optimizing a retrieval component in order to improve performance on downstream tasks.

### I. Number of Instances per Dataset

The number of training, development and test data points in each of our datasets is shown in Table 7.

---

## Footnotes

1. Code to run experiments with RAG has been open-sourced as part of the HuggingFace Transformers Library [66] and can be found at [https://github.com/huggingface/transformers/blob/master/examples/rag/](https://github.com/huggingface/transformers/blob/master/examples/rag/). An interactive demo of RAG models can be found at [https://huggingface.co/rag/](https://huggingface.co/rag/).
2. https://github.com/pytorch/fairseq
3. https://github.com/huggingface/transformers
