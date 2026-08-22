# LLM Multi-Agent Systems: Challenges and Open Problems

**Shanshan Han¹, Qifan Zhang¹, Weizhao Jin², Zhaozhuo Xu³**

¹University of California, Irvine, CA, USA
²University of Southern California, Los Angeles, CA, USA
³Stevens Institute of Technology, Hoboken, NJ, USA

Correspondence to: Shanshan Han <shanshan.han@uci.edu>

arXiv:2402.03578v3 [cs.MA] 28 Jan 2026

---

## Abstract

This paper explores multi-agent systems and identifies challenges that remain inadequately addressed. By leveraging the diverse capabilities and roles of individual agents, multi-agent systems can tackle complex tasks through agent collaboration. We discuss optimizing task allocation, fostering robust reasoning through iterative debates, managing complex and layered context information, and enhancing memory management to support the intricate interactions within multi-agent systems. We also explore potential applications of multi-agent systems in blockchain systems to shed light on their future development and application in real-world distributed systems.

## 1. Introduction

Multi-agent systems enhance the capabilities of single LLM agents by leveraging collaborations among agents and their specialized abilities (Talebirad & Nadiri, 2023; Zhang et al., 2023a; Park et al., 2023; Li et al., 2023; Jinxin et al., 2023). It utilizes collaboration and coordination among agents to execute tasks that are beyond the capability of any individual agent. In multi-agent systems, each agent is equipped with distinctive capabilities and roles, collaborating towards the fulfillment of some common objectives. Such collaboration, characterized by activities such as debate and reflection, has proven particularly effective for tasks requiring deep thought and innovation. Recent works include simulating interactive environments (Park et al., 2023; Jinxin et al., 2023), role-playing (Li et al., 2023), and reasoning (Du et al., 2023; Liang et al., 2023), demonstrating the huge potential of multi-agent systems in handling complex real-world scenarios.

While existing works have demonstrated the impressive capabilities of multi-agent systems, the potential for advanced multi-agent systems far exceeds the progress made to date. A large number of existing works focus on devising planning strategies within a single agent by breaking down the tasks into smaller, more manageable tasks (Chen et al., 2022; Ziqi & Lu, 2023; Yao et al., 2023; Long, 2023; Besta et al., 2023; Wang et al., 2022b). Yet, multi-agent systems involve agents of various specializations and more complex interactions and layered context information, which poses challenges to the designing of the workflow as well as the whole system. Also, existing literature pays limited attention to memory storage, while memory plays a critical role in collaborations between agents. It enables agents to access some common sense, align context with their tasks, and, further, learn from past workflows and adapt their strategies accordingly.

To date, multiple significant challenges that differentiate multi-agent systems and single-agent systems remain inadequately addressed. We summarize them as follows:

- Optimizing task allocation to leverage agents' unique skills and specializations.
- Fostering robust reasoning through iterative debates or discussions among a subset of agents to enhance intermediate results.
- Managing complex and layered context information, such as context for overall tasks, single agents, and some common knowledge between agents, while ensuring alignment to the general objective.
- Managing various types of memory that serve different objectives coherent with the interactions in multi-agent systems.

This paper explores multi-agent systems, offering a survey of the existing works while shedding light on the challenges and open problems in it. We study major components in multi-agent systems, including planning and memory storage, and address unique challenges posed by multi-agent systems, compared with single-agent systems. We also explore potential applications of multi-agent systems in blockchain systems from two perspectives, including 1) utilizing multi-agent systems as tools, and 2) assigning an agent to each blockchain node to make it represent the user, such that the agent can complete some tasks on behalf of the user in the blockchain network.

## 2. Overview

### 2.1. Structure of Multi-agent Systems

The structure of multi-agent systems can be categorized into various types, based on each agent's functionality and their interactions.

**Equi-Level Structure.** LLM agents in an equi-level system operate at the same hierarchical level, where each agent has its own role and strategy, but neither holds a hierarchical advantage over the other, e.g., DMAS (Chen et al., 2023); see Figure 1(a). The agents in such systems can have same, neutral, or opposing objectives. Agents with the same goals collaborate towards a common goal without centralized leadership. The emphasis is on collective decision-making and shared responsibilities (Li et al., 2019). With opposing objectives, the agents negotiate or debate to convince the others or achieve some final solution (Terekhov et al., 2023; Du et al., 2023; Liang et al., 2023; Chan et al., 2023).

**Hierarchical Structure.** Hierarchical structures (Gronauer & Diepold, 2022; Ahilan & Dayan, 2019) typically consist of a leader and one or multiple followers; see Figure 1(b). The leader's role is to guide or plan, while the followers respond or execute based on the leader's instructions. Hierarchical structures are prevalent in scenarios where coordinated efforts directed by a central authority are essential. Multi-agent systems that explore Stackelberg games (Von Stackelberg, 2010; Conitzer & Sandholm, 2006) fall into this category (Harris et al., 2023). This type of game is distinguished by a leadership-followership dynamic and the sequential nature of decision-making: agents make decisions in a sequential order, where the leader player first generates an output (e.g., instructions), then the follower players take an action based on the leader's instruction.

**Nested Structure.** Nested structures, or hybrid structures, constitute sub-structures of equi-level and/or hierarchical structures within the same multi-agent system (Chan et al., 2023); see Figure 1(c). The "big picture" of the system can be either equi-level or hierarchical; however, since some agents have to handle complex tasks, they break down the tasks into smaller ones and construct a sub-system — either equi-level or hierarchical — and "invite" several agents to help with those tasks. In such systems, the interplay between different levels of hierarchy and peer-to-peer interaction contributes to complexity. Also, the interaction among these different structures can lead to intricate dynamics, where strategies and responses become complicated due to the presence of various influencing factors, including external elements like context or environment.

**Dynamic Structure.** Dynamic structures mean that the states of the multi-agent system — e.g., the role of agents, their relations, and the number of agents in the system — may change over time (Talebirad & Nadiri, 2023). As an example, Talebirad & Nadiri (2023) enable the addition and removal of agents so the system can adapt to the tasks at hand. A multi-agent system may also be contextually adaptive, with interaction patterns inside the system modified based on internal system states or external factors, such as context. Agents in such systems can dynamically reconfigure their roles and relationships in response to changing conditions.

*Figure 1 (described): Three diagrams illustrate (a) an equi-level structure — three agents connected to one another with no hierarchy; (b) a hierarchical structure — one leader agent connected to multiple follower agents; and (c) a nested structure — a leader agent connected to follower agents, one of which manages its own internal sub-group of agents.*

### 2.2. Overview of Challenges in Multi-Agent Systems

This paper surveys various components of multi-agent systems and discusses the challenges compared with single-agent systems. We discuss planning, memory management, as well as potential applications of multi-agent systems in distributed systems, e.g., blockchain systems.

**Planning.** In a single-agent system, planning involves the LLM agent breaking down large tasks into a sequence of small, manageable tasks to achieve specific goals efficiently while enhancing interpretability, controllability, and flexibility (Li et al., 2024; Zhang et al., 2023b; Nye et al., 2021; Wei et al., 2022). The agent can also learn to call external APIs for extra information missing from the model weights, or connect LLMs with websites, software, and tools (Patil et al., 2023; Zhou et al., 2023; Cai et al., 2023) to aid reasoning and improve performance. While agents in a multi-agent system have the same capabilities as those in single-agent systems, they encounter challenges inherited from the workflow of multi-agent systems. In §3 we discuss partitioning workflow and allocating sub-tasks to agents — a process we call "global planning" (§3.1) — and then discuss task decomposition for each single agent. Unlike single-agent planning, agents in multi-agent systems must deal with more sophisticated context to reach alignment within the system and, further, achieve consistency toward the overall objective (§3.2).

**Memory management.** Memory management in single-agent systems includes short-term memory during a conversation, long-term memory that stores historical conversations, and, if any, external data storage that serves as a complementary information source for inference, e.g., RAG (Lewis et al., 2020). Memory management in multi-agent systems must handle complex context data and sophisticated interaction and history information, requiring advanced memory design. We classify memory types involved in multi-agent systems in §4.1 and discuss potential challenges posed by sophisticated memory structures in §4.2.

**Application.** We discuss applications of multi-agent systems in blockchain, a distributed system involving sophisticated layers and applications. Multi-agent systems can serve as a tool due to their ability to handle sophisticated blockchain tasks (§5.1). Blockchain can also be integrated with multi-agent systems due to their distributed nature, where an intelligent agent can be allocated to a blockchain node to perform sophisticated actions, such as negotiations, on behalf of a user (§5.2).

## 3. Planning

Planning in multi-agent systems involves understanding the overall tasks and designing the workflow among agents based on their roles and specializations (global planning), and breaking down tasks for each agent into small, manageable tasks (local planning). This process must account for agent functionalities, dynamic interactions among agents, and more complex context compared with single-agent systems — introducing unique challenges and opportunities.

### 3.1. Global Planning

Global planning refers to understanding the overall task, splitting it into smaller ones, and coordinating the sub-tasks among agents. It requires careful consideration of task decomposition and agent coordination.

**Designing effective workflow based on agents' specializations.** Partitioning responsibilities and designing effective workflows is crucial to ensuring each agent's tasks are executable, meaningful, and directly contribute to the overall system objective. The biggest challenges are: 1) the partition of workflow should maximize utilization of each agent's unique capabilities; 2) each agent's tasks must align with the overall goal; and 3) the design must account for context both at the overall-task level and the individual-agent level. This requires a deep understanding of the task and of each agent's specific strengths and limitations.

**Introducing loops for a subset of agents to enhance intermediate results.** Multi-agent systems can integrate loops within one or multiple subsets of agents to improve the quality of intermediate or locally optimal results. In such loops, agents debate or discuss to reach an outcome accepted by the agents involved. This iterative process refines intermediate results and allows agents to adjust their reasoning and plans mid-loop, improving their ability to handle task uncertainty.

**Game Theory.** Game theory provides a structured framework for understanding strategic interactions in multi-agent systems, particularly those involving debates or discussions. A key concept is equilibrium, e.g., Nash Equilibrium (Kreps, 1989) and Stackelberg Equilibrium (Von Stackelberg, 2010; Conitzer & Sandholm, 2006) — a state where no agent benefits from unilaterally changing strategy given others' strategies. Game theory has been applied to multi-agent systems, especially Stackelberg equilibrium (Gerstgrasser & Parkes, 2023; Harris et al., 2023), since its leader/follower structure maps naturally onto hierarchical multi-agent architectures. Gerstgrasser & Parkes (2023) design a general framework to identify Stackelberg Equilibrium in Markov games, and Harris et al. (2023) extend the Stackelberg model to let agents consider external context such as traffic and weather. However, challenges remain in defining appropriate payoff structures for both collective and individual strategies, and in efficiently reaching equilibrium states — highlighting the ongoing need for refinement in applying game theory to complex multi-agent scenarios.

### 3.2. Single-Agent Task Decomposition

Task decomposition in a single agent involves generating intermediate reasoning steps to complete a task or reach an answer — transforming direct input→output mappings into input→rationale→output mappings (Wei et al., 2022; Zhang et al., 2023b). Task decomposition can take several forms:

i. **Chain of Thoughts (CoT)** (Wei et al., 2022) — transforms big tasks into step-by-step manageable tasks representing the agent's reasoning process.
ii. **Multiple CoTs** (Wang et al., 2022a) — explores multiple independent CoT reasoning paths and returns the best-performing one.
iii. **Program-of-Thoughts (PoT)** (Chen et al., 2022) — uses language models to generate text and programming-language statements, arriving at a final answer.
iv. **Table-of-Thoughts (Tab-CoT)** (Ziqi & Lu, 2023) — uses a tabular format for reasoning, explicitly modeling complex reasoning in a highly structured manner.
v. **Tree-of-Thoughts (ToT)** (Yao et al., 2023; Long, 2023) — extends CoT with a tree structure to explore multiple reasoning possibilities at each step, enabling backtracking.
vi. **Graph-of-Thoughts-Rationale (GoT-Rationale)** (Besta et al., 2023) — explores an arbitrary graph structure to aggregate thoughts and enhance them via loops.
vii. **Rationale-Augmented Ensembles** (Wang et al., 2022b) — automatically aggregates diverse rationales to overcome brittleness from sub-optimal individual rationales.

In multi-agent systems, task decomposition for a single agent becomes more intricate: each agent must understand layered context, including 1) the overall task, 2) the specific context of its own sub-tasks, and 3) contextual information from other agents. Agents must align these multi-dimensional contexts with their decomposed tasks to ensure coherent, effective functioning. The paper summarizes the resulting challenges as follows:

- **Aligning Overall Context** — each agent must clearly understand its role and how it fits into the overall task, so its outputs harmonize with other agents' outputs and stay directed at the common goal.
- **Aligning Context Between Agents** — agents must understand and integrate contextual information from other agents so that shared information is fully utilized.
- **Aligning Context for Decomposed Tasks** — when tasks are broken into smaller sub-tasks, each agent's decomposed task must fit both its individual task and the overall goal while integrating other agents' context; agents must continually adapt their understanding as new context arrives.
- **Consistency in Objectives** — consistency must be maintained across all levels, from overall goals down to individual and decomposed tasks. Harris et al. (2023) extend the Stackelberg model to let agents incorporate external context provided by other agents, but aligning complex context with decomposed tasks during reasoning remains unresolved.

## 4. Agent Memory and Information Retrieval

Memory in single-LLM-agent systems refers to the agent's ability to record, manage, and utilize data — such as past queries and external data sources — to aid inference, decision-making, and reasoning (Yao et al., 2023; Park et al., 2023; Li & Qiu, 2023; Wang et al., 2023; Guo et al., 2023). While single-agent memory focuses on internal data management, multi-agent systems require agents to collaborate, necessitating both individual memory capabilities and a sophisticated mechanism for sharing, integrating, and managing information across agents — posing challenges for memory and information retrieval.

### 4.1. Classifications of Memory in Multi-agent Systems

Based on the workflow of a multi-agent system, memory is categorized as follows:

- **Short-term memory** — the immediate, transient memory used during a conversation or interaction (e.g., working memory in Jinxin et al., 2023). It is ephemeral and does not persist once the interaction ends.
- **Long-term memory** — stores historical queries and responses (chat histories from earlier sessions) to support future inference, typically in external storage such as a vector database.
- **External data storage** — an emerging area where models integrate with external storage (e.g., vector databases) to access additional knowledge, enhancing grounding and enriching responses (Lewis et al., 2020) with more informative, accurate, and contextually relevant output.
- **Episodic Memory** — a collection of interactions within the multi-agent system. By referencing past interactions with contextual similarity to the current query, agents can significantly improve response relevance and accuracy, enabling more adaptive and intelligent problem-solving.
- **Consensus Memory** — a unified source of shared information (common sense, domain knowledge, etc.), e.g., the skill library in Jinxin et al. (2023). Agents use consensus memory to align understanding and strategy, enhancing cohesive collaboration.

While both single- and multi-agent systems use short- and long-term memory, multi-agent systems introduce additional complexity due to the need for inter-agent communication, information sharing, and adaptive memory management.

### 4.2. Challenges in Multi-agent Memory Management

Managing memory in multi-agent systems raises challenges and open problems, especially around safety, security, and privacy:

- **Hierarchical Memory Storage** — different agents have varied functionalities and access needs; some may hold sensitive data that must not be accessible to other agents. Robust access control is crucial while still keeping consensus memory accessible to all clients. Where data is not sensitive, unifying data storage can reduce redundancy and improve consistency and efficiency of memory maintenance.
- **Maintenance of Consensus Memory** — since consensus memory is shared by all collaborating agents, ensuring the integrity of that shared knowledge is critical; tampering or unauthorized modification can cause systemic task failures, so rigorous access control is needed to mitigate breach risk.
- **Communication and information exchange** — effective communication and information exchange between agents is essential, since each agent may hold critical pieces of information that must be seamlessly integrated for overall system performance.
- **Management of Episodic Memory** — leveraging past interactions to enhance responses to new queries is challenging; determining how to effectively recall and utilize contextually relevant past interactions for current problem-solving is an open issue.

These challenges highlight the need for continuous research toward robust, secure, and efficient memory management methodologies for multi-agent systems.

## 5. Applications in Blockchain

Multi-agent systems offer significant advantages to blockchain systems by augmenting their capabilities and efficiency, serving as sophisticated tools for various tasks on blockchain and Web3 systems. Blockchain nodes can also be viewed as agents with specific roles and capabilities (Ankile et al., 2023). Since both blockchain systems and multi-agent systems are inherently distributed, blockchain networks can be integrated with multi-agent systems seamlessly — assigning a dedicated agent to each blockchain node can enhance data analysis and processing while bolstering security and privacy.

### 5.1. Multi-Agent Systems As a Tool

Some potential directions where multi-agent systems can act as tools to benefit blockchain systems:

**Smart Contract Analysis.** Smart contracts are programs stored on a blockchain that run when predetermined conditions are met. Multi-agents can work together to analyze and audit smart contracts, with different specializations such as identifying security vulnerabilities, legal compliance, and contract-efficiency optimization — providing a more comprehensive review than a single agent could achieve alone.

**Consensus Mechanism Enhancement.** Consensus mechanisms like Proof of Work (Gervais et al., 2016) or Proof of Stake (Saleh, 2021) are critical for validating transactions and maintaining network integrity. Multi-agent systems can collaborate to monitor network activity, analyze transaction patterns, identify security threats, and propose enhancements to the consensus mechanism, making the blockchain more secure and efficient.

**Fraud Detection.** Fraud detection is a key task in financial monitoring. Ankile et al. (2023) study fraud detection from the perspective of an external observer detecting price manipulation by analyzing transaction sequences or price movements. Multi-agent systems can benefit fraud detection by deploying agents with different roles — e.g., monitoring transactions for fraudulent activity and analyzing user behavior — each focusing on different behavior patterns to improve detection accuracy and efficiency.

### 5.2. Blockchain Nodes as Agents

Ankile et al. (2023) identify blockchain nodes as agents and study fraud detection from an external observer's perspective. However, powerful LLM agents with analysis and reasoning capabilities can do much more, especially combined with game theory enabling negotiation and debate:

**Smart Contract Management and Optimization.** Multi-agent systems can automate and optimize smart contract execution with more flexible terms and dynamic external information from users. Agents can negotiate contract terms on behalf of users, manage contract execution, and optimize gas fees (e.g., on Ethereum; Wood et al., 2014), analyzing context such as past actions and predefined criteria with flexibility. Such negotiations can use game theory — Stackelberg Equilibrium (Von Stackelberg, 2010; Conitzer & Sandholm, 2006) when a leader negotiator exists, or Nash Equilibrium (Kreps, 1989) when no leader exists.

## 6. Conclusion

The exploration of multi-agent systems in this paper underscores their significant potential in advancing the capabilities of LLM agents beyond the confines of single-agent paradigms. By leveraging the specialized abilities and collaborative dynamics among agents, multi-agent systems can tackle complex tasks with enhanced efficiency and innovation. This study has illuminated challenges that need to be addressed to better harness the power of multi-agent systems, including optimizing task planning, managing complex context information, and improving memory management. Furthermore, potential applications of multi-agent systems in blockchain technologies reveal new avenues for development, suggesting a promising future for these systems in distributed computing environments.

## References

- Ahilan, S. and Dayan, P. Feudal multi-agent hierarchies for cooperative reinforcement learning. *arXiv preprint arXiv:1901.08492*, 2019.
- Ankile, L., Ferreira, M. X., and Parkes, D. I see you! Robust measurement of adversarial behavior. In *Multi-Agent Security Workshop@NeurIPS'23*, 2023.
- Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Gianinazzi, L., Gajda, J., Lehmann, T., Podstawski, M., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. *arXiv preprint arXiv:2308.09687*, 2023.
- Cai, T., Wang, X., Ma, T., Chen, X., and Zhou, D. Large language models as tool makers. *arXiv preprint arXiv:2305.17126*, 2023.
- Chan, C.-M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., and Liu, Z. ChatEval: Towards better LLM-based evaluators through multi-agent debate. *arXiv preprint arXiv:2308.07201*, 2023.
- Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. *arXiv preprint arXiv:2211.12588*, 2022.
- Chen, Y., Arkin, J., Zhang, Y., Roy, N., and Fan, C. Scalable multi-robot collaboration with large language models: Centralized or decentralized systems? *arXiv preprint arXiv:2309.15943*, 2023.
- Conitzer, V. and Sandholm, T. Computing the optimal strategy to commit to. In *Proceedings of the 7th ACM Conference on Electronic Commerce*, pp. 82–90, 2006.
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. *arXiv preprint arXiv:2305.14325*, 2023.
- Gerstgrasser, M. and Parkes, D. C. Oracles & followers: Stackelberg equilibria in deep multi-agent reinforcement learning. In *International Conference on Machine Learning*, pp. 11213–11236. PMLR, 2023.
- Gervais, A., Karame, G. O., Wüst, K., Glykantzis, V., Ritzdorf, H., and Capkun, S. On the security and performance of proof of work blockchains. In *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security*, pp. 3–16, 2016.
- Gronauer, S. and Diepold, K. Multi-agent deep reinforcement learning: A survey. *Artificial Intelligence Review*, pp. 1–49, 2022.
- Guo, Z., Cheng, S., Wang, Y., Li, P., and Liu, Y. Prompt-guided retrieval augmentation for non-knowledge-intensive tasks. *arXiv preprint arXiv:2305.17653*, 2023.
- Harris, K., Wu, S., and Balcan, M. F. Stackelberg games with side information. In *Multi-Agent Security Workshop@NeurIPS'23*, 2023.
- Jinxin, S., Jiabao, Z., Yilei, W., Xingjiao, W., Jiawen, L., and Liang, H. CGMI: Configurable general multi-agent interaction framework. *arXiv preprint arXiv:2308.12503*, 2023.
- Kreps, D. M. Nash equilibrium. In *Game Theory*, pp. 167–177. Springer, 1989.
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33:9459–9474, 2020.
- Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., and Ghanem, B. CAMEL: Communicative agents for "mind" exploration of large scale language model society. *arXiv preprint arXiv:2303.17760*, 2023.
- Li, X. and Qiu, X. MoT: Memory-of-thought enables ChatGPT to self-improve. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 6354–6374, 2023.
- Li, X., Sun, M., and Li, P. Multi-agent discussion mechanism for natural language generation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, pp. 6096–6103, 2019.
- Li, Y., Wen, H., Wang, W., Li, X., Yuan, Y., Liu, G., Liu, J., Xu, W., Wang, X., Sun, Y., et al. Personal LLM agents: Insights and survey about the capability, efficiency and security. *arXiv preprint arXiv:2401.05459*, 2024.
- Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., and Shi, S. Encouraging divergent thinking in large language models through multi-agent debate. *arXiv preprint arXiv:2305.19118*, 2023.
- Long, J. Large language model guided tree-of-thought. *arXiv preprint arXiv:2305.08291*, 2023.
- Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D., et al. Show your work: Scratchpads for intermediate computation with language models. *arXiv preprint arXiv:2112.00114*, 2021.
- Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*, pp. 1–22, 2023.
- Patil, S. G., Zhang, T., Wang, X., and Gonzalez, J. E. Gorilla: Large language model connected with massive APIs. *arXiv preprint arXiv:2305.15334*, 2023.
- Saleh, F. Blockchain without waste: Proof-of-stake. *The Review of Financial Studies*, 34(3):1156–1190, 2021.
- Talebirad, Y. and Nadiri, A. Multi-agent collaboration: Harnessing the power of intelligent LLM agents. *arXiv preprint arXiv:2306.03314*, 2023.
- Terekhov, M., Graux, R., Neville, E., Rosset, D., and Kolly, G. Second-order jailbreaks: Generative agents successfully manipulate through an intermediary. In *Multi-Agent Security Workshop@NeurIPS'23*, 2023.
- Von Stackelberg, H. *Market Structure and Equilibrium*. Springer Science & Business Media, 2010.
- Wang, W., Dong, L., Cheng, H., Liu, X., Yan, X., Gao, J., and Wei, F. Augmenting language models with long-term memory. *arXiv preprint arXiv:2306.07174*, 2023.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*, 2022a.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., and Zhou, D. Rationale-augmented ensembles in language models. *arXiv preprint arXiv:2207.00747*, 2022b.
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35:24824–24837, 2022.
- Wood, G. et al. Ethereum: A secure decentralised generalised transaction ledger. *Ethereum Project Yellow Paper*, 151(2014):1–32, 2014.
- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. *arXiv preprint arXiv:2305.10601*, 2023.
- Zhang, J., Xu, X., and Deng, S. Exploring collaboration mechanisms for LLM agents: A social psychology view, 2023a.
- Zhang, Z., Yao, Y., Zhang, A., Tang, X., Ma, X., He, Z., Wang, Y., Gerstein, M., Wang, R., Liu, G., et al. Igniting language intelligence: The hitchhiker's guide from chain-of-thought reasoning to language agents. *arXiv preprint arXiv:2311.11797*, 2023b.
- Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Bisk, Y., Fried, D., Alon, U., et al. WebArena: A realistic web environment for building autonomous agents. *arXiv preprint arXiv:2307.13854*, 2023.
- Ziqi, J. and Lu, W. Tab-CoT: Zero-shot tabular chain of thought. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 10259–10277, Toronto, Canada, July 2023. Association for Computational Linguistics.
