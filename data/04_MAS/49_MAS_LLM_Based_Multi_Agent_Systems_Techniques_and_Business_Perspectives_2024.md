# LLM-based Multi-Agent Systems: Techniques and Business Perspectives

**Yingxuan Yang** — Shanghai Jiao Tong University, Shanghai, China (zoeyyx@sjtu.edu.cn)
**Qiuying Peng** — OPPO Research Institute, Shenzhen, China (qypeng.ustc@gmail.com)
**Jun Wang** — OPPO Research Institute, Shenzhen, China (junwang.lu@gmail.com)
**Ying Wen** — Shanghai Jiao Tong University & SII, Shanghai, China (ying.wen@sjtu.edu.cn)
**Weinan Zhang*** — Shanghai Jiao Tong University & SII, Shanghai, China (wnzhang@sjtu.edu.cn)

*Corresponding Author

arXiv:2411.14033v2 [cs.AI] 28 Dec 2024

---

## Abstract

In the era of (multi-modal) large language models, most operational processes can be reformulated and reproduced using LLM agents. The LLM agents can perceive, control, and get feedback from the environment so as to accomplish given tasks in an autonomous manner. Besides the environment-interaction property, LLM agents can call various external tools to ease the task completion process. The tools can be regarded as a predefined operational process with private or real-time knowledge that does not exist in the parameters of LLMs. As a natural trend of development, the tools for calling are becoming autonomous agents, thus the full intelligent system turns out to be an LLM-based Multi-Agent System (LaMAS). Compared to the previous single-LLM-agent system, LaMAS has the advantages of i) dynamic task decomposition and organic specialization, ii) higher flexibility for system changing, iii) proprietary data preserving for each participating entity, and iv) feasibility of monetization for each entity. This paper discusses the technical and business landscapes of LaMAS. To support the ecosystem of LaMAS, we provide a preliminary version of such a LaMAS protocol considering technical requirements, data privacy, and business incentives. As such, LaMAS would be a practical solution to achieve artificial collective intelligence in the near future.

**Keywords:** LLM-based Multi-Agent System, large language model, data privacy, monetization

## 1. Background and Trend

The development of Large Language Models (LLMs) marks a key advancement in artificial intelligence. These models have transformed from simple text processors to sophisticated systems capable of reasoning, understanding multimodal inputs, and making autonomous decisions. Such developments have enabled the emergence of AI agents powered by LLMs, which can adapt to diverse tasks, comprehend context, and interact with their environments autonomously. *(For presentation brevity, the multi-modal LLM concept is merged into the LLM concept throughout this paper.)*

A critical transition in LLM capabilities is their evolution from passive tools that merely respond to commands to active agents capable of independent decision-making and action-taking. Initially, LLMs were primarily used for single-purpose tasks such as text generation or analysis. Recent advances have equipped them to interact with graphical user interfaces (GUIs) and perform complex operations such as web browsing, app navigation, and system control. Beyond these capabilities, modern LLMs have transformed into autonomous agents that dynamically select and use tools based on contextual requirements — highlighting their dual nature: they utilize tools but can also function as tools within modular systems, enabling multi-agent architectures where agents collaborate to solve complex problems.

The rise of LLM-based Multi-Agent Systems (LaMAS) marks a significant leap in AI applications. Although such systems may require greater computational resources compared to single-agent approaches, they offer crucial advantages that justify this trade-off: inherent fault tolerance through agent redundancy, natural task decomposition without explicit workflow design, and organic specialization in complex problem-solving. When one agent fails, others can seamlessly continue operations, providing robust reliability that centralized systems cannot match. While single-agent systems demand careful orchestration of execution workflows for each task type, multi-agent systems naturally emerge with collaborative specialization patterns, letting each agent focus on its core competencies within the larger system architecture.

Recognizing these benefits, researchers have developed LaMAS frameworks to enable complex task collaboration. Beyond traditional paradigms like SaaS, PaaS, and IaaS, LaMAS introduces a novel approach by seamlessly integrating intelligent agents into cloud ecosystems. This framework supports deployment of specialized agents capable of collaboration while maintaining data privacy and security, and establishes a marketplace for agent monetization, allowing users to customize and combine agent services according to their needs. The system architecture emphasizes modular design, standardized communication protocols, and robust security measures, fostering sustainable innovation.

**Incentivization via Monetization Mechanisms.** Just as Internet applications are highly incentivized to connect to the Internet, agents in a LaMAS are highly incentivized based on monetization mechanism design. First, the experience data generated from interacting within a LaMAS is crucial for training well-functional agents. In LaMAS, agents receive task instructions from upstream agents, perform inner-agent reasoning and tool usage, send task instructions to downstream agents, acquire returned information, and obtain the final task-accomplishment results. Such experience data is more valuable and of larger volume than a single agent just connecting to users. Second, similar to Internet monetization via online advertising, there will be a monetization mechanism over the LaMAS: for each accomplished task assigned a business value (e.g., the user books a hotel or purchases an item), there will be a promotion fee from the merchant provided to the engaged team of agents, and a credit allocation mechanism can be built based on participation or essential contribution to task accomplishment. As such, the entity behind each agent has essential motivation to build a highly intelligent agent connecting to the LaMAS.

**Entity's Responsibility based on Agent Intelligence.** For Cable or 4/5G Internet, each entity behind an Internet service (company, institute, or team) is responsible for maintaining stable function and connection; if its server crashes or connection is disabled, dependent services are highly influenced, so the entity should take charge of the influence it makes. Analogously, in the LaMAS ecosystem, the entity behind each agent is responsible for making the ecosystem run smoothly and intelligently: first, each agent needs to support stable function and connection (inherited from Internet services); second, and more importantly, the intelligence provided by the agent must meet or exceed predefined standards, since low intelligence from one agent could make the whole LaMAS less functional for accomplishing intelligence tasks.

This paper presents perspectives on LaMAS by discussing its technical and business landscapes: key AI technical aspects (system architectures, collaboration protocols, agent training methods) and business aspects (data privacy-preserving and monetization via traffic and intelligence). With these analyses, LaMAS is expected to form a new technical-business paradigm in the coming years.

## 2. Key AI Technical Aspects

### 2.1. Architecture of LLM Agents

The architecture of LLM-based AI agents consists of several interrelated components essential for autonomous operation and intelligent interaction. At its core, this architecture is designed to effectively process inputs, maintain contextual relevance, make informed decisions, and generate appropriate responses.

The **interaction wrapper** serves as the principal interface through which the agent interacts with its environment and other agents. It manages the flow of incoming and outgoing communications, adapts to various input modalities, standardizes them for internal processing, and implements protocol-specific adaptations to ensure seamless integration with various communication standards — preserving the internal consistency of the agent's operations.

**Memory management** is pivotal to the architecture, including both short-term working memory and long-term episodic storage. The short-term memory buffer retains immediate context and recent interactions, facilitating conversational coherence, while the long-term memory system archives significant experiences and learned patterns, enabling the agent to adapt its responses based on historical interactions and enhance decision-making in contextually rich scenarios.

The **cognitive functionality** of the architecture is currently underpinned by Chain-of-Thought (CoT) reasoning. This structured reasoning framework decomposes complex tasks into manageable logical steps, facilitating clarity and thoroughness in problem-solving. CoT enables the agent to articulate intermediate reasoning states, verify logical consistency, and engage in self-correction through systematic analysis of its own reasoning processes.

To enhance operational capacity beyond natural language processing, a **tool integration framework** is necessary. This subsystem discovers and registers tools, maps parameters between natural language commands and tool APIs, monitors execution, handles errors, and interprets results — ensuring effective integration of external functionalities into the agent's decision-making process.

The architecture also features a sophisticated **routing mechanism** that governs connections with neighboring agents, facilitating dynamic neighbor discovery, capability-based routing decisions, load balancing across the agent network, and policy-based access control — vital for efficient communication and collaboration within multi-agent systems.

Finally, the architecture incorporates **feedback loops** enabling continuous learning and adaptation, allowing the agent to process interaction outcomes, update its internal models, and refine decision-making strategies based on experiential learning. Together, these elements establish a robust foundation for LaMAS's autonomous operation and significantly enhance collaborative capabilities.

*Figure 1 (described): An illustration of LaMAS showing a user issuing natural-language instructions (e.g., navigation, ordering lunch, sending an email, checking weather) to a simple/complex task interface with app icons (Chat, Shopping, Navigation, Calendar, Music, Video, Document, Meeting), which in turn map onto a multi-agent system workflow — e.g., "Organizing an Online Event" — where agents discuss ideas, check schedules, prepare and revise an agenda, and finalize a document collaboratively.*

### 2.2. Mechanisms and Architectures of LaMAS

As a multi-agent system, the design of mechanisms and architectures of a LaMAS is crucial for its success. Roughly, according to the coordination form of a MAS, there are three major architectures:

1. **Fully centralized architectures** — the whole system has full control of the engaged agents, a very high requirement; centralized training with centralized execution can be used, and agents act with high coordination. In practice, this applies only when agents are applications developed over an OS-like platform that grant the platform data and control access.
2. **Decentralized architectures with global credit allocation** — the system cannot fully control engaged agents but can allocate credit to each for each accomplished task; centralized training with centralized execution can still be applied. This is more practical since each agent (and its entity) does not need to grant data or control access to the platform, while the platform can still incentivize agents to improve collaboration performance via credit allocation.
3. **Fully decentralized architecture** — no access to data or control for each engaged agent and no credit allocation from the platform; agents must find their own way to collaborate and improve themselves. Here, mechanism design is critical from the beginning.

### 2.3. Protocols of Agent Interaction

The LaMAS framework necessitates sophisticated interaction protocols to facilitate effective agent collaboration. These protocols must bridge the gap between traditional structured formats and natural language understanding, addressing unique challenges posed by LLM-based agents' probabilistic decision-making and emergent capabilities.

**Core Challenges and Key Issues.** Developing LaMAS protocols presents fundamental challenges in protocol effectiveness measurement, behavioral diversity optimization, and non-transitive interaction management. Key open questions include:

- How do we assess the effectiveness of interaction protocols for collaboration?
- How can we create protocols that encourage diverse agent behaviors while ensuring system performance?
- How can we design protocols to foster both diversity and effectiveness in agent behaviors?
- How do we handle non-transitivity in agent interactions within these protocols?

From these challenges, three critical issues in protocol design emerge. First, LaMAS requires a layered protocol architecture to manage diverse agent interactions efficiently, enabling dynamic protocol selection based on task and agent capabilities. Second, as system scale increases, traditional protocols face limitations in managing communication overhead and maintaining consistency, necessitating innovative approaches to protocol design. Third, LaMAS should leverage LLM agents' strengths in language understanding and contextual interpretation — e.g., handling ambiguous commands or enabling real-time negotiation to clarify information.

**Core Protocol Framework.** A comprehensive protocol framework is proposed, consisting of five essential components: the instruction processing protocol, the message exchange protocol, the consensus formation protocol, the credit allocation protocol, and the experience management protocol.

*Figure 2 (described): A protocol hierarchy diagram with three tiers — High-Level Protocols (User Instruction Protocol, feeding into Message Passing Protocol and Consensus & Voting Protocol), Mid-Level Protocols (Credit Allocation & Propagation Protocol, Experience Logging & Learning Protocol), and Low-Level Implementation (Sync Communication, Async Communication, Data Storage, Machine Learning).*

- The **Instruction Processing Protocol** standardizes the interpretation of user instructions through structured parsing mechanisms and context-aware processing pipelines, implementing sophisticated disambiguation techniques to handle uncertain or incomplete instructions while maintaining consistency across multiple interaction rounds.
- The **Message Exchange Protocol** establishes the foundation for inter-agent communication through standardized message formats and adaptive transmission mechanisms, dynamically switching between synchronous and asynchronous modes based on task requirements and system load, and implementing priority-based routing algorithms to optimize message delivery under varying conditions.
- The **Consensus Formation Protocol** implements distributed decision-making mechanisms through a combination of voting systems and negotiation frameworks, adapting consensus thresholds dynamically based on task criticality and system state. When proposals conflict, agents resolve disagreements through negotiation, and voting protocols let agents express preferences and reach decisions even without full consensus, preventing deadlocks.
- The **Credit Allocation Protocol** addresses fair contribution assessment through multi-level propagation mechanisms; agents receive credit corresponding to their contributions via task-specific metrics and performance-based distribution algorithms, ensuring equitable reward allocation while incentivizing collaborative behavior.
- The **Experience Management Protocol** facilitates collective learning through structured logging and pattern extraction mechanisms. Each agent logs experiences and learning outcomes — successes, failures, strategy effectiveness, and interactions with other agents — implementing cross-agent knowledge sharing algorithms for systematic performance improvement through accumulated experience.

The effectiveness of LaMAS depends on the seamless integration of these protocols; the hierarchical organization enables dynamic protocol selection and efficient resource utilization while maintaining system scalability.

### 2.4. Agent Training Methods

In LaMAS, each agent has an incentive to improve itself to gain more credits from the platform. "Agent training" here refers to methods of improving agent performance, including tuning-free methods and parameter-tuning methods.

**Tuning-free Methods.** These are strategies to improve performance without modifying model parameters — beneficial when direct parameter tuning is costly or impractical:

- **Prompt Engineering** — designing specific input prompts to elicit desired responses without any parameter adjustment.
- **Few-Shot Learning** — providing limited examples within the prompt to help the agent understand new tasks; in zero-shot learning, models tackle tasks solely through natural language instructions leveraging pre-trained knowledge. Both approaches enable flexibility and adaptability in multi-agent environments.
- **External Tool Utilization** — agents enhance capabilities by interacting with external tools or APIs (databases, calculators, etc.), performing complex tasks without additional model training.

These tuning-free methods are particularly valuable in LaMAS, enabling agents to adapt quickly and collaborate on complex tasks in dynamic environments with minimal computational cost.

**Parameter-tuning Methods.** To directly tune the parameters of the LLMs behind each agent, alignment methods and multi-agent reinforcement learning (MARL) methods can be used. Alignment methods for tuning LLMs are generally based on supervised learning loss on the target output or expert preference: directly fitting expert output corresponds to behavioral cloning in agent imitation learning, while training on expert preference pairs improves the agent's policy in a learning-to-rank manner. This kind of method has not been much utilized in multi-agent tasks since the alignment target is not clearly formulated in such scenarios. MARL is a key method for training agent policy in a multi-agent system, formulating the task as a multi-agent sequential decision-making problem. Considering mainly cooperative MARL (agents pursuing team success, i.e., fulfilling the user's task), MARL methods divide into three major categories according to the form of agent coordination in training and execution: i) centralized training with centralized execution, ii) centralized training with decentralized execution, and iii) decentralized training and execution.

### 2.5. Attacks and Defenses in LaMAS

As LaMAS systems handle sensitive data and critical operations, security is a top concern. The distributed nature of LaMAS introduces unique vulnerabilities beyond those of single-LLM systems: malicious actors can target individual agents as well as exploit inter-agent communications and collective decision-making processes.

**Attack Surface and Vulnerabilities.** LaMAS face three main types of attacks:

1. **Prompt injection attacks** — manipulate input prompts to trick models into generating harmful responses. These are particularly dangerous in LaMAS, where compromised agents can propagate malicious prompts across the system; research shows slight changes in input phrasing can bypass defenses, and system prompts can be altered using escape characters and context omission.
2. **Memory and data poisoning attacks** — target the knowledge bases agents use for decision-making. In LaMAS, poisoned data can affect multiple agents simultaneously; contaminated knowledge bases in RAG systems can cause cascading errors throughout the agent network, and poisoned training samples with specific triggers can compromise fine-tuned agents, impacting system reliability.
3. **Model inversion and extraction attacks** — aim to reconstruct training data or extract model details through targeted queries. These attacks are particularly effective in LaMAS, where attackers can leverage responses from multiple agents to enhance extraction efficiency; the risk of data leakage is especially high for systems handling sensitive personal or commercial data.

**Defense Mechanisms and Future Directions.** Several defense strategies address specific LaMAS vulnerabilities:

- **Input sanitization** techniques, such as prompt randomization and query encapsulation, help neutralize prompt injection attacks, though they may introduce communication overhead; adaptive delimiter strategies can help maintain communication efficiency.
- **Perplexity-based filtering** can detect adversarial prompts without compromising model utility; in LaMAS, cross-validating perplexity scores across agents can enhance this, though careful calibration is required to avoid false positives.
- **Adversarially robust fine-tuning** — a dual-model approach generating and validating adversarial samples during training offers significant benefits; further optimization balances robustness and utility, valuable for system-wide application while preserving agent specialization.

Challenges remain: current defenses often struggle with the dynamic nature of agent interactions (complex communication patterns can trigger false positives), and the computational overhead of comprehensive security measures can affect performance, requiring a balance between security and efficiency. Future research directions include standardized security evaluation frameworks accounting for individual agent vulnerabilities and system-wide risks, lightweight security measures maintaining communication efficiency, and adaptive defense mechanisms that evolve with emerging threats. A holistic approach combining robust model architectures, effective training procedures, and dynamic defense mechanisms will be critical for maintaining public trust as LaMAS systems grow in complexity and impact.

## 3. Key Business Aspects

Drawing from research on LaMAS, this section presents a vision of its business implications across three critical dimensions: privacy preservation, traffic monetization, and intelligence monetization.

### 3.1. Privacy Preservation in LaMAS

The rise of LaMAS introduces privacy challenges beyond those of traditional multi-agent systems. Unlike conventional agents exchanging structured data, LLM agents handle rich, contextual information that may contain sensitive data embedded in natural language conversations, reasoning processes, and knowledge representations. Privacy preservation is critical because these systems process natural language data, which can inadvertently leak sensitive information through semantic connections and implicit knowledge representations.

**Privacy-Preserving Challenges** exist at three levels:

- **Semantic level** — LLMs' natural language processing may inadvertently reveal sensitive information through contextual associations and semantic connections; traditional privacy mechanisms designed for structured data are insufficient here, especially against attacks exploiting semantic vulnerabilities.
- **Agent interaction level** — continuous information exchange between agents introduces privacy risks; sensitive information can be exposed through behavioral patterns and response characteristics, not just direct content, and maintaining conversation history/context windows creates persistent vulnerabilities over time.
- **System architecture level** — the distributed nature of LaMAS complicates enforcing privacy guarantees across all components while maintaining efficiency; dynamic agent interactions and evolving knowledge further challenge robust privacy protection.

**Privacy-Preserving Technologies** explored to address these challenges include:

- **Homomorphic Encryption (HE)** — enables secure computation on encrypted data, supporting private agent-to-agent communication and inference; promising for privacy-preserving machine learning and secure data sharing, though computational complexity remains a significant challenge in LaMAS.
- **Secure Multi-Party Computation (SMPC)** — enables secure collaborative computation among multiple agents; used in privacy-preserving data analysis and collaborative learning in traditional multi-agent systems, though scaling to large LaMAS remains an open question.
- **Trusted Execution Environments (TEEs)** — provide hardware-based security guarantees (e.g., Intel SGX, ARM TrustZone, AMD SEV) creating secure enclaves for sensitive computations; integrating TEEs into LaMAS requires careful security/performance trade-off consideration.
- **Differential Privacy (DP)** — offers mathematical methods for privacy-preserving data analysis; effective for protecting sensitive information in collaborative tasks but faces unique NLP challenges such as managing privacy budgets and preserving utility.

**Research Directions and Open Challenges.** Existing privacy metrics do not fully capture the complexities of semantic information leakage in NLP, so LaMAS-specific privacy frameworks accounting for both direct and indirect information flows in semantic spaces are needed. Integrating privacy-preserving technologies across LaMAS requires a unified approach combining data protection, secure computation, and communication security; performance optimization and scalability remain major hurdles as agent networks expand. Future research should focus on comprehensive privacy frameworks tailored to LaMAS, including standardized privacy protocols, efficient implementations, and evaluation metrics for practical deployment.

### 3.2. Traffic Monetization

Traffic Monetization in LaMAS involves generating commercial value by managing user traffic and optimizing ads using the strengths of various agents — improving traffic flow, boosting click-through rates (CTR), and increasing conversion rates (CVR). LaMAS leverages each agent's capabilities to enhance user engagement and make advertising strategies more effective, while ensuring fair and transparent revenue allocation.

**Business Scenarios and Revenue Generation.** Agents analyze user behaviors and preferences to optimize traffic management and deploy targeted advertisements, building user profiles and using intelligent recommendation systems to personalize ads. Revenue mainly comes from advertising, using Cost Per Click (CPC) and Cost Per Action (CPA) models: in CPC, advertisers pay based on clicks with agents earning commissions based on their contribution to traffic management and ad effectiveness; in CPA, payments are made for completed purchases, rewarding agents that drive conversions with a higher revenue share. Additional income can come from user subscriptions for premium features or personalized services (e.g., advanced analytics dashboards, exclusive tool access).

*Figure 3 (described): A radial diagram of "Traffic Monetization" with segments for Business Scenarios (Subscription, CPA, CPC), Profit Allocation (Contribution Assessment, Shapley Value), and Roles of Agents (Advert Agent, Data Analysis Agent, Transaction Agent, Subscription Agent).*

**Profit Allocation Mechanisms.** Converting revenue into fairly distributed profits starts by assessing each agent's contribution to traffic generation, ad clicks, and conversions using metrics like CTR and CVR. LaMAS may use blockchain-based smart contracts to automate distribution and minimize bias/human error. A scoring system rates agents on performance (including user feedback and engagement), with higher-scoring agents receiving a larger revenue share. Attribution methods such as the Shapley Value ensure profits are allocated based on each agent's contribution; dynamic adjustment mechanisms allow real-time updates to revenue shares based on performance and market conditions, and metrics like CPM (Cost Per Mille) add a more nuanced view of ad performance beyond clicks and conversions.

**Roles of Application Agents.** Advertising agents manage and deploy ads, using data analytics to optimize performance and select placements. Data analysis agents analyze user behavior to help advertising agents refine strategies and identify emerging trends. Transaction agents handle purchases, ensure smooth transactions, track conversions, and link sales performance to specific ads. Subscription agents manage premium services and personalized features, contributing additional revenue streams and long-term engagement through retention and loyalty.

Future research should focus on improving attribution models like the Shapley Value for fairer profit allocation, and adding metrics like CPM to allocate revenue more accurately.

### 3.3. Intelligence Monetization

Intelligence Monetization in LaMAS represents a significant evolution in AI commercialization by leveraging the collaborative capabilities of specialized agents. Unlike traditional single-model paradigms, multi-agent systems enable dynamic interactions among specialized agents, each addressing specific tasks, facilitating more versatile and robust intelligence solutions. This paradigm is exemplified by Microsoft's Copilot Studio Platform (launched November 19, 2024), which supports an ecosystem of over 1,800 large models and offers open APIs and integration tools for enterprise customization and scalability.

**Revenue Generation through Data-Driven Services.** A key revenue model is the sale of data-driven services: specialized agents analyze distinct datasets (consumer preferences, product usage, market trends) to generate actionable insights, delivered as reports, forecasts, or tailored recommendations that businesses can purchase. For instance, one agent may provide personalized user-behavior insights for marketing while another offers market trend analysis for strategic planning. Successful implementations like OpenAI's GPT-4 API demonstrate how multiple specialized models can work in concert, with distinct agents handling data processing/preprocessing, deep pattern recognition and insight generation, recommendation transformation, and platform integration.

**Innovative Licensing and Agent Marketplaces.** LaMAS introduces licensing approaches beyond traditional software licensing, most prominently Agent-as-a-Service (AaaS) — as seen in Google Cloud's AutoML — enabling dynamic agent deployment based on computational needs, with usage-based pricing and automatic scaling. Complementing this is the emergence of agent marketplace platforms creating ecosystems for third-party agent development and deployment, as demonstrated by Hugging Face's model hub adapted for LLM deployment. Hybrid deployment architectures combining on-premise agent deployment for sensitive operations with cloud-based agents for scalable tasks (as in IBM's Watson services) are also expected to become increasingly popular.

Looking ahead, as demand for AI-driven insights grows, LaMAS's role in delivering scalable, actionable intelligence will continue to expand, with future developments focused on enhancing agent collaboration for real-time insights and adapting to emerging business models and industries.

### 3.4. Integration of 3 Business Aspects

The three key business aspects of LaMAS form an interconnected framework driving both commercial success and ethical operation. Data privacy ensures trust and compliance while allowing secure data usage; this foundation supports traffic monetization by generating user engagement data while adhering to privacy regulations. Intelligence monetization transforms these privacy-preserved interactions into actionable insights and services. As LaMAS evolves, maintaining a balance between privacy, commercial success, and technological progress will be critical for long-term sustainability, with future development focused on strengthening these connections while adapting to evolving privacy regulations and market demands.

## 4. Case Study

Building on the technical foundations and business considerations of LaMAS, this section delves into real-world implementations to illustrate how these theoretical frameworks are realized in practice, exploring how different architectural choices influence system efficiency, data privacy, and monetization capabilities.

### 4.1. Architectures in LaMAS

Real-world LaMAS implementations reveal various architectural patterns, each addressing specific operational requirements and constraints. Architectural choices significantly influence system capabilities, from privacy protection to operational efficiency. Four fundamental patterns have emerged in practice:

*Figure 4 (described): Four network diagrams illustrating Star, Ring, Graph, and Bus architectures of LaMAS.*

- **Star Architecture** — a central agent coordinates communication with all other agents; this centralized control model works well when one agent is responsible for task distribution and overall orchestration.
- **Ring Architecture** — agents are arranged in a circular configuration, each communicating with its predecessor and successor; this decentralized structure supports sequential task processing, with each agent having a specific role in the task pipeline.
- **Graph Architecture** — a fully (or non-fully) interconnected network where each agent can communicate directly with any other agent (or its neighbors), providing maximum flexibility and redundancy to support complex interactions.
- **Bus Architecture** — uses a fixed workflow or Standard Operating Procedure (SOP), where tasks are sent to a central bus that then distributes them to appropriate agents or processes, ensuring a clear input-output mechanism and structured, sequential task flow.

### 4.2. A Decentralized Star Architecture in LaMAS

The first case study illustrates LaMAS implementation in a **music service scenario**. The system uses a centralized architecture where several agents — a Personal Agent, an Orchestrator Agent, and a Song Agent — collaborate to process the user's music playback requests, with the Orchestrator Agent acting as a central hub managing communication and coordinating tasks.

*Figure 5 (described): A sequence diagram of the Centralized Architecture of LaMAS, showing the User asking to "Play the new Taylor Swift song from yesterday," which flows through the Personal Agent and Orchestrator Agent, which searches for and finds "Cruel Summer" by Taylor Swift, then requests the Song Agent to start playback.*

However, this centralized approach means all agents must send their data — including sensitive user information — through the Orchestrator Agent to complete tasks. While efficient for coordination, this creates potential privacy and security risks since user data passes through multiple agents.

*Figure 6 (described): A sequence diagram of Centralized data handling of LaMAS, adding a Transaction Logger and a highlighted "Data Protection Phase" where user requests are logged with encrypted user IDs, alongside credit distribution updates for the Personal Agent, Orchestrator Agent, and Song Agent.*

To address these privacy concerns, a modified **decentralized Star Architecture** is proposed, illustrated through a **travel booking scenario**. In this design, the Orchestrator Agent still coordinates tasks but avoids directly handling sensitive data; specialized agents (like a Navigation Agent or Ticket Agent) process their tasks independently and interact directly with user data when needed, reducing privacy risks while maintaining system efficiency.

*Figure 7 (described): A diagram of the Decentralized Star Architecture of LaMAS, with a User connected to an Orchestrator Agent, which coordinates a Navigation Agent and Ticket Agent; a sequence diagram below shows "Plan trip to Shanghai" flowing with control via the Orchestrator but data flowing directly between the User and the Navigation/Ticket agents (e.g., sharing location data, route checks, booking details, ticket confirmation).*

In the decentralized architecture, the Orchestrator Agent focuses on breaking user instructions into smaller tasks and deciding execution order, staying uninvolved in sensitive data processing and reconnecting only when tasks are completed or additional coordination is required. Each specialized agent handles its specific tasks within its own data domain, ensuring privacy and security.

*Figure 8 (described): A sequence diagram of Decentralized data handling of LaMAS, again featuring a highlighted Data Protection Phase (logging user requests and search/booking operations with encrypted or excluded user information) and, at the end, credit distribution updates across the Orchestrator Agent, Navigation Agent, and Ticket Agent, managed by a Transaction Logger.*

A fair credit allocation system, managed by the Transaction Logger, ensures all agents receive appropriate rewards based on the tasks they complete and the resources they use — improving data protection and system security while keeping operations efficient.

## 5. Conclusion & Future

This paper provides an analysis of the future development of LLM-based Multi-Agent Systems (LaMAS) from the perspectives of techniques and business. Technically, compared to traditional single-LLM-agent systems, LaMAS has higher potential for overall performance and system flexibility; commercially, LaMAS brings the feasibility of proprietary data preservability and monetization through traffic and intelligence, which essentially incentivizes various entities to contribute to the whole ecosystem. Several effective protocols for multi-agent communication and collaboration are being developed, which will drive the implementation of the LaMAS ecosystem toward achieving artificial collective intelligence in the near future.

## References

1. Gabriel Alon and Michael Kamfonas. 2023. Detecting language model attacks with perplexity. *arXiv preprint arXiv:2308.14132*.
2. Abdollah Amirkhani and Amir Hossein Barshooi. 2022. Consensus in multiagent systems: a review. *Artif. Intell. Rev.* 55, 5 (June 2022), 3897–3935.
3. Haris Aziz. 2010. Multiagent systems: algorithmic, game-theoretic, and logical foundations (review). *SIGACT News* 41, 1 (March 2010), 34–37.
4. Tom B. Brown, Benjamin Mann, Nick Ryder, et al. 2020. Language Models are Few-Shot Learners. *arXiv:2005.14165*.
5. E. Brynjolfsson and A. McAfee. 2014. The Second Machine Age: Work, Progress, and Prosperity in a Time of Brilliant Technologies. *Journal of Advertising Research*.
6. Lucian Busoniu, Robert Babuska, and Bart De Schutter. 2008. A comprehensive survey of multiagent reinforcement learning. *IEEE Transactions on Systems, Man, and Cybernetics, Part C* 38, 2, 156–172.
7. Davide Caffagni et al. 2024. The (r)evolution of multimodal large language models: A survey. *arXiv:2402.12451*.
8. Chi-Min Chan, Weize Chen, Yusheng Su, et al. 2023. ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate. *arXiv:2308.07201*.
9. Hao Chen, Kim Laine, and Peter Rindal. 2017. Fast private set intersection from homomorphic encryption. *Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security*, 1243–1255.
10. Weize Chen, Yusheng Su, Jingwei Zuo, et al. 2023. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. *arXiv:2308.10848*.
11. Weize Chen, Ziming You, Ran Li, et al. 2024. Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligence. *arXiv:2407.07061*.
12. Jung Hee Cheon, Andrey Kim, Miran Kim, and Yongsoo Song. 2017. Homomorphic encryption for arithmetic of approximate numbers. *Advances in Cryptology–ASIACRYPT 2017*, 409–437.
13. Google Cloud. 2024. AutoML - Google Cloud. Accessed 2024-11-12.
14. Victor Costan. 2016. Intel SGX explained. *IACR Cryptol. EPrint Arch.*
15. Cynthia Dwork. 2006. Differential privacy. *International Colloquium on Automata, Languages, and Programming*, 1–12.
16. Hugging Face. 2024. Hugging Face Model Hub. Accessed 2024-11-12.
17. Abou Zakaria Faroukhi, Imane El Alaoui, Youssef Gahi, and Aouatif Amine. 2020. Big data monetization throughout Big Data Value Chain: a comprehensive review. *Journal of Big Data* 7, 1–22.
18. Adam Fourney et al. 2024. Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks. *arXiv:2411.04468*.
19. Dayuan Fu, Biqing Qi, Yihuai Gao, et al. 2024. MSI-Agent: Incorporating Multi-Scale Insight into Embodied Agents for Superior Planning and Decision-Making. *arXiv:2409.16686*.
20. Dawei Gao, Zitao Li, Xuchen Pan, et al. 2024. AgentScope: A Flexible yet Robust Multi-Agent Platform. *arXiv:2402.14034*.
21. Alireza Ghafarollahi and Markus J. Buehler. 2024. SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning. *arXiv:2409.05556*.
22. Ran Gilad-Bachrach et al. 2016. Cryptonets: Applying neural networks to encrypted data with high throughput and accuracy. *International Conference on Machine Learning*, 201–210.
23. Taicheng Guo et al. 2024. Large Language Model based Multi-Agents: A Survey of Progress and Challenges. *arXiv:2402.01680*.
24. Vahid Hajipour, Siavash Hekmat, and Mohammad Amini. 2023. A value-oriented Artificial Intelligence-as-a-Service business plan using integrated tools and services. *Decision Analytics Journal* 8, 100302.
25. Keegan Hines et al. 2024. Defending Against Indirect Prompt Injection Attacks With Spotlighting. *arXiv:2403.14720*.
26. Sirui Hong et al. 2024. MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. *The Twelfth International Conference on Learning Representations*.
27. IBM. 2024. watsonx Assistant. Accessed 2024-11-12.
28. Neel Jain et al. 2023. Baseline defenses for adversarial attacks against aligned language models. *arXiv:2309.00614*.
29. P. K. Kannan and H. Li. 2017. Digital Advertising: A Review and Future Research Directions. *Journal of Advertising*.
30. Brian Knott et al. 2021. Crypten: Secure multi-party computation meets machine learning. *Advances in Neural Information Processing Systems* 34, 4961–4973.
31. V. Kumar and W. Reinartz. 2016. *Creating Enduring Customer Value*. Wharton School Press.
32. Ao Li, Yuexiang Xie, Songze Li, et al. 2024. Agent-Oriented Planning in Multi-Agent Systems. *arXiv:2410.02189*.
33. Yi Li and Wei Xu. 2019. PrivPy: General and scalable privacy-preserving data mining. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 1299–1307.
34. Yuan Li, Yixuan Zhang, and Lichao Sun. 2023. MetaAgents: Simulating Interactions of Human Behaviors for LLM-based Task-oriented Coordination via Collaborative Generative Agents. *arXiv:2310.06500*.
35. Tian Liang et al. 2024. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. *arXiv:2305.19118*.
36. Qiqiang Lin et al. 2024. Hammer: Robust Function-Calling for On-Device Language Models via Function Masking. *arXiv:2410.04587*.
37. Yehuda Lindell. 2020. Secure multiparty computation. *Commun. ACM* 64, 1, 86–96.
38. Wei Liu et al. 2024. Autonomous Agents for Collaborative Task under Information Asymmetry. *arXiv:2406.14928*.
39. Yi Liu et al. 2023. Prompt Injection attack against LLM-integrated Applications. *arXiv:2306.05499*.
40. Ryan Lowe et al. 2017. Multi-agent actor-critic for mixed cooperative-competitive environments. *Advances in Neural Information Processing Systems* 30.
41. Samuele Marro et al. 2024. A Scalable Communication Protocol for Networks of Large Language Models. *arXiv:2410.11905*.
42. John X Morris, Wenting Zhao, Justin T Chiu, Vitaly Shmatikov, and Alexander M Rush. 2023. Language model inversion. *arXiv:2311.13647*.
43. Charles O'Neill et al. 2023. Adversarial Fine-Tuning of Language Models. *arXiv:2308.13768*.
44. OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, et al. 2024. GPT-4 Technical Report. *arXiv:2303.08774*.
45. Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. 2023. Gorilla: Large Language Model Connected with Massive APIs. *arXiv:2305.15334*.
46. Sandro Pinto and Nuno Santos. 2019. Demystifying arm trustzone: A comprehensive survey. *ACM Computing Surveys* 51, 6, 1–36.
47. Dean A Pomerleau. 1991. Efficient training of artificial neural networks for autonomous navigation. *Neural Computation* 3, 1, 88–97.
48. Chen Qian et al. 2023. ChatDev: Communicative Agents for Software Development. *arXiv:2307.07924*.
49. Alec Radford, Jong Wook Kim, Chris Hallacy, et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. *arXiv:2103.00020*.
50. Rafael Rafailov et al. 2024. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems* 36.
51. Saman Rajaei. 2024. Multi-Agent-as-a-Service — A Senior Engineer's Overview. Towards Data Science.
52. Tabish Rashid et al. 2020. Monotonic value function factorisation for deep multi-agent reinforcement learning. *Journal of Machine Learning Research* 21, 178, 1–51.
53. Alexander Robey, Eric Wong, Hamed Hassani, and George J Pappas. 2023. Smoothllm: Defending large language models against jailbreaking attacks. *arXiv:2310.03684*.
54. Timo Schick et al. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. *arXiv:2302.04761*.
55. AMD Sev-Snp. 2020. Strengthening VM isolation with integrity protection and more. White Paper.
56. Zeyang Sha and Yang Zhang. 2024. Prompt stealing attacks against large language models. *arXiv:2402.12959*.
57. Lloyd Shapley. 1953. A Value for n-Person Games. In *Contributions to the Theory of Games*, Vol. II. Princeton University Press, 307–317.
58. Yoav Shoham and Kevin Leyton-Brown. 2009. *Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations*.
59. David Sjödin, Vinit Parida, Maximilian Palmié, and Joakim Wincent. 2021. How AI capabilities enable business model innovation. *Journal of Business Research* 134, 574–587.
60. Sainbayar Sukhbaatar, Rob Fergus, et al. 2016. Learning multiagent communication with backpropagation. *Advances in Neural Information Processing Systems* 29.
61. Ming Tan. 1993. Multi-agent reinforcement learning: Independent vs. cooperative agents. *Proceedings of the Tenth International Conference on Machine Learning*, 330–337.
62. Zheng Tian, Ying Wen, Zhichen Gong, et al. 2019. A regularized opponent model with maximum entropy objective. *Proceedings of the 28th International Joint Conference on Artificial Intelligence*, 602–608.
63. Patara Trirat, Wonyong Jeong, and Sung Ju Hwang. 2024. AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML. *arXiv:2410.02958*.
64. VentureBeat. 2024. Microsoft quietly assembles the largest AI agent ecosystem — and no one else is close. Accessed 2024-11-21.
65. Jun Wang et al. 2024. OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models. *arXiv:2410.09671*.
66. Jason Wei et al. 2023. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *arXiv:2201.11903*.
67. Muning Wen, Jakub Kuba, Runji Lin, et al. 2022. Multi-agent reinforcement learning is a sequence modeling problem. *Advances in Neural Information Processing Systems* 35, 16509–16521.
68. Michael Wooldridge. 2009. *An Introduction to MultiAgent Systems*.
69. Qingyun Wu et al. 2023. Autogen: Enabling next-gen LLM applications via multi-agent conversation framework. *arXiv:2308.08155*.
70. Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, and Leo Schwinn. 2024. Efficient adversarial training in LLMs with continuous attacks. *arXiv:2405.15589*.
71. Tianbao Xie et al. 2023. Openagents: An open platform for language agents in the wild. *arXiv:2310.10634*.
72. Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, and Qian Lou. 2024. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models. *arXiv:2406.00083*.
73. Jun Yan et al. 2024. Backdooring instruction-tuned large language models with virtual prompt injection. *Proceedings of NAACL 2024 (Volume 1)*, 6065–6086.
74. Shunyu Yao, Jeffrey Zhao, Dian Yu, et al. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv:2210.03629*.
75. Chaoyun Zhang et al. 2024. Large Language Model-Brained GUI Agents: A Survey. *arXiv:2411.18279*.
76. Weinan Zhang, Junwei Liao, Ning Li, and Kounianhua Du. 2024. Agentic Information Retrieval. *arXiv:2410.09713*.
77. Zeyu Zhang, Xiaohe Bo, Chen Ma, et al. 2024. A Survey on the Memory Mechanism of Large Language Model based Agents. *arXiv:2404.13501*.
78. Ruiwen Zhou, Yingxuan Yang, Muning Wen, et al. 2024. TRAD: Enhancing LLM Agents with Step-Wise Thought Retrieval and Aligned Decision. *Proceedings of SIGIR '24*, 3–13.
79. Wangchunshu Zhou et al. 2024. Symbolic Learning Enables Self-Evolving Agents. *arXiv:2406.18532*.
80. Mingchen Zhuge, Wenyi Wang, Louis Kirsch, et al. GPTSwarm: Language Agents as Optimizable Graphs. *Forty-first International Conference on Machine Learning*.
81. Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. *arXiv:2307.15043*.
