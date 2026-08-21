---
id: fm-architecture
status: final
---

## The shape of the book

Twenty-eight parts, 240 chapters, fifteen projects, a six-chapter capstone, and
fifteen appendices. The ordering is not arbitrary and it is not a taxonomy: it
is a dependency order. Each part is placed at the earliest point where every
concept it needs has already been established.

That constraint is enforced mechanically. Every chapter declares the concepts it
`requires` and the concepts it `provides`, and the build fails if a chapter
depends on one introduced later. The graphs below are the human-readable view of
that same dependency structure.

## Part dependency graph

Solid arrows are hard prerequisites: the target is not readable without the
source. Dashed arrows are soft — the target is clearer with the source, but
stands on its own.

```mermaid {#fig:part-graph caption="Hard and soft dependencies between the twenty-eight parts. Reading order is a topological sort of this graph; the numbered sequence in the table of contents is one such sort."}
graph TD
  I[I · Mathematics] --> IV[IV · Classical ML]
  I --> VI[VI · Deep Learning]
  II[II · Python] --> III[III · Data Science]
  II --> IV
  III --> IV
  IV --> V[V · ML Engineering]
  IV --> VI
  V -.-> XXIV[XXIV · MLOps / LLMOps]
  VI --> VII[VII · Transformers]
  VI --> XIII[XIII · Multimodal]
  VII --> VIII[VIII · NLP]
  VII --> IX[IX · Foundation Models]
  VII --> XIII
  VIII --> XI[XI · Embeddings]
  IX --> X[X · LLMs]
  X --> XII[XII · RAG]
  X --> XIV[XIV · Fine-Tuning]
  X --> XV[XV · Quantization]
  X --> XVI[XVI · Reasoning]
  X --> XVII[XVII · Agents]
  XI --> XII
  XII --> XVIII[XVIII · Agentic Systems]
  XIII --> XII
  XVI --> XVII
  XVII --> XVIII
  XVII --> XIX[XIX · MCP]
  XVIII --> XX[XX · AI for Data Science]
  XVIII --> XXI[XXI · AI Coding]
  XIX --> XVIII
  XII --> XXII[XXII · System Design]
  XVIII --> XXII
  XV --> XXIII[XXIII · Serving & Infra]
  XXII --> XXIII
  XXII --> XXIV
  XXIII --> XXIV
  XXIV --> XXV[XXV · Evaluation]
  XVII --> XXVI[XXVI · Security]
  XXV --> XXVI
  XXVI --> XXVII[XXVII · Responsible AI]
  XXV -.-> XXVIII[XXVIII · Research]
  XXVII -.-> XXVIII
  IX -.-> XXVIII
```

Three properties of that graph are worth naming, because they explain the book's
structure better than the list of parts does.

**Part VII is the articulation point.** Remove it and the graph falls into two
disconnected halves. Everything before it is the general theory of learning from
data; everything after it is a consequence of one architecture. This is why the
book refuses to let you skip it.

**Evaluation comes late but applies throughout.** Part XXV depends on almost
everything, because evaluating a retrieval system, a reasoning model and an
agent are genuinely different problems requiring their own vocabulary. The cost
of that placement is that Parts XII through XXI each have to discuss evaluation
locally before the general treatment arrives. That duplication is deliberate:
the alternative — a rigorous evaluation part early, before you have anything to
evaluate — teaches nothing.

**Security depends on agents, not the reverse.** {{maturity:ESTABLISHED}} The
threat model of a system that only generates text and the threat model of a
system that can execute tools are not the same problem, and treating security
before Part XVII would mean treating only the smaller one.

## Concept flow through the book

The parts organise the material. The concepts themselves flow differently — a
handful of ideas are introduced early and then reappear, deepened, throughout.

```mermaid {#fig:concept-spine caption="The five concept spines. Each is introduced once, then reinterpreted in progressively more demanding settings rather than re-taught."}
graph LR
  subgraph OPT[Optimisation]
    A1[gradient descent<br/>Part I] --> A2[backpropagation<br/>Part VI] --> A3[Adam, schedules<br/>Part VI] --> A4[pretraining at scale<br/>Part IX] --> A5[preference optimisation<br/>Part IX, XIV]
  end
  subgraph REP[Representation]
    B1[vectors, norms<br/>Part I] --> B2[features<br/>Part III] --> B3[learned features<br/>Part VI] --> B4[contextual embeddings<br/>Part VIII] --> B5[retrieval geometry<br/>Part XI] --> B6[multimodal alignment<br/>Part XIII]
  end
  subgraph ATT[Attention]
    C1[dot products<br/>Part I] --> C2[soft alignment<br/>Part VII] --> C3[KV cache<br/>Part VII, X] --> C4[serving economics<br/>Part XXIII]
  end
  subgraph UNC[Uncertainty]
    D1[probability<br/>Part I] --> D2[inference, A/B<br/>Part III] --> D3[calibration<br/>Part IV] --> D4[hallucination<br/>Part X] --> D5[LLM-as-judge<br/>Part XXV]
  end
  subgraph SYS[Systems]
    E1[pipelines<br/>Part V] --> E2[drift<br/>Part V] --> E3[agent state<br/>Part XVII] --> E4[architecture<br/>Part XXII] --> E5[observability<br/>Part XXIV]
  end
```

When a later chapter says a concept is "revisited", it means this: the
definition does not change, but the setting is harder and the earlier
simplifications are now paid for.

## Recommended learning sequence

The default sequence is the book's own order. Two variants are worth stating
explicitly.

**The complete sequence** — 24 to 30 months at two chapters a week, including
projects. Parts I through XXVIII in order, with each project built when it
appears rather than deferred. This is the sequence the dependency graph was
designed for and the only one where nothing is ever taken on faith.

**The practitioner sequence** — roughly 12 months for someone already writing
production Python. Parts I and II as reference rather than reading; Part III
skimmed; Parts IV, V and VI read properly, because the classical material is
where evaluation discipline and the bias-variance trade-off are actually
learned; then Parts VII onward in full. The risk of this route is a reader who
can build a Transformer but cannot diagnose an overfitting curve, which is a
real and common gap.

{#tbl:sequence caption="Estimated effort by part group, at roughly ten hours per week. Project time is the dominant term and is where most of the learning happens."}

| Parts | Material | Chapters | Reading | Projects | Total |
|---|---|---|---|---|---|
| I–II | Mathematics, Python | 20 | 6 weeks | — | 6 weeks |
| III–V | Data science, classical ML, ML engineering | 28 | 10 weeks | 2 projects, 4 weeks | 14 weeks |
| VI | Deep learning | 13 | 6 weeks | 1 project, 2 weeks | 8 weeks |
| VII–VIII | Transformers, NLP | 17 | 8 weeks | 1 project, 3 weeks | 11 weeks |
| IX–X | Foundation models, LLMs | 20 | 9 weeks | 1 project, 3 weeks | 12 weeks |
| XI–XII | Embeddings, RAG | 19 | 9 weeks | 1 project, 4 weeks | 13 weeks |
| XIII–XV | Multimodal, fine-tuning, quantization | 28 | 12 weeks | 3 projects, 7 weeks | 19 weeks |
| XVI–XIX | Reasoning, agents, agentic systems, MCP | 31 | 14 weeks | 3 projects, 8 weeks | 22 weeks |
| XX–XXIV | Applied AI, system design, infrastructure, ops | 35 | 15 weeks | 2 projects, 6 weeks | 21 weeks |
| XXV–XXVIII | Evaluation, security, responsible AI, research | 29 | 12 weeks | 1 project, 3 weeks | 15 weeks |
| Capstone | Multimodal agentic platform | 6 | 3 weeks | 8 weeks | 11 weeks |

## Difficulty progression

Difficulty is not monotonic, and pretending otherwise sets readers up to
conclude they have hit a wall when they have merely hit Part VI.

```mermaid {#fig:difficulty caption="Relative difficulty by part. The two genuine step changes are backpropagation in Part VI and the shift from single-component reasoning to distributed-system reasoning in Part XXII."}
graph LR
  P1["I–III<br/><b>Foundational</b><br/>slow but not hard"] --> P2["IV–V<br/><b>Moderate</b><br/>many ideas, each small"]
  P2 --> P3["VI<br/><b>First step change</b><br/>backpropagation"]
  P3 --> P4["VII–VIII<br/><b>Hard, then easier</b><br/>attention rewards effort"]
  P4 --> P5["IX–XII<br/><b>Moderate</b><br/>conceptually clear"]
  P5 --> P6["XIII–XVI<br/><b>Uneven</b><br/>breadth, not depth"]
  P6 --> P7["XVII–XIX<br/><b>Moderate</b><br/>design over theory"]
  P7 --> P8["XX–XXIV<br/><b>Second step change</b><br/>systems thinking"]
  P8 --> P9["XXV–XXVII<br/><b>Subtle</b><br/>easy to read, hard to do"]
  P9 --> P10["XXVIII<br/><b>Open</b><br/>no settled answers"]
```

> IMPORTANT: Part VI, Chapter 53 (backpropagation derived from scratch) is the
> single hardest point in the first half of the book, and readers who stall
> usually stall there. It is also the chapter that makes the following two
> hundred pages feel obvious. Budget a week for it, and implement it before
> moving on — not after.

The second step change, at Part XXII, catches a different reader. The material
is not mathematically harder; it is harder because correctness stops being a
property of a component and becomes a property of a system under load, partial
failure, and adversarial input. Readers strong in ML and weak in distributed
systems find this harder than Part VI.

## Technology stack

The book uses a deliberately small stack. Every choice below is a teaching
choice, and each is introduced only after the concept beneath it.

{#tbl:stack caption="The stack, and where each piece first appears. Frameworks are always preceded by a from-scratch implementation of what they abstract."}

| Layer | Tool | First used | Why this one |
|---|---|---|---|
| Language | Python 3.12+ | Part II | Where the ecosystem is |
| Numerics | NumPy | Part II | The array semantics every other library inherits |
| Data | Pandas | Part II | Dataframe semantics, including the sharp edges |
| Classical ML | scikit-learn | Part IV | Consistent API, honest defaults |
| Deep learning | PyTorch | Part VI | Explicit control flow; easiest to read the internals of |
| Models and datasets | Hugging Face Transformers, Datasets | Part VIII | The de facto distribution layer |
| Vector search | Qdrant | Part XI | Payload filtering and hybrid search without extra services |
| Sparse retrieval | BM25 (from scratch, then a library) | Part XI | The baseline dense retrieval must beat |
| APIs | FastAPI | Part XXII | Async-native, typed, generates its own schema |
| Agent orchestration | LangGraph | Part XVIII | Explicit graph state, after building the loop by hand |
| Tool protocol | MCP | Part XIX | The interoperability layer, built from the spec |
| Local inference | llama.cpp, Ollama, MLX | Part XV | Three different answers to the same constraint |
| Serving | vLLM | Part XXIII | Continuous batching and paged attention in the open |
| Containers | Docker, Kubernetes | Part XXIII | What production actually runs on |

> NOTE: Framework versions move faster than a book can. Pinned versions live in
> the repository's requirements file, not in the prose; chapters teach the
> concept and the shape of the API, so that a version bump costs you a
> documentation lookup rather than a re-read.

## Research-paper reading strategy

The book cites papers because at some point you will need to read them without
an intermediary. That is a skill, and it is trainable.

**The three-pass method.** First pass, five minutes: title, abstract,
introduction, section headings, figures, conclusion. You are answering one
question — is this paper relevant to me? Second pass, an hour: read the body,
skip proofs, study every figure and table, and mark what you do not understand.
You should now be able to summarise the contribution to someone else. Third
pass, several hours, and only for papers that matter to you: reconstruct the
work. Re-derive the equations. Predict what the ablation table will show before
you read it. You have understood a paper when you can identify what its authors
got wrong or left undone.

**Read in dependency order, not publication order.** Reading the Transformer
paper before the attention papers it responds to produces a description you can
recite. Reading Bahdanau, then Luong, then Vaswani produces an understanding of
why each design decision was made, because you have seen the problem each one
solved. The annotated bibliography records, for each work, what preceded it and
what descends from it, so the chains are visible.

**Read the ablations before the headline.** A paper's main result tells you what
the authors want you to believe. The ablation table tells you which components
actually carry the result. When the two disagree, the ablation table is more
informative.

**Distrust single-paper results.** {{maturity:ESTABLISHED}} A result that has
not been independently reproduced is a hypothesis. This is not cynicism; it is
the reason maturity labels exist in this book.

{#tbl:paper-schedule caption="Paper-reading load by part group. The book's chapters cover the content; reading the originals develops the skill of reading originals."}

| Parts | Papers to read in full | Emphasis |
|---|---|---|
| I–V | 0–2 | Concentrate on the material; papers come later |
| VI | 4 | Backpropagation, ResNet, Adam, batch normalisation |
| VII–VIII | 6 | The attention lineage, in order, then BERT |
| IX–X | 8 | Scaling laws, instruction tuning, RLHF, DPO |
| XI–XII | 5 | Dense retrieval, RAG, reranking, GraphRAG |
| XIII–XVI | 8 | CLIP, ViT, LoRA, quantization, reasoning |
| XVII–XIX | 5 | ReAct, tool use, the MCP specification |
| XX–XXVIII | 12 | Chosen by the reader's direction |

## Project progression

Fifteen projects, each placed immediately after the part that makes it possible,
and each deliberately harder than the last along one specific axis.

```mermaid {#fig:projects caption="The fifteen projects and the capstone. Each arrow means the later project reuses code, infrastructure, or evaluation harness from the earlier one."}
graph TD
  P1[1 · Tabular prediction<br/><i>after Part V</i>] --> P2[2 · Data science pipeline<br/><i>after Part V</i>]
  P2 --> P3[3 · Image classifier<br/><i>after Part VI</i>]
  P3 --> P4[4 · Transformer from scratch<br/><i>after Part VII</i>]
  P4 --> P5[5 · Small language model<br/><i>after Part X</i>]
  P5 --> P8[8 · Fine-tuned LLM<br/><i>after Part XIV</i>]
  P5 --> P9[9 · Local inference<br/><i>after Part XV</i>]
  P6[6 · Production RAG<br/><i>after Part XII</i>] --> P7[7 · Document intelligence<br/><i>after Part XIII</i>]
  P6 --> P10[10 · Single agent<br/><i>after Part XVII</i>]
  P10 --> P11[11 · Multi-agent system<br/><i>after Part XVIII</i>]
  P11 --> P12[12 · MCP tool ecosystem<br/><i>after Part XIX</i>]
  P12 --> P13[13 · AI data scientist<br/><i>after Part XX</i>]
  P9 --> P14[14 · Production platform<br/><i>after Part XXIV</i>]
  P13 --> P14
  P7 --> P14
  P14 --> P15[15 · Evaluation harness<br/><i>after Part XXV</i>]
  P15 --> CAP[Capstone · Multimodal agentic<br/>data and knowledge platform]
  P8 --> CAP
```

Every project specifies requirements, architecture, dataset, folder structure,
implementation, testing, evaluation, deployment, monitoring, and extensions.
They are not exercises with a solution key; they are systems with acceptance
criteria.

The capstone is not a fifteenth project. It is the integration of all of them
into one deployed system, and it is the point of the book.

## Appendices

The appendices are reference material, and several are generated from the same
data the chapters use rather than written separately — the glossary, the acronym
list, the notation table, and the annotated bibliography. This is why they
cannot contradict the text: there is only one copy of each definition.

The remaining appendices — the formula sheet, the Python reference, the five
architecture-comparison tables, the metrics reference, and the two checklists —
are written, and are best read after the corresponding parts rather than before.
