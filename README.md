# AgenticRAG API & Usage Guide

This repository contains three LangGraph-based Retrieval-Augmented Generation (RAG) workflows plus two light-weight Python modules. The goal of this document is to provide a single reference for every public function, class, and component so you can quickly decide which building block to reuse or extend and understand how to run it end to end.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `main.py` | Minimal CLI entry point that currently prints a health-check message. |
| `openai_agent.py` | Stand-alone LangGraph agent wired to OpenAI/ LangSmith credentials. |
| `1-AgenticRAG.ipynb` | Notebook that builds a multi-tool agent workflow over LangChain and LangGraph docs. |
| `2-CorrectiveRAG.ipynb` | Notebook that implements a corrective RAG graph with automatic question rewriting and web search. |
| `4-AdaptiveRAG.ipynb` | Notebook that routes between web search and multiple internal retrievers, and validates hallucinations. |
| `pyproject.toml`, `uv.lock` | Dependency management via `uv`. |

---

## Setup

### 1. Install dependencies

Use [`uv`](https://github.com/astral-sh/uv) (already configured):

```bash
uv sync
```

Run commands with the managed environment:

```bash
uv run python main.py
```

### 2. Required environment variables

Create a `.env` file (loaded with `python-dotenv`) and populate the keys referenced throughout the repo.

| Variable | Used By | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | All notebooks, `openai_agent.py` | ChatGPT models & OpenAI embeddings. |
| `LANGCHAIN_API_KEY` | `openai_agent.py` | LangSmith tracing (exported as `LANGSMITH_API_KEY`). |
| `GROQ_API_KEY` | `1-AgenticRAG.ipynb`, `4-AdaptiveRAG.ipynb` | Access to Groq-hosted `qwen3-32b`. |
| `TAVILY_API_KEY` | `2-CorrectiveRAG.ipynb`, `4-AdaptiveRAG.ipynb` | Tavily web search tool. |
| `CONFLUENCE_URL`, `CONFLUENCE_EMAIL`, `CONFLUENCE_API_KEY`, `CONFLUENCE_SPACE` | `4-AdaptiveRAG.ipynb` | Pulls internal documents through the Confluence loader. |

Optional but recommended: `GROQ_API_KEY`, `TAVILY_API_KEY`, and the Confluence credentials can be omitted if you do not intend to run the related workflows.

### 3. Launching notebooks

```bash
uv run jupyter lab
```

Open any of the notebooks to execute the cells sequentially. Each notebook finishes by compiling a LangGraph and demonstrating `.invoke(...)`.

---

## Running the shipped modules

### CLI entry point (`main.py`)

```bash
uv run python main.py
```

Output:

```
Hello from agenticrag!
```

Use this as a placeholder smoke test or replace `main()` with richer CLI logic.

### Loading the OpenAI LangGraph agent

```python
from langchain_core.messages import HumanMessage
from openai_agent import make_alternative_graph

agent = make_alternative_graph()
result = agent.invoke({"messages": [HumanMessage(content="Add 2 and 3")]} )
print(result["messages"][-1].content)
```

The helper automatically binds the `add(a, b)` tool and returns a compiled LangGraph. Pass an ordered list of `langchain_core.messages.BaseMessage` instances under the `messages` key.

---

## Public Python Modules

### `main.py`

| Symbol | Type | Description | Example |
| --- | --- | --- | --- |
| `main()` | Function | Prints a heartbeat message. You can expand it into a CLI runner. | `if __name__ == "__main__": main()` |

### `openai_agent.py`

| Symbol | Type | Description | Notes |
| --- | --- | --- | --- |
| `State` | `TypedDict` | Defines the LangGraph state with a `messages` list. Uses `add_messages` to append updates rather than replace them. | Inputs to graph functions must be `list[BaseMessage]`. |
| `make_default_graph()` | Function | Builds the simplest LangGraph: `START -> agent -> END`, where `agent` calls the default `ChatOpenAI` model. | Use when you do not need tool-calling. |
| `make_alternative_graph()` | Function | Builds a tool-enabled graph. Adds the `add(a, b)` tool, uses `ToolNode`, and introduces a conditional edge that loops through the tool node until no tool calls remain. | Returns a compiled graph ready for `.invoke(...)`. |
| `agent` | Variable | Result of `make_alternative_graph()` executed at import time for convenience. | Importing `openai_agent` will immediately load environment variables and instantiate this graph. |

#### Usage patterns

```python
from langchain_core.messages import HumanMessage
from openai_agent import make_default_graph, agent

default_graph = make_default_graph()
default_graph.invoke({"messages": [HumanMessage(content="Summarize LangGraph.")]})

# The pre-built tool agent:
agent.invoke({"messages": [HumanMessage(content="Use the add tool on 5 and 7") ]})
```

---

## Notebook Workflows & APIs

> Each notebook is self-contained. Run the ingestion cells first (loaders, splitters, FAISS indexes) and then execute the LangGraph cells. The functions listed below serve as the public APIs within each workflow.

### 1. `1-AgenticRAG.ipynb` — Multi-tool Agentic RAG

**Purpose:** Compare LangGraph and LangChain documentation using two retriever tools and a Groq LLM planner.

**Key setup cells:**
- Loads docs from LangGraph and LangChain tutorial URLs.
- Splits and stores them in FAISS vectorstores.
- Wraps both retrievers with `create_retriever_tool`, producing `retriever_vector_db_blog` and `retriever_vector_langchain_blog`.

**State schema:** `AgentState` (`messages: Sequence[BaseMessage]` with `add_messages` aggregation).

| Symbol | Kind | Input State | Output | Description / Notes |
| --- | --- | --- | --- | --- |
| `agent(state)` | Function | `messages` | `{"messages": [response]}` | Groq `ChatGroq` (`qwen/qwen3-32b`) bound to both retriever tools. Decides whether to call tools or respond directly. |
| `grade_documents(state)` | Function | `messages` with last entry containing retrieved docs | `"generate"` or `"rewrite"` | Uses `PromptTemplate` + Groq model to judge doc relevance. |
| `generate(state)` | Function | `messages` (question + retrieved docs) | `{"messages": [answer_text]}` | Pulls `rlm/rag-prompt` from LangChain Hub, pipes through Groq, returns answer. |
| `rewrite(state)` | Function | `messages` | `{"messages": [HumanMessage]}` | Asks Groq to reformulate the user question when retrieval fails. |
| `workflow` / `graph` | LangGraph | — | Compiled graph | Topology: `START -> agent -> (END or ToolNode) -> grade_documents -> generate|rewrite`. |

**Example invocation inside the notebook:**

```python
graph.invoke({"messages": "What is LangGraph?"})
```

For multi-turn usage replace the string with a list of `HumanMessage`/`AIMessage` objects.

### 2. `2-CorrectiveRAG.ipynb` — Corrective Retrieval Graph

**Purpose:** Demonstrate RAG with document grading, question rewriting, and optional Tavily web search when retrieved docs are irrelevant.

**Data/Tools:**
- Loads Lilian Weng blog posts, splits with `RecursiveCharacterTextSplitter`, indexes in FAISS.
- `TavilySearchResults` adds open-web fallback.
- `ChatOpenAI` (GPT-3.5) powers grading, rewriting, and generation.

**State schema:** `GraphState` typed dict with `question`, `generation`, `web_search`, and `documents`.

| Symbol | Kind | Input State | Output | Description |
| --- | --- | --- | --- | --- |
| `retrieve(state)` | Function | `question` | Adds `documents` | Pulls top chunks from FAISS retriever. |
| `grade_documents(state)` | Function | `question`, `documents` | Filters `documents`, sets `web_search` flag | Uses `GradeDocuments` Pydantic model + structured OpenAI call. |
| `decide_to_generate(state)` | Function | `web_search`, `documents` | `"generate"` or `"transform_query"` | Routes to rewriting if everything was filtered out. |
| `transform_query(state)` | Function | `question` | Rewrites `question` | `question_rewriter` chain optimizes for future retrieval or search. |
| `web_search(state)` | Function | `question` | Appends Tavily hits to `documents` | Called when `web_search == "Yes"`. |
| `generate(state)` | Function | `question`, `documents` | Adds `generation` | Uses `rlm/rag-prompt` + GPT-3.5 to craft answers. |
| `workflow` / `app` | LangGraph | — | Compiled | Flow: `START -> retrieve -> grade -> (generate | transform_query -> web_search -> generate) -> END`. |

**Example invocation:**

```python
app.invoke({"question": "What are the types of agent memory?"})
```

Inspect `result["generation"]` for the final answer and `result["documents"]` for supporting context.

### 4. `4-AdaptiveRAG.ipynb` — Adaptive Routing & Validation

**Purpose:** Combine multiple enterprise knowledge sources with web search while validating generations against hallucinations and answer quality.

**Data/Tools:**
- Three retriever tools: Technia marketing site, Helium docs, and Helium Confluence space.
- Tavily web search fallback.
- Routing (`RouteQuery`) decides between vectorstore or web search per query.
- Graders: `GradeDocuments`, `GradeHallucinations`, `GradeAnswer` (each exposes a `binary_score` field used in downstream branching).
- Question rewriter optimized for vectorstore retrieval.

**State schema:** `GraphState` with `question`, `generation`, `documents`.

| Symbol | Kind | Input | Output | Description |
| --- | --- | --- | --- | --- |
| `question_router` | Chain | `question` | `datasource` (`vectorstore`/`web_search`) | Structured output wrapper around `ChatOpenAI`. |
| `route_question(state)` | Function | `question` | `"web_search"` or `"vectorstore"` | Uses `question_router` to decide the first hop in the graph. |
| `retrieve(state)` | Function | `question` | Aggregated `documents` | Invokes each retriever tool and concatenates their results. |
| `grade_documents(state)` | Function | `question`, `documents` | Filters docs | Removes irrelevant chunks using `retrieval_grader`. |
| `decide_to_generate(state)` | Function | filtered `documents` | `"generate"` or `"transform_query"` | Ensures the graph only generates when relevant evidence remains. |
| `transform_query(state)` | Function | `question` | Rewritten `question` | Uses the vectorstore-focused prompt. |
| `web_search(state)` | Function | `question` | `documents` containing Tavily results | Triggered when the router selects `web_search`. |
| `generate(state)` | Function | `question`, `documents` | Adds `generation` | Uses `rag_chain` with `gpt-4o-mini`. |
| `grade_generation_v_documents_and_question(state)` | Function | `question`, `documents`, `generation` | `"useful"`, `"not useful"`, `"not supported"` | First checks grounding (`GradeHallucinations`), then QA alignment (`GradeAnswer`). Routes back to `generate` or `transform_query` if validation fails. |
| `agent(state)` | Function | `messages` | `{"messages": [response]}` | Optional Groq-based planner that binds the same toolset; use it if you need a conversational agent outside the adaptive graph. |
| `workflow` / `app` | LangGraph | — | Compiled graph | Flow: `START -> route_question -> (web_search -> generate or retrieve -> grade -> generate) -> validation loop -> END`. |

**Example invocation:**

```python
app.invoke({"question": "What is Helium?"})
```

Check `result["generation"]` for the validated answer. If validation fails, the graph automatically rewrites the query or regenerates until a grounded answer is produced.

---

## Shared Utilities & Helper Components

| Component | Location | Purpose |
| --- | --- | --- |
| `format_docs(docs)` | `2-CorrectiveRAG.ipynb`, `4-AdaptiveRAG.ipynb` | Concatenates document text for prompting. |
| `TavilySearchResults(k=3)` | `2-CorrectiveRAG.ipynb`, `4-AdaptiveRAG.ipynb` | Adds structured web context. |
| `create_retriever_tool(...)` | All notebooks | Wraps LangChain retrievers so planners can call them as tools. |
| `ToolNode` & `tools_condition` | `1-AgenticRAG.ipynb`, `openai_agent.py` | Bridges LangGraph nodes with LangChain tools automatically. |
| `StateGraph` / `.compile()` / `.invoke()` | All workflows | LangGraph primitives for building production-ready state machines. |

---

## Extending the Project

- **Add new retrievers:** Follow the FAISS build pattern (loader → splitter → `FAISS.from_documents` → `as_retriever`) and register the tool within the relevant graph.
- **Swap models:** Replace any `ChatOpenAI`/`ChatGroq` constructors with your preferred model IDs and update the required environment variables.
- **Harden validation:** In the adaptive graph, extend `grade_generation_v_documents_and_question` with additional heuristics (e.g., citation checks) before routing to END.
- **Promote notebooks to modules:** Convert a notebook with `jupyter nbconvert --to python notebook.ipynb` and expose its functions in a package for easier importing or testing.

---

## Troubleshooting

- **Missing API keys:** All loaders and LLM calls fail fast if the corresponding key is absent. Confirm `.env` is loaded and exported before running notebooks or scripts.
- **FAISS rebuilds:** Each notebook rebuilds its FAISS index from scratch. To reuse them across sessions, persist the vectorstores with `vectorstore.save_local()` and load them later.
- **Stalled LangGraph invocations:** Verify your input matches the expected state shape (e.g., `{"messages": [HumanMessage(...)]}` for agentic graphs, or `{"question": "..."} ` for corrective/adaptive graphs).

With these references you can confidently reuse or evolve every public component in the repository. Happy building! 