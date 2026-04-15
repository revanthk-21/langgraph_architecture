# Agentic AI Framework — Ashok Leyland
# ═══════════════════════════════════════

## Project Structure

```
agentic_framework/
├── core/
│   ├── state.py          ← Single shared AgentState TypedDict
│   ├── tool_base.py      ← BaseTool class + HOW TO ADD NEW TOOLS
│   ├── tool_registry.py  ← Central registry of all tools
│   └── llm.py            ← AWS Bedrock LLM + Embeddings clients
│
├── tools/
│   ├── rag/
│   │   ├── embed_query.py      ← Titan Embeddings
│   │   ├── retrieve_docs.py    ← FAISS MMR retrieval
│   │   ├── grade_relevance.py  ← Confidence scoring
│   │   ├── generate_answer.py  ← Claude Sonnet generation
│   │   └── rewrite_query.py    ← Query rewriting for re-retrieval
│   │
│   ├── dfmea/
│   │   ├── case_router.py      ← Route new|new_env|design_change
│   │   ├── rag_context.py      ← RAG retrieval for DFMEA enrichment ★
│   │   ├── parse_import.py     ← Wraps your existing xlsx parser
│   │   ├── generate_elements.py  ← Steps 1,4,7,8,9 + assemble + export
│   │   └── ...
│   │
│   └── optimizer/
│       ├── solve_ode.py        ← Quarter-car RK45 integration
│       ├── compute_rms.py      ← RMS cabin acceleration
│       ├── propose_k.py        ← Optuna TPE sampler
│       ├── check_convergence.py
│       └── summarize.py        ← LLM engineering summary
│
├── agents/
│   └── graph.py          ← THE GRAPH: all nodes + edges in one place
│
├── api/
│   └── main.py           ← FastAPI endpoints with SSE streaming
│
└── requirements.txt
```

## How RAG integrates into DFMEA

The `dfmea_rag_context` node runs BEFORE any generation step:

```
case_router → [parse_import] → dfmea_rag_context → generate_elements → ...
                                        ↑
                              Retrieves from FAISS:
                              - Known failure modes for this component
                              - Historical S/O/D ratings
                              - Relevant AIAG-VDA guidelines

The retrieved context is stored in state["dfmea_rag_context"]
and threaded into EVERY downstream prompt automatically.
```

## How to Add a New Tool

1. Create `tools/<domain>/my_tool.py` inheriting `BaseTool`
2. Implement `run()` or `async arun()` — takes state, returns partial dict
3. Register in `core/tool_registry.py`
4. Add `graph.add_node(...)` and `graph.add_edge(...)` in `agents/graph.py`

See `core/tool_base.py` for the full template with examples.

## Running

```bash
# Install
pip install -r requirements.txt

# Set env vars
export AWS_DEFAULT_REGION=ap-south-1
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...

# Run
uvicorn api.main:app --reload --port 8000
```

## API Examples

```bash
# RAG
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are common failure modes for coil springs?"}'

# DFMEA — new design
curl -X POST http://localhost:8000/api/dfmea \
  -d '{"component":"front coil spring","subsystem":"suspension","case":"new"}'

# DFMEA — import existing, new environment
curl -X POST http://localhost:8000/api/dfmea \
  -d '{"component":"coil spring","subsystem":"suspension","case":"new_env","import_path":"data/existing_dfmea.xlsx"}'

# Optimize spring stiffness
curl -X POST http://localhost:8000/api/optimize \
  -d '{"k_bounds":[5000,50000],"ode_params":{"ms":400,"mu":45,"c":1500,"kt":150000}}'
```
