"""
tools/rag/retrieve_docs.py
──────────────────────────
Retrieves top-k document chunks from the FAISS vector store using
Maximal Marginal Relevance (MMR) to balance relevance and diversity.

Chunks are appended to state["rag_retrieved_docs"] via the operator.add
reducer — safe for parallel retrieval from multiple stores.
"""

from __future__ import annotations
from core.tool_base import BaseTool, ToolResult
from core.state import AgentState
from core.llm import get_embeddings
from langchain_community.vectorstores import FAISS

# ── Config ───────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "data/faiss_index"   # path to your persisted FAISS index
TOP_K            = 6                     # chunks to retrieve
FETCH_K          = 20                    # candidates before MMR filtering
LAMBDA_MULT      = 0.5                   # MMR diversity weight (0=max diversity, 1=max relevance)


class RetrieveDocsTool(BaseTool):
    name        = "retrieve_docs"
    description = "Retrieve top-k document chunks from FAISS using MMR."
    reads       = ["rag_query"]
    writes      = ["rag_retrieved_docs", "rag_sources"]

    def __init__(self):
        self._store: FAISS | None = None

    def _get_store(self) -> FAISS:
        """Lazy-load FAISS index once per process."""
        if self._store is None:
            self._store = FAISS.load_local(
                FAISS_INDEX_PATH,
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return self._store

    def run(self, state: AgentState) -> ToolResult:
        query = state["rag_query"]
        store = self._get_store()

        docs = store.max_marginal_relevance_search(
            query,
            k=TOP_K,
            fetch_k=FETCH_K,
            lambda_mult=LAMBDA_MULT,
        )

        chunks  = [d.page_content for d in docs]
        sources = [d.metadata.get("source", "unknown") for d in docs]

        return {
            "rag_retrieved_docs": chunks,   # appended via operator.add
            "rag_sources": sources,
        }
