"""
core/tool_base.py
─────────────────
HOW TO FORMAT INDIVIDUAL TOOLS IN THIS FRAMEWORK
═════════════════════════════════════════════════

Every tool in this system follows the same contract:

    Input  : AgentState  (reads only the fields it needs)
    Output : dict        (only the fields it writes — merged into state by LangGraph)

This makes tools:
  • Independently testable (just pass a dict, check the returned dict)
  • Safely composable    (no side effects on state outside declared outputs)
  • Easy to add          (register in tool_registry.py, wire an edge in graph.py)

─────────────────────────────────────────────────────────────────────────────
TOOL TEMPLATE  (copy-paste this to create a new tool)
─────────────────────────────────────────────────────────────────────────────

from core.tool_base import BaseTool, ToolResult
from core.state import AgentState

class MyNewTool(BaseTool):
    name        = "my_new_tool"
    description = "One sentence: what this tool does and when to use it."

    # Declare which state fields this tool READS and WRITES.
    # This is documentation — LangGraph doesn't enforce it, but your
    # teammates (and future-you) will thank you.
    reads  = ["field_a", "field_b"]
    writes = ["field_c"]

    def run(self, state: AgentState) -> ToolResult:
        # 1. Extract what you need
        value_a = state.get("field_a", "default")

        # 2. Do work (call LLM, run scipy, parse xlsx, etc.)
        result = value_a + "_processed"

        # 3. Return ONLY the fields you are writing.
        #    LangGraph merges this dict into the shared state.
        return {"field_c": result}

─────────────────────────────────────────────────────────────────────────────
ASYNC TOOLS  (for LLM calls — use this variant instead)
─────────────────────────────────────────────────────────────────────────────

class MyAsyncTool(BaseTool):
    name = "my_async_tool"

    async def arun(self, state: AgentState) -> ToolResult:
        result = await some_async_llm_call(state["rag_query"])
        return {"rag_answer": result}

─────────────────────────────────────────────────────────────────────────────
REGISTERING A NEW TOOL  (after writing it)
─────────────────────────────────────────────────────────────────────────────

In core/tool_registry.py:

    from tools.my_domain.my_new_tool import MyNewTool
    TOOL_REGISTRY["my_new_tool"] = MyNewTool()

In agents/graph.py, add a node and an edge:

    graph.add_node("my_new_tool", registry["my_new_tool"].as_node())
    graph.add_edge("previous_node", "my_new_tool")

That's it. The tool is live.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from core.state import AgentState

# Return type: partial state dict (only the keys being written)
ToolResult = dict[str, Any]


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    reads: list[str] = []
    writes: list[str] = []

    @abstractmethod
    def run(self, state: AgentState) -> ToolResult:
        """Synchronous execution. Override this or arun, not both."""
        ...

    async def arun(self, state: AgentState) -> ToolResult:
        """Async execution. Override for LLM / IO-bound tools."""
        return self.run(state)

    def as_node(self):
        """
        Returns a coroutine function compatible with LangGraph's add_node().
        LangGraph calls this with the current state and merges the returned dict.
        """
        async def _node(state: AgentState) -> ToolResult:
            return await self.arun(state)
        _node.__name__ = self.name
        return _node
