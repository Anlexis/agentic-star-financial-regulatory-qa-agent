"""AgentCore Platform v1.0"""

# FIN-C2-001 — DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the financial regulatory RAG Q&A pipeline:
#
#   START
#     → input_validate   (InputValidateNode)
#     → retrieve         (RetrieveNode)
#     → rerank_filter    (RerankFilterNode)
#     → generate_answer  (GenerateAnswerNode — grounded synthesis)
#     → output_format    (OutputFormatNode)
#     → END
#
# Called by RegulatoryQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   ✅ Inherits BaseGraph (fully custom topology — no forced backbone)
#   ✅ Implements all 7 BaseGraph ABC methods
#   ✅ register_nodes() does NOT call super() (abstract in BaseGraph)
#   ✅ Does NOT register initialize / finalize (outer backbone concerns)
#   ✅ All inner nodes declare required_trust_level = TrustLevel.ANONYMOUS
#   ✅ get_output() designed together with RegulatoryQAGraphNode.merge_output()
#   ✅ Static config reaches the domain nodes through their constructors
#      (the node contract is execute(self, state) -> dict — state only)
#   ✅ _extra_initial_state() seeds the caller's input_context (context bridge)
#   ❌ No platform-SDK imports (framework/ and shared/ only)
#   ❌ Not placed under src/subagents/

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for FIN-C2-001.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by RegulatoryQAGraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
        START
          → input_validate   (InputValidateNode)
          → retrieve         (RetrieveNode)
          → rerank_filter    (RerankFilterNode)
          → generate_answer  (GenerateAnswerNode — grounded synthesis)
          → output_format    (OutputFormatNode)
          → END

    All nodes are FunctionNode subclasses with ANONYMOUS trust_level.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "fin_c2_001_regulatory_qa_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory config for the deterministic RAG inner graph.

        The settings forwarded by RegulatoryQAGraphNode._parent_config() are
        already type/range-validated there; absent keys mean node defaults.
        """
        pass

    # ── Caller-context bridge ─────────────────────────────────────────────────

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the inner state with the caller's input_context.

        GraphNode.execute() (framework, SDK 1.0.1) does not forward the outer
        state's input_context into subgraph.invoke(), so the outer graph
        stashes it in a ContextVar (RegulatoryQAGraphNode.extract_input) and
        this hook reads it back — see src/graph/context_bridge.py. Without
        this, inner-node reads of state["input_context"] (the per-invocation
        retrieval override) would always see {}.
        """
        return {"input_context": get_caller_input_context()}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.
        Every key registered here is referenced in add_edges().

        Config injection: the node contract is `execute(self, state) -> dict`
        — no per-invocation config parameter.  The two config-driven nodes
        therefore receive the declared settings (read from config/config.yaml
        and forwarded by RegulatoryQAGraphNode._parent_config() into this
        graph's `config`) through their constructors.  When the graph is built
        standalone (`DomainWorkflowGraph()`), `self.config` is empty and both
        nodes fall back to their module defaults.
        """
        settings = self.config or {}

        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode(config=settings)
        self._nodes["rerank_filter"] = RerankFilterNode(config=settings)
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear RAG Q&A topology.

        Linear flow:
            input_validate → retrieve → rerank_filter → generate_answer
            → output_format → END.

        No conditional branching — all paths through the RAG pipeline are
        linear.  route() satisfies the ABC but is not used at runtime.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by the BaseGraph ABC.

        Linear topology; add_conditional_edges() is not used, so this method
        is never called at runtime.  Returns END on error so an unexpected
        invocation does not re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by RegulatoryQAGraphNode.merge_output()
        in graph.py as the `sub_result` argument.  Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output() emits:   "answer", "citations", "qa_output", "status"
            Outer merge_output() reads: sub_result.get(...) for each key above.
        """
        return {
            "answer": state.get("answer"),
            "citations": state.get("citations"),
            "qa_output": state.get("qa_output"),
            "status": state.get("status"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
