"""AgentCore Platform v1.0"""

# FIN-C2-001 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — identical to Cat 1, do NOT override add_edges()):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                             ↓ (RETRY, max 3)
#                                          pre_process
#
#   `main` slot is a GraphNode subclass (RegulatoryQAGraphNode) that delegates
#   the full domain workflow to DomainWorkflowGraph (inner BaseGraph).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (multi-step topology)
#   src/graph/context_bridge.py        ← input_context hand-off (outer → inner)
#
# Rules enforced:
#   ✅ FinancialRegulatoryQAAgent inherits AgentBaseGraph (framework base class,
#      direct inheritance)
#   ✅ super().register_nodes() called first (fills initialize + finalize)
#   ✅ RegulatoryQAGraphNode assigned to self._nodes["main"]
#   ✅ PreProcessNode (VERIFIED_EXTERNAL) in pre_process slot (trust gate)
#   ✅ PostProcessNode (ANONYMOUS) in post_process slot (output gate)
#   ✅ _security_gate_output on the agent class (canonical output gate)
#   ✅ merge_output() returns only changed keys
#   ✅ class name matches config/agent.yaml class: field exactly
#   ❌ add_edges() NOT overridden on the outer graph
#   ❌ No platform-SDK imports (framework/ and shared/ only)

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.trust_level import TrustLevel
from framework.schemas.agent_state import AgentState
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode, _security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Runtime-parameter file: src/graph/graph.py -> parents[2] is the repo root.
# config/agent.yaml (the static manifest) holds only registration identity;
# every runtime parameter lives in config/config.yaml.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _runtime_config() -> dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    This is the same file the platform registry loads and passes as
    Graph(config=...); the standalone server (src/api/server.py) reads it here
    so the deployed agent and a registry-loaded agent see identical
    configuration. Returns an empty dict — never raises — when the file is
    absent, unreadable, not valid YAML, or not a mapping (the graph then runs
    on its built-in defaults).
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _config_number(value: Any, lo: float, hi: float) -> Optional[float]:
    """Validate a declared numeric setting: a real number, finite, within [lo, hi].

    Bools, strings, non-numerics, NaN/Infinity, and out-of-range values return
    None (the caller then keeps the node's built-in default). A non-finite
    threshold is the dangerous case: NaN comparisons are always False, so a
    NaN score_threshold would silently pass every low-confidence chunk.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


class RegulatoryQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of FinancialRegulatoryQAAgent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph).
    Called by the AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()  — instantiate and return DomainWorkflowGraph
      extract_input() — pull validated_input from outer state; bridge input_context
      merge_output()  — map sub_result fields into outer state delta (changed keys only)
      error_strategy  — "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    # S-1 declared on the wrapper too: the CI gate only AST-scans FunctionNode
    # subclasses, so a GraphNode main slot passes the pipeline without one and is
    # flagged at review. Same level the nodes in this repo already declare.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates and normalises the raw user_input and writes
        the result to validated_input.  Prefer that; fall back to user_input
        if validated_input is absent (e.g. in unit tests).

        Also bridges the caller's input_context to the inner graph:
        GraphNode.execute() does not forward input_context on subgraph.invoke()
        (SDK 1.0.1), and extract_input is the last hook in this repo's code
        that sees the outer state before the inner invoke — see
        src/graph/context_bridge.py.
        """
        set_caller_input_context(cast("dict[str, Any] | None", state.get("input_context")))
        return cast(str, state.get("validated_input", state.get("user_input", "")))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  → "answer", "citations", "qa_output", "status"
          This merge_output() reads → sub_result.get(...) for each of these keys.

        PostProcessNode (outer post_process) reads qa_output from state to apply
        the output gate and set formatted_output.
        """
        return {
            "answer": sub_result.get("answer"),
            "citations": sub_result.get("citations"),
            "qa_output": sub_result.get("qa_output"),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> dict[str, Any]:
        """Forward the declared runtime settings to the inner graph.

        Reads config/config.yaml (see _runtime_config) and returns a flat
        settings dict, which DomainWorkflowGraph.register_nodes() injects into
        the domain node constructors (RetrieveNode / RerankFilterNode).
        Constructor injection is the config route because the node contract is
        ``execute(self, state) -> dict`` — a node may not take a
        per-invocation config argument.

        Every forwarded value is validated here (type, finiteness, range) so a
        malformed configuration file can neither crash graph construction nor
        weaken the retrieval confidence gate: a non-finite score_threshold
        would compare False against every score and wave every low-confidence
        chunk through. Invalid or absent keys are simply not forwarded; each
        node then falls back to its module default.
        """
        cfg = _runtime_config()
        retrieval_raw = cfg.get("retrieval")
        retrieval: dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
        llm_raw = cfg.get("llm")
        llm: dict[str, Any] = llm_raw if isinstance(llm_raw, dict) else {}

        declared: dict[str, Any] = {}

        top_k = _config_number(retrieval.get("top_k"), 1, 50)
        if top_k is not None and top_k == int(top_k):
            declared["top_k"] = int(top_k)

        score_threshold = _config_number(retrieval.get("score_threshold"), 0.0, 1.0)
        if score_threshold is not None:
            declared["score_threshold"] = score_threshold

        hybrid_search = retrieval.get("hybrid_search")
        if isinstance(hybrid_search, bool):
            declared["hybrid_search"] = hybrid_search

        template = llm.get("system_prompt_template")
        if isinstance(template, str) and template:
            declared["system_prompt_template"] = template

        temperature = _config_number(llm.get("temperature"), 0.0, 2.0)
        if temperature is not None:
            declared["temperature"] = temperature

        max_tokens = _config_number(llm.get("max_tokens"), 1, 100_000)
        if max_tokens is not None and max_tokens == int(max_tokens):
            declared["max_tokens"] = int(max_tokens)

        return declared


class FinancialRegulatoryQAAgent(AgentBaseGraph):
    """Outer graph for FIN-C2-001 (Cat 2 — RAG pattern).

    Inherits AgentBaseGraph directly (framework base class). Domain logic is
    fully encapsulated in RegulatoryQAGraphNode (main slot), which delegates
    to DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed — identical to Cat 1):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process: PreProcessNode    (VERIFIED_EXTERNAL — trust gate)
      - main:        RegulatoryQAGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode  (ANONYMOUS — output gate)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.

    Runtime configuration: the platform registry loads config/config.yaml and
    passes it as Graph(config=...); the standalone server does the same via
    _runtime_config(). AgentBaseGraph itself consumes max_retry from that
    config (retry routing), so the declared value is live in both deployments.

    Class name MUST match config/agent.yaml `class:` field exactly
    (FinancialRegulatoryQAAgent) — src/api/server.py imports this class.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "FinancialRegulatoryQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = RegulatoryQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.

    def _security_gate_output(self, content: str) -> Optional[str]:
        """Canonical output-gate entry point on the agent class.

        FIN-C2-001 answers regulated-domain compliance questions: output must
        never leak credentials/secrets.  This method delegates to the shared
        credential scanner that PostProcessNode (the post_process backbone
        slot) applies at runtime, so there is a single source of truth for the
        pattern set.  Returns the first violation name, or None when the
        output is clean.
        """
        return _security_gate_output(content)
