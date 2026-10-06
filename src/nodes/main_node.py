"""AgentCore Platform v1.0"""

# FIN-C2-001 — MainNode (Cat 1 backbone reference node)
#
# Compatibility contract:
#   This template is Cat 2 — its `main` backbone slot is filled by
#   RegulatoryQAGraphNode (src/graph/graph.py), which delegates the domain
#   workflow to DomainWorkflowGraph.  MainNode is NOT wired into that graph.
#
#   It is retained deliberately, with a concrete tested contract, as the Cat 1
#   single-slot `main`-node reference implementation:
#     - it backs the Cat 1 example src/examples/graph_cat1_sample.py
#       (the canonical single-node `main` slot pattern), and
#     - tests/unit/test_main_node.py exercises it to pin the mandatory node
#       contract — `execute(self, state) -> dict` (never `_invoke_impl`) —
#       returning SUCCESS with a `result`.
#   Because it is a real, exercised reference (not an inert stub), it is kept
#   rather than removed.

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

_REFERENCE_MSG = (
    "MainNode is the Cat 1 single-slot `main`-node reference implementation "
    "(backs src/examples/graph_cat1_sample.py); the Cat 2 FIN-C2-001 graph uses "
    "RegulatoryQAGraphNode -> DomainWorkflowGraph for its main slot."
)


class MainNode(FunctionNode):
    """Cat 1 single-slot `main`-node reference (see module docstring for the contract)."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # Every execute() emits at least one domain audit event.
        emit_trace_event(
            "main_node_reference_invoked",
            {
                "note": "MainNode is the Cat 1 main-slot reference node",
                "cat2_main_slot": "RegulatoryQAGraphNode + DomainWorkflowGraph",
            },
            state,
        )
        return {
            "status": AgentStatus.SUCCESS.value,
            "result": _REFERENCE_MSG,
        }
