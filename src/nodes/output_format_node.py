"""AgentCore Platform v1.0"""

# FIN-C2-001 — OutputFormatNode (inner domain node 5, last in DomainWorkflowGraph)
# Assembles the final regulatory Q&A output from the grounded answer and its
# citations.  This is the last inner node — it produces the qa_output string
# that the outer PostProcessNode will gate.
#
# Output schema note: the external document renders monetary figures in units
# of 1,000 (the outer output gate independently enforces that grid — see
# post_process_node.py).  The bundled sample corpus contains no monetary
# figures, so the note line is appended only when the assembled document
# actually carries a monetary-form token; the detection reuses the gate's own
# grammar so renderer and gate can never drift apart.
#
# Inner node — ANONYMOUS trust (the outer pre_process slot already enforced
# VERIFIED_EXTERNAL).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.nodes.post_process_node import _NUM_TOKEN_RE
from src.schemas.state import from_json

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72

# Printed when the document renders monetary figures (approved external
# schema: amounts are expressed in units of 1,000).
_SCHEMA_NOTE = "※ Monetary figures in this response are expressed in units of 1,000."


def _assemble_output(answer: str, citations: List[Dict[str, Any]]) -> str:
    """Assemble the final Q&A document from the answer and its citations."""
    lines = [
        _SEPARATOR,
        "FINANCIAL REGULATORY Q&A RESPONSE",
        _SEPARATOR,
        "",
        answer,
        "",
    ]
    if citations:
        lines.append(_SUBSEP)
        lines.append("Citations")
        lines.append(_SUBSEP)
        for c in citations:
            lines.append(f"  [{c.get('marker', '?')}] {c.get('source', 'unknown')} " f"(p.{c.get('page', 'N/A')})")
    document = "\n".join(lines)
    if _NUM_TOKEN_RE.search(document):
        lines.append("")
        lines.append(_SCHEMA_NOTE)
        document = "\n".join(lines)
    lines.append(_SEPARATOR)
    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the final regulatory Q&A output (inner domain node).

    Reads answer and citations from State, renders the full Q&A document,
    and writes it to qa_output (and result) for the outer PostProcessNode.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        answer:    str  — grounded answer text from GenerateAnswerNode
        citations: str  — JSON list of cited sources

    Output state keys (partial dict):
        qa_output: str
        result:    str  (same as qa_output — backbone convention)
        status:    str
        error_log: list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        answer: str = state.get("answer") or ""
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), [])

        if not answer.strip():
            logger.error("OutputFormatNode: answer missing in state")
            emit_trace_event(
                "output_format_failed",
                {"reason": "missing_answer"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: answer missing in state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. " + ("OutputFormatNode: answer missing in state"),
            }

        qa_output = _assemble_output(answer, citations)

        logger.info(
            "OutputFormatNode: qa_output_chars=%d citations=%d",
            len(qa_output),
            len(citations),
        )
        emit_trace_event(
            "output_format_complete",
            {"output_length": len(qa_output), "citation_count": len(citations)},
            state,
        )

        return {
            "qa_output": qa_output,
            "result": qa_output,
            "status": AgentStatus.SUCCESS.value,
        }
