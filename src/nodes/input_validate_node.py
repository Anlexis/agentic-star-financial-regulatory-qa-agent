"""AgentCore Platform v1.0"""

# FIN-C2-001 — InputValidateNode
# Inner domain node 1: domain-level query validation + normalisation.
#
# Distinct from PreProcessNode (trust + structural query check):
# this node applies domain rules — expands financial regulatory acronyms
# (FSA, FISC, AML, KYC) so retrieval accuracy improves, and rejects a
# query that is empty after normalisation.
#
# Inner node — ANONYMOUS trust (the outer PreProcessNode with
# VERIFIED_EXTERNAL already enforced trust; inner nodes must be ANONYMOUS so
# the outer invocation context passes through the GraphNode boundary without
# rejection).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Financial-regulatory acronym glossary — expanded for retrieval accuracy.
# Read-only; never mutated in execute().
_ACRONYM_GLOSSARY: Dict[str, str] = {
    "fsa": "FSA (Financial Services Agency 金融庁)",
    "fisc": "FISC (Center for Financial Industry Information Systems)",
    "aml": "AML (Anti-Money Laundering 犯罪収益移転防止法)",
    "kyc": "KYC (Know Your Customer 本人確認)",
    "fiea": "FIEA (金融商品取引法 Financial Instruments and Exchange Act)",
}

_WS_RE = re.compile(r"\s+")


def _expand_acronyms(query: str) -> str:
    """Append canonical expansions for any known acronym present in the query."""
    lowered = query.lower()
    expansions = [
        expansion
        for acronym, expansion in _ACRONYM_GLOSSARY.items()
        if re.search(rf"\b{re.escape(acronym)}\b", lowered)
    ]
    if not expansions:
        return query
    return f"{query} [{'; '.join(expansions)}]"


class InputValidateNode(FunctionNode):
    """Domain validation + normalisation of the compliance query for FIN-C2-001.

    Applies domain rules beyond the structural check in PreProcessNode:
    whitespace normalisation and financial-acronym expansion for retrieval.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        validated_input: str  — normalised query from PreProcessNode
                                Falls back to user_input for unit-test convenience.

    Output state keys (partial dict):
        normalized_query: str
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")

        if not isinstance(raw, str) or not raw.strip():
            logger.error("InputValidateNode: query is empty")
            emit_trace_event(
                "input_validate_failed",
                {"reason": "empty_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: query is empty"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. " + ("InputValidateNode: query is empty"),
            }

        collapsed = _WS_RE.sub(" ", raw).strip()
        normalized_query = _expand_acronyms(collapsed)

        logger.info(
            "InputValidateNode: normalised query chars=%d (from %d)",
            len(normalized_query),
            len(raw),
        )
        emit_trace_event(
            "input_validate_complete",
            {"query_length": len(normalized_query)},
            state,
        )

        return {
            "normalized_query": normalized_query,
            "status": AgentStatus.SUCCESS.value,
        }
