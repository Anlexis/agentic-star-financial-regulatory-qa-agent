"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# FIN-C2-001 — Financial Regulatory Q&A Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# All dict/list-valued fields are stored as JSON-serialized Optional[str].
# Use to_json() / from_json() helpers below at every producer and consumer
# node — one contract end-to-end.  Never type a dict/list field as a bare
# dict/list; that causes msgpack serialization failures.

import json
import math
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse a caller-controlled numeric: FINITE float within [lo, hi], else None.

    Rejects bools, non-numerics, and — critically — non-finite values: float()
    happily parses "NaN"/"Infinity" (and Python's json accepts bare NaN in
    request bodies), and IEEE NaN comparisons are always False, which turns a
    threshold check into silent FAIL-OPEN. Every caller-supplied number must
    come through here (or an equivalent explicit finite check).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


class State(AgentState):
    """Flat TypedDict for FIN-C2-001.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.

    dict/list fields use JSON-serialized Optional[str].
    formatted_output is NOT re-declared here — it is inherited from AgentState
    (re-declaring it with a bare type breaks the state contract).
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode (pre_process backbone slot)
    # ------------------------------------------------------------------

    # Validated + normalised compliance query string.
    # Produced by PreProcessNode; consumed by inner InputValidateNode.
    validated_input: Optional[str]

    # JSON-serialised channel/request metadata dict (stored as str).
    # Shape: {"source": str, "channel": str}
    enriched_context: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # Domain-normalised query (acronyms expanded, whitespace collapsed).
    # Produced by inner InputValidateNode; consumed by RetrieveNode.
    normalized_query: Optional[str]

    # JSON-serialised list of retrieved regulatory chunks (stored as str).
    # Each item: {"text": str, "source": str, "page": int, "score": float}
    retrieved_chunks: Optional[str]

    # JSON-serialised list of reranked + score-filtered chunks (stored as str).
    # Same item shape as retrieved_chunks; ordered by score desc, deduplicated,
    # dropped below config retrieval.score_threshold.
    reranked_chunks: Optional[str]

    # Grounded answer text with [SOURCE-N] citation markers + advisory disclaimer.
    # Produced by GenerateAnswerNode.
    answer: Optional[str]

    # JSON-serialised list of cited sources (stored as str).
    # Each item: {"marker": str, "source": str, "page": int}
    citations: Optional[str]

    # Final assembled Q&A output (answer + citations block + disclaimer).
    # Produced by inner OutputFormatNode; gated by outer PostProcessNode.
    qa_output: Optional[str]

    # ------------------------------------------------------------------
    # Outer layer — set by PostProcessNode (post_process backbone slot)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller (same content as qa_output after
    # the output gate passes).  formatted_output (from AgentState) is also set.
    result: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState
