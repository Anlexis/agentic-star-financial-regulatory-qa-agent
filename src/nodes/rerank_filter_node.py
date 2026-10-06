"""AgentCore Platform v1.0"""

# FIN-C2-001 — RerankFilterNode
# Inner domain node 3: rerank retrieved chunks and drop low-confidence hits.
#
# Applies the config retrieval.score_threshold gate (default 0.75 — a
# regulated domain needs a high confidence bar), reranks surviving chunks by
# score descending, and deduplicates identical passages.  Guarantees
# GenerateAnswerNode only sees grounded, high-score context
# (anti-hallucination).  The threshold is operator configuration only — there
# is deliberately no caller override, so a caller can never lower the
# grounding bar.
#
# Inner node — ANONYMOUS trust (the outer pre_process slot already enforced
# VERIFIED_EXTERNAL).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import math
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

_DEFAULT_SCORE_THRESHOLD = 0.75


class RerankFilterNode(FunctionNode):
    """Rerank + score-filter the retrieved regulatory chunks.

    Inner node — ANONYMOUS trust (see module docstring).

    Constructor config (see __init__):
        score_threshold: float — confidence gate; defaults to
                                 _DEFAULT_SCORE_THRESHOLD.

    Input state keys:
        retrieved_chunks: str  — JSON list of scored chunks from RetrieveNode

    Output state keys (partial dict):
        reranked_chunks: str   — JSON list of surviving chunks
        status:          str
        error_log:       list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Bind the declared retrieval settings at construction time.

        The node contract is `execute(self, state) -> dict` — no extra
        parameters.  Static configuration therefore arrives through the
        constructor: DomainWorkflowGraph.register_nodes() passes the declared
        settings (config/config.yaml `retrieval` block), and absent keys fall
        back to the module defaults.
        """
        super().__init__()
        self._config: Dict[str, Any] = dict(config or {})

    def _resolve_threshold(self) -> float:
        """Resolve the confidence threshold from config, fail-safe.

        A malformed configured value (non-numeric, bool, NaN/Infinity, out of
        [0, 1]) falls back to the module default rather than weakening the
        gate: a NaN threshold would compare False against every score and
        wave every low-confidence chunk through to the answer.
        """
        raw = self._config.get("score_threshold", _DEFAULT_SCORE_THRESHOLD)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return _DEFAULT_SCORE_THRESHOLD
        threshold = float(raw)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            return _DEFAULT_SCORE_THRESHOLD
        return threshold

    def execute(self, state: AgentState) -> dict[str, Any]:
        retrieved: List[Dict[str, Any]] = from_json(state.get("retrieved_chunks"), [])

        if not retrieved:
            logger.error("RerankFilterNode: retrieved_chunks missing in state")
            emit_trace_event(
                "rerank_filter_failed",
                {"reason": "missing_retrieved_chunks"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RerankFilterNode: retrieved_chunks missing in state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("RerankFilterNode: retrieved_chunks missing in state"),
            }

        threshold = self._resolve_threshold()

        # Filter by score threshold, then rerank by score desc, then dedup.
        seen: set[tuple[Any, Any]] = set()
        reranked: List[Dict[str, Any]] = []
        for chunk in sorted(retrieved, key=lambda c: c.get("score", 0.0), reverse=True):
            score = chunk.get("score", 0.0)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                continue  # malformed score never clears the confidence gate
            if not math.isfinite(float(score)) or float(score) < threshold:
                continue
            key = (chunk.get("source"), chunk.get("page"))
            if key in seen:
                continue
            seen.add(key)
            reranked.append(chunk)

        logger.info(
            "RerankFilterNode: %d/%d chunks passed threshold=%.2f",
            len(reranked),
            len(retrieved),
            threshold,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "input_count": len(retrieved),
                "surviving_count": len(reranked),
                "score_threshold": threshold,
            },
            state,
        )

        return {
            "reranked_chunks": to_json(reranked),
            "status": AgentStatus.SUCCESS.value,
        }
