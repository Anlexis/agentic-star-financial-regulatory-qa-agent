"""AgentCore Platform v1.0"""

# FIN-C2-001 — PreProcessNode
# Outer backbone pre_process slot (trust gate + caller-data validation).
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL trust (required_trust_level — the trust gate)
#   - Reject empty / over-long queries early (fail-fast)
#   - Strip control characters and collapse whitespace
#   - Validate every declared input_context field against explicit bounds
#     (fail CLOSED on an invalid field; never echo the rejected value)
#   - Write validated_input (normalised query str) + enriched_context to State
#   - Emit an audit event for every validation decision
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json

logger = logging.getLogger(__name__)

# Maximum accepted query length (bound the input; 2000 chars covers any
# realistic compliance question).
_MAX_QUERY_CHARS = 2000

# Control characters (except tab/newline) are stripped before processing.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")

# ── Caller-data (input_context) validation bounds ─────────────────────────────
# input_context is caller-supplied and untrusted: every declared field is
# bounds-checked before it can influence the run. Violations return
# status=ERROR naming the offending FIELD only — never the offending VALUE
# (caller data must not round-trip into error logs). Undeclared keys are
# ignored; the /invoke adapter separately caps the serialized size.
#
# channel — free text is rejected: the value is stored in state metadata, so
# it is locked to an inert identifier alphabet.
_CHANNEL_RE = re.compile(r"^[a-z0-9_]{1,32}$")
# top_k — optional per-invocation retrieval-depth override (RetrieveNode).
_TOP_K_MIN = 1
_TOP_K_MAX = 10


def _validate_input_context(input_context: Any) -> tuple[dict[str, Any], Optional[str]]:
    """Validate the caller's input_context against the declared contract.

    Returns (normalised_context, error_message). Error messages name the
    field only — never the rejected value. Accepted fields:

        channel: str matching ^[a-z0-9_]{1,32}$   (absent → "unknown")
        top_k:   int 1..10                        (absent → configured default)

    Any other key is ignored. A non-mapping input_context is rejected.
    """
    if input_context is None:
        return {"channel": "unknown"}, None
    if not isinstance(input_context, dict):
        return {}, "input_context must be an object"

    normalised: dict[str, Any] = {}

    channel = input_context.get("channel")
    if channel is None:
        normalised["channel"] = "unknown"
    elif isinstance(channel, str) and _CHANNEL_RE.match(channel):
        normalised["channel"] = channel
    else:
        return {}, ("input_context.channel must be a lowercase identifier " "(a-z, 0-9, _; 1-32 chars)")

    top_k = input_context.get("top_k")
    if top_k is not None:
        # Strict integer check: bools are ints in Python, floats/strings/NaN
        # and non-finite JSON values all fail isinstance — one rule covers
        # every malformed shape, and the range bound covers the rest.
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not _TOP_K_MIN <= top_k <= _TOP_K_MAX:
            return {}, f"input_context.top_k must be an integer between {_TOP_K_MIN} and {_TOP_K_MAX}"
        normalised["top_k"] = top_k

    return normalised, None


class PreProcessNode(FunctionNode):
    """Input validation for FIN-C2-001.

    Validates the caller-supplied compliance query and the input_context
    contract before the domain workflow runs.  This is the outer backbone's
    pre_process slot — the only node with VERIFIED_EXTERNAL trust so that
    unauthenticated or anonymous callers are rejected here (fail-fast; inner
    domain nodes carry ANONYMOUS trust and never see untrusted input
    directly).

    Input state keys:
        user_input:    str   — caller-supplied compliance question
        input_context: dict  — optional caller parameters (channel, top_k)

    Output state keys (partial dict):
        validated_input:  str        — normalised query string
        enriched_context: str        — JSON-serialised channel metadata
        status:           str        — AgentStatus.SUCCESS.value or ERROR
        error_log:        list[str]  — set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context") or {}

        # ── Emptiness check ───────────────────────────────────────────────────
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            logger.warning("PreProcessNode: user_input is empty or missing")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "empty_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: user_input is empty or missing"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("PreProcessNode: user_input is empty or missing"),
            }

        # ── Length bound ──────────────────────────────────────────────────────
        if len(user_input) > _MAX_QUERY_CHARS:
            logger.warning(
                "PreProcessNode: query exceeds %d chars (%d)",
                _MAX_QUERY_CHARS,
                len(user_input),
            )
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "query_too_long", "length": len(user_input)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PreProcessNode: query exceeds {_MAX_QUERY_CHARS} characters"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (f"PreProcessNode: query exceeds {_MAX_QUERY_CHARS} characters"),
            }

        # ── Caller-data contract (input_context) ──────────────────────────────
        context, context_error = _validate_input_context(input_context)
        if context_error:
            # Name the field, never the value.
            logger.warning("PreProcessNode: input_context validation failed")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "invalid_input_context"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PreProcessNode: {context_error}"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. " + (f"PreProcessNode: {context_error}"),
            }

        # ── Normalise (strip control chars, collapse whitespace) ──────────────
        cleaned = _CONTROL_CHARS_RE.sub("", user_input)
        normalised_query = _WS_RE.sub(" ", cleaned).strip()

        logger.info("PreProcessNode: validated query chars=%d", len(normalised_query))
        emit_trace_event(
            "pre_process_validated",
            {"query_length": len(normalised_query)},
            state,
        )

        return {
            "validated_input": normalised_query,
            "enriched_context": to_json(
                {
                    "source": "FinancialRegulatoryQAAgent",
                    "channel": context.get("channel", "unknown"),
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }
