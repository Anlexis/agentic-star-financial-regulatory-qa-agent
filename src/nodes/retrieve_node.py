"""AgentCore Platform v1.0"""

# FIN-C2-001 — RetrieveNode
# Inner domain node 2: retrieve top-k regulatory chunks for the query.
#
# Deterministic keyword-overlap retrieval over a small bundled sample
# regulatory corpus.  Production wires a real vector store (config
# retrieval.vector_store) here; the node contract (query in → scored chunks
# out) is unchanged.  The framework ships no vector-store client, so the
# bundled build scores by term overlap to stay fully offline-testable.
#
# Retrieval depth: an optional caller override (input_context.top_k, bounded
# 1..10) takes precedence over the configured retrieval.top_k, which in turn
# falls back to the module default.
#
# Inner node — ANONYMOUS trust (the outer pre_process slot already enforced
# VERIFIED_EXTERNAL; inner nodes must be ANONYMOUS so the outer invocation
# context passes through the GraphNode boundary without rejection).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5

# Caller-override bounds — must match the contract validated in
# PreProcessNode._validate_input_context (the ingest boundary). Re-checked
# here so a direct inner-graph invocation gets the same fail-closed rule.
_TOP_K_MIN = 1
_TOP_K_MAX = 10

# Sample regulatory corpus (public FSA/FISC/AML material). Read-only; a real
# deployment swaps this for a vector store via config retrieval.vector_store.
# Never mutated in execute().
_SAMPLE_CORPUS: List[Dict[str, Any]] = [
    {
        "text": "FSA guidance requires financial institutions to maintain a "
        "risk-based AML/CFT program including ongoing customer due "
        "diligence and suspicious transaction reporting.",
        "source": "FSA AML/CFT Guidelines",
        "page": 12,
    },
    {
        "text": "Under the 犯罪収益移転防止法 (Act on Prevention of Transfer of "
        "Criminal Proceeds), specified business operators must verify "
        "customer identity (KYC) at the point of establishing a "
        "transaction.",
        "source": "AML Act (犯罪収益移転防止法)",
        "page": 3,
    },
    {
        "text": "FISC security guidelines v12 set control baselines for financial "
        "information systems covering access control, encryption, and "
        "incident response readiness.",
        "source": "FISC Security Guidelines v12",
        "page": 45,
    },
    {
        "text": "The 金融商品取引法 (FIEA) imposes disclosure and conduct-of-business "
        "obligations on registered financial instruments business "
        "operators, including suitability principle compliance.",
        "source": "FIEA (金融商品取引法)",
        "page": 88,
    },
    {
        "text": "The 銀行法 (Bank Act) governs licensing, sound-management, and "
        "reporting obligations of banks, including large-exposure limits "
        "and arm's-length rules for related-party transactions.",
        "source": "Bank Act (銀行法)",
        "page": 21,
    },
    {
        "text": "FSA administrative guidelines describe the supervisory review of "
        "internal controls and the expectation that compliance officers "
        "escalate material regulatory breaches to senior management.",
        "source": "FSA Supervisory Guidelines",
        "page": 57,
    },
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9぀-ヿ一-鿿]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _score(query_tokens: List[str], chunk_text: str) -> float:
    """Deterministic term-overlap score in [0, 1] (Jaccard-like)."""
    if not query_tokens:
        return 0.0
    q = set(query_tokens)
    c = set(_tokenize(chunk_text))
    if not c:
        return 0.0
    overlap = len(q & c)
    return round(overlap / len(q), 4)


class RetrieveNode(FunctionNode):
    """Retrieve top-k regulatory chunks for the normalised query.

    Deterministic term-overlap retrieval over a bundled sample corpus;
    production swaps in a real vector store via config retrieval.vector_store.

    Inner node — ANONYMOUS trust (see module docstring).

    Constructor config (see __init__):
        top_k: int — configured retrieval depth; defaults to _DEFAULT_TOP_K.

    Input state keys:
        normalized_query: str   — from InputValidateNode
                                  Falls back to validated_input / user_input.
        input_context:    dict  — optional caller parameters; top_k (int 1..10)
                                  overrides the configured depth for this
                                  invocation. Invalid values fail CLOSED.

    Output state keys (partial dict):
        retrieved_chunks: str  — JSON-serialised list of scored chunks
        status:           str
        error_log:        list[str]  (only on ERROR)
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

    def _resolve_top_k(self, state: AgentState) -> tuple[int, Optional[str]]:
        """Resolve the retrieval depth: caller override > config > default.

        The caller override (input_context.top_k) is re-validated here with
        the same strict rule as the ingest boundary: an integer (bools
        rejected) within [1, 10]. An invalid override fails CLOSED with a
        field-naming error — never silently falls back.
        """
        input_context = state.get("input_context")
        if isinstance(input_context, dict) and input_context.get("top_k") is not None:
            override = input_context["top_k"]
            if not isinstance(override, int) or isinstance(override, bool) or not _TOP_K_MIN <= override <= _TOP_K_MAX:
                return 0, f"input_context.top_k must be an integer between {_TOP_K_MIN} and {_TOP_K_MAX}"
            return override, None

        configured = self._config.get("top_k", _DEFAULT_TOP_K)
        if not isinstance(configured, int) or isinstance(configured, bool) or configured < 1:
            # Malformed configuration never crashes retrieval — fall back.
            return _DEFAULT_TOP_K, None
        return configured, None

    def execute(self, state: AgentState) -> dict[str, Any]:
        query = state.get("normalized_query") or state.get("validated_input") or state.get("user_input", "")

        if not isinstance(query, str) or not query.strip():
            logger.error("RetrieveNode: no query available for retrieval")
            emit_trace_event(
                "retrieve_failed",
                {"reason": "empty_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: no query available for retrieval"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("RetrieveNode: no query available for retrieval"),
            }

        top_k, top_k_error = self._resolve_top_k(state)
        if top_k_error:
            logger.error("RetrieveNode: invalid caller retrieval override")
            emit_trace_event(
                "retrieve_failed",
                {"reason": "invalid_top_k_override"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"RetrieveNode: {top_k_error}"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. " + (f"RetrieveNode: {top_k_error}"),
            }

        query_tokens = _tokenize(query)
        # Build a LOCAL scored list — never mutate the module-level corpus.
        scored: List[Dict[str, Any]] = []
        for chunk in _SAMPLE_CORPUS:
            scored.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "score": _score(query_tokens, chunk["text"]),
                }
            )

        scored.sort(key=lambda c: c["score"], reverse=True)
        retrieved = scored[:top_k]

        logger.info(
            "RetrieveNode: retrieved %d/%d chunks (top_k=%d) top_score=%.4f",
            len(retrieved),
            len(_SAMPLE_CORPUS),
            top_k,
            retrieved[0]["score"] if retrieved else 0.0,
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "retrieved_count": len(retrieved),
                "top_k": top_k,
                "top_score": retrieved[0]["score"] if retrieved else 0.0,
            },
            state,
        )

        return {
            "retrieved_chunks": to_json(retrieved),
            "status": AgentStatus.SUCCESS.value,
        }
