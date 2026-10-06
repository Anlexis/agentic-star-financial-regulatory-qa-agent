"""AgentCore Platform v1.0"""

# FIN-C2-001 — GenerateAnswerNode
# Inner domain node 4 (grounded synthesis): build the cited answer.
#
# Deterministic grounded synthesis from the reranked chunks: the answer text
# is assembled from the retrieved passages with [SOURCE-N] citation markers so
# the grounding contract is verifiable offline.  The framework ships no LLM
# client, so this build reads no LLM configuration.  A live-LLM build wires in
# here by rendering prompts/fin_qa.j2 (the declared llm.system_prompt_template)
# under the same grounding rules.  See docs/02_design.md "Implementation Note —
# LLM synthesis".
#
# The caller's question is deliberately NOT embedded in the answer: the
# response is built from retrieved regulatory content only.  Echoing caller
# text would let a request smuggle arbitrary content — including fake
# [SOURCE-N] markers that masquerade as citations — into a document whose
# whole value is that every statement traces to a retrieved source.
#
# Anti-hallucination: the node cites ONLY chunks present in reranked_chunks.
# When no chunk survives the score gate it returns an explicit "insufficient
# grounding" answer rather than fabricating regulatory text.
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

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# Mandatory advisory disclaimer for a regulated domain.
_DISCLAIMER = (
    "※ 本回答は一般的な参考情報であり、法的アドバイスではありません。"
    "具体的な法令解釈については、専門家にご相談ください。"
)

_INSUFFICIENT_GROUNDING = (
    "The retrieved regulatory corpus does not contain enough high-confidence "
    "material to answer this question. Please consult the primary regulatory "
    "text or a qualified compliance professional."
)


def _build_answer(chunks: List[Dict[str, Any]]) -> str:
    """Assemble a grounded answer with [SOURCE-N] markers.

    Renders retrieved regulatory passages only — no caller-supplied text is
    embedded (see the module docstring), so every [SOURCE-N] marker in the
    answer was emitted here and maps to a citation entry.
    """
    lines: List[str] = [
        "Based on the retrieved regulatory sources:",
        "",
    ]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[SOURCE-{i}] {chunk.get('text', '')}")
    return "\n".join(lines).strip()


class GenerateAnswerNode(FunctionNode):
    """Generate a grounded, cited answer from the reranked regulatory chunks.

    Deterministic grounded synthesis (no LLM configuration is read); a
    live-LLM build renders prompts/fin_qa.j2 (declared
    llm.system_prompt_template) and invokes the platform LLM here.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        reranked_chunks:  str  — JSON list of grounded chunks

    Output state keys (partial dict):
        answer:    str  — grounded answer text with [SOURCE-N] markers
        citations: str  — JSON list of cited sources
        status:    str
        error_log: list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        chunks: List[Dict[str, Any]] = from_json(state.get("reranked_chunks"), [])
        query = state.get("normalized_query") or state.get("validated_input") or ""

        # ── Anti-hallucination: no grounded context → explicit fallback ───────
        if not chunks:
            logger.info("GenerateAnswerNode: no grounded chunks — insufficient grounding")
            answer = f"{_INSUFFICIENT_GROUNDING}\n\n{_DISCLAIMER}"
            emit_trace_event(
                "generate_answer_insufficient_grounding",
                {"query_length": len(query)},
                state,
            )
            return {
                "answer": answer,
                "citations": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        # ── Grounded synthesis ────────────────────────────────────────────────
        answer_body = _build_answer(chunks)
        answer = f"{answer_body}\n\n{_DISCLAIMER}"

        citations = [
            {
                "marker": f"SOURCE-{i}",
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page", 0),
            }
            for i, chunk in enumerate(chunks, 1)
        ]

        logger.info(
            "GenerateAnswerNode: answer_chars=%d citations=%d",
            len(answer),
            len(citations),
        )
        emit_trace_event(
            "generate_answer_complete",
            {"answer_length": len(answer), "citation_count": len(citations)},
            state,
        )

        return {
            "answer": answer,
            "citations": to_json(citations),
            "status": AgentStatus.SUCCESS.value,
        }
