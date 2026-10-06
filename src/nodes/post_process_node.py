"""AgentCore Platform v1.0"""

# FIN-C2-001 — PostProcessNode
# Outer backbone post_process slot: the external-output boundary for the
# regulatory Q&A document.  Three independent layers, in order:
#
#   (1) credential scan — API keys, JWTs, Bearer tokens, password assignments
#       anywhere in the assembled document withhold the answer entirely
#       (sanitised stub, status=ERROR);
#   (2) verbatim caller-text redaction — the response is built from retrieved
#       regulatory content only, so a verbatim embedding of the caller's raw
#       question (or derived query text) is a leak, not a feature: any such
#       embedding is replaced with [REDACTED];
#   (3) monetary precision grid — the documented external schema expresses
#       monetary figures in units of 1,000; every monetary-form token is
#       snapped onto that grid (off-grid values are full-precision figures
#       leaking to the external surface), with an audit event per redaction.
#
# The domain output gate is the module-level function `_security_gate_output`
# called from inside execute() — NOT an instance method on the node class
# (the framework auto-wraps node instance methods on the real invoke path,
# which would raise AttributeError).  The agent class
# (FinancialRegulatoryQAAgent) exposes the same credential scanner as the
# canonical output-gate entry point and delegates to this module (single
# source of truth).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)

# Credential patterns that MUST NOT appear in the formatted Q&A output.
_CREDENTIAL_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:sk|pk|ak)-[A-Za-z0-9]{16,}", "api_key_pattern"),
    (r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "jwt_pattern"),
    (r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", "bearer_token"),
    (
        r"(?:password|passwd|secret|api_key|token|access_key|private_key)" r"\s*[:=]\s*\S{8,}",
        "credential_assignment",
    ),
]

# State fields that must NEVER be embedded verbatim in the external response.
# The document is assembled from retrieved regulatory passages and their
# citations only — caller-derived text (the raw question, its normalised
# forms, request metadata) re-appearing verbatim means caller-controlled
# content reached the external surface.
_BLOCKED_FIELDS = frozenset(
    {
        "user_input",
        "validated_input",
        "normalized_query",
        "enriched_context",
    }
)

# Approved external precision: monetary figures are expressed in units of
# 1,000 (must match the schema note rendered by
# src/nodes/output_format_node.py — the document RENDERS on this grid, this
# gate ENFORCES it).
_EXTERNAL_ROUND_UNIT = 1000
# EXPLICIT output schema — monetary values are identified by FORM and by
# CURRENCY CONTEXT, never by magnitude:
#   form:    comma-grouped numbers (9,999 / 1,234,567) and unformatted runs
#            of 5+ digits (a rendering-regression leak);
#   context: any bare 1-4 digit number associated with a currency marker is
#            monetary even though short — SYMMETRICALLY: a 3-letter uppercase
#            code or a currency symbol (incl. fullwidth ￥ and 円/₩), before or
#            after the value, separated by horizontal whitespace / a single
#            newline or attached, signed or unsigned. A SEPARATED 3-letter
#            uppercase word counts as a code on purpose ("SKF 6205" snaps):
#            there a false snap fails SAFE while a missed leak does not.
# Structural tokens stay untouched: page references ("p.21"), version tags
# ("v12"), bare counts, years without currency adjacency ("in 2026"), this
# repo's own citation markers ("[SOURCE-3]"), and store-supplied document
# identifiers ("FSA-2026-0114", "ISO-20022", the ISIN "JP1234567890").
# ALL matched tokens, at ANY magnitude, must sit on the rounding grid;
# off-grid = a full-precision figure reaching the external surface,
# snapped + audited.

# The identifier alphabet of THIS document, read off the renderer rather than
# assumed. src/nodes/output_format_node.py interpolates three things verbatim:
# the citation markers this repo emits itself ("[SOURCE-3]" — hyphen), the
# store-supplied `source` and `page` fields (`(p.N/A)` is the renderer's own
# fallback and `page` is never coerced, so `(p.12/45)` renders too — slash;
# the production build swaps the bundled corpus for a vector store whose
# document keys are conventionally snake_case — underscore), and the retrieved
# regulatory passages. A number may not be ENTERED from inside any such token
# in either direction: "FSA-2026-0114" and "JP1234567890" are names, and
# rewriting one names a different document.
# `.` sits in the LEADING guard ONLY. Leading, it stops an alternative entering
# a number part-way through and treating the tail of a fraction as a value of
# its own ("0.123456" can be entered neither at "123456" nor at "23456").
# Trailing, it would let an amount that ends a sentence escape the grid ("The
# penalty reaches JPY 9999.").
_IDENT_CHAR = r"A-Za-z0-9_\-/"
_LEAD_GUARD = rf"(?<![{_IDENT_CHAR}.])"
_TRAIL_GUARD = rf"(?![{_IDENT_CHAR}])"

# Grammar (group-based; no lookbehinds inside, so the delimiter can be an
# arbitrary horizontal run). Every value accepts an optional explicit +/- sign.
# Branch order matters: currency-context branches first, then form-based.
_CURRENCY_MARKER = r"(?:\b[A-Z]{3}|[¥￥$€£円₩])"
# Delimiter between a currency marker and its value: horizontal whitespace and at
# most ONE newline — never a paragraph break. A plain `\s*` spans blank lines, so a
# 3-letter uppercase word ending a line would bind to the number that opens the next
# block and rewrite it ("Currency: JPY\n\n3. Cash Position" -> "0. Cash Position").
# Every enumerated leak form (spaces, tabs, single newline, signed, symmetric,
# comma-grouped) still matches.
_GATE_DELIM = r"[ \t]*(?:\n[ \t]*)?"

# A monetary amount may carry a DECIMAL part, and every value alternative absorbs
# it into the SAME token. Two separate things go wrong without that:
#   - the fraction of "9999.99999" is a standalone 5+-digit run in its own
#     right, so the form-based branch latches onto it and rewrites a percentage
#     into "9999.100,000" — a number the document never contained;
#   - in currency context the integer part snaps while the fraction dangles
#     ("JPY 1234.56" -> "JPY 1,000.56"), producing a value that is neither the
#     true figure nor on the grid.
# The fix is NOT to exempt decimals from the grid. An off-grid amount in
# explicit currency context still snaps — as ONE number ("JPY 1234.56" ->
# "JPY 1,000") — while a ratio, percentage, version or page reference, which was
# never a monetary token, is left byte-identical.
#
# The `(?!\.\d)` arm is what makes absorption stick. A plain `(?:\.\d+)?` lets
# the engine backtrack out of the fraction and re-match the integer part alone
# whenever the text right after the fraction fails the trailing guard —
# "JPY 1234.56m" would go back to matching "JPY 1234" and the dangling-fraction
# bug returns. Either the fraction is taken whole, or there is none there.
_VAL_FRACTION = r"(?:\.\d+|(?!\.\d))"

_NUM_TOKEN_RE = re.compile(
    _LEAD_GUARD
    # marker THEN value: "JPY 9999", "JPY  -9999", "JPY\t9999", "¥9999", "USD\n+9999".
    # The value alternatives accept a comma-grouped form FIRST: the regex is
    # leftmost-first, so without it "JPY 1,234" would match as marker + "1"
    # (mangling the number on the snap) instead of as the whole grouped value
    # — an on-grid "JPY 1,000" must stay byte-identical, and an off-grid
    # "JPY 1,234" must snap as 1234, not as 1.
    + rf"(?:(?P<pre>{_CURRENCY_MARKER}{_GATE_DELIM})"
    rf"(?P<val_after>[+-]?\d{{1,3}}(?:,\d{{3}})+{_VAL_FRACTION}|[+-]?\d{{1,4}}{_VAL_FRACTION})"
    # value THEN marker: "9999 JPY", "-9999\tJPY", "9999円", "+9999  $"
    rf"|(?P<val_before>[+-]?\d{{1,4}}{_VAL_FRACTION})(?P<post>{_GATE_DELIM}(?:[A-Z]{{3}}\b|[¥￥$€£円₩]))"
    # form-based, standalone at any magnitude: comma-grouped or 5+-digit runs
    rf"|(?P<val_form>[+-]?\d{{1,3}}(?:,\d{{3}})+{_VAL_FRACTION}|[+-]?\d{{5,}}{_VAL_FRACTION}))" + _TRAIL_GUARD
)

# ISO 4217 alphabetic codes. Consulted for ONE decision only: whether an
# ATTACHED "<three uppercase letters><digits>" token is an amount or a name.
# The two are lexically identical — "JPY-9999" (a signed amount, a real leak
# form) and "FSA-2026" / "ISO-20022" / "SKF-6205" (document and part numbers)
# have the same shape — so no amount of guard-widening can separate them;
# something has to know which three-letter words are currencies. ISO 4217 is a
# CLOSED, standardised vocabulary; the set of identifiers never could be, which
# is why the knowledge sits on this side.
# The restriction applies to the ATTACHED form ONLY. A SEPARATED marker
# ("SKF 6205") stays unrestricted, because there the ambiguity is genuine and a
# false snap fails safe. Withdrawn codes are kept: regulatory text quotes
# historical amounts, and listing a code can only make the gate snap MORE.
_ISO_CURRENCY_CODES = frozenset(
    (
        "AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BOV "
        "BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUP CVE CZK "
        "DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL "
        "HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT "
        "LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MXV MYR "
        "MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF "
        "SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP "
        "TRY TTD TWD TZS UAH UGX USD USN UYI UYU UYW UZS VED VES VND VUV WST XAF XAG XAU "
        "XBA XBB XBC XBD XCD XCG XDR XOF XPD XPF XPT XSU XTS XUA XXX YER ZAR ZMW ZWG "
        # withdrawn but still quoted in historical regulatory text
        "BYR CUC HRK LTL LVL MRO SLL STD VEF ZMK ZWL"
    ).split()
)


def _is_attached_identifier(match: "re.Match[str]") -> bool:
    """True when this match is "<three uppercase letters><digits>" — a name.

    Only the marker-then-value branch with an ASCII-ALPHABETIC marker and an
    EMPTY delimiter can be ambiguous; a currency SYMBOL ("¥9999", "円9999") is
    never an identifier prefix, and a SEPARATED marker is deliberately left
    unrestricted so the gate keeps failing safe there.
    """
    pre = match.group("pre") or ""
    if not (pre.isascii() and pre.isalpha()):
        return False
    return pre.upper() not in _ISO_CURRENCY_CODES


def _enforce_precision(result: str) -> tuple[str, int]:
    """Snap every monetary-form token onto the approved external grid.

    Returns (sanitised_result, redaction_count). A redaction means a
    full-precision monetary figure reached the external surface — the gate
    rounds it onto the approved grid. The currency marker, the original
    delimiter whitespace, and the explicit sign of the original token are all
    preserved on the snapped replacement.
    """
    redactions = 0

    def _snap(match: re.Match[str]) -> str:
        nonlocal redactions
        pre = match.group("pre") or ""
        post = match.group("post") or ""
        if _is_attached_identifier(match):
            return match.group(0)
        token = match.group("val_after") or match.group("val_before") or match.group("val_form")
        # float(), not int(): the token may carry a decimal fraction, and the
        # whole amount — not just its integer part — is what sits on the grid.
        value = float(token.replace(",", ""))  # float() understands leading +/-
        if value % _EXTERNAL_ROUND_UNIT == 0:
            return match.group(0)
        redactions += 1
        snapped = round(value / _EXTERNAL_ROUND_UNIT) * _EXTERNAL_ROUND_UNIT
        plus = "+" if token.startswith("+") and snapped >= 0 else ""
        return f"{pre}{plus}{snapped:,d}{post}"

    return _NUM_TOKEN_RE.sub(_snap, result), redactions


def _security_gate_output(content: str) -> Optional[str]:
    """Scan output for disallowed credential/secret patterns.

    Returns the first violation name, or None if the output is clean.
    Module-level function (not a node instance method) — the framework
    auto-wraps node instance methods on the real invoke path, so the gate
    must live at module level.
    """
    for pattern, name in _CREDENTIAL_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return name
    return None


def _redact_blocked_fields(result: str, state: AgentState) -> tuple[str, List[str]]:
    """Replace verbatim embeddings of caller-derived state text with [REDACTED].

    Returns (sanitised_result, redacted_field_names). Only substantial values
    (len > 10) are matched so short incidental overlaps are not redacted.
    """
    redacted: List[str] = []
    sanitised = result
    for field in sorted(_BLOCKED_FIELDS):
        value = state.get(field)
        if isinstance(value, str) and len(value) > 10 and value in sanitised:
            sanitised = sanitised.replace(value, "[REDACTED]")
            redacted.append(field)
    return sanitised, redacted


class PostProcessNode(FunctionNode):
    """Apply the output gate and expose the final regulatory Q&A answer.

    Outer backbone post_process slot.  Declared ANONYMOUS — trust was
    already enforced at PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        qa_output: str  — assembled Q&A answer from inner OutputFormatNode
        answer:    str  — grounded answer text (fallback source)

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        qa_output: str = state.get("qa_output") or state.get("answer") or ""

        # ── Fallback for empty answer ─────────────────────────────────────────
        if not qa_output.strip():
            logger.warning("PostProcessNode: qa_output is empty — using fallback message")
            qa_output = "[Regulatory Q&A] No answer content generated. " "Check error_log for upstream failures."

        # ── Layer 1: credential scan (withhold entirely) ──────────────────────
        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=qa_output,
            domain="FIN FinancialRegulatoryQAAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(qa_output, str) and not _security_gate_output(qa_output + _review):
            qa_output = qa_output + _review

        violation = _security_gate_output(qa_output)
        if violation:
            logger.error("PostProcessNode: credential pattern detected in output — %s", violation)
            emit_trace_event(
                "post_process_credential_violation",
                {"violation": violation},
                state,
            )
            sanitised = (
                f"[ANSWER REDACTED: output contained a disallowed pattern "
                f"({violation}). Contact the compliance security team.]"
            )
            return {
                "formatted_output": sanitised,
                "result": sanitised,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: credential pattern detected — {violation}"],
            }

        # ── Layer 2: verbatim caller-text redaction ───────────────────────────
        sanitised_output, redacted_fields = _redact_blocked_fields(qa_output, state)
        if redacted_fields:
            logger.error(
                "PostProcessNode: caller-derived text embedded verbatim in output — %s",
                ", ".join(redacted_fields),
            )
            emit_trace_event(
                "post_process_blocked_field_redaction",
                {"fields": redacted_fields},
                state,
            )

        # ── Layer 3: monetary precision grid ──────────────────────────────────
        sanitised_output, precision_redactions = _enforce_precision(sanitised_output)
        if precision_redactions:
            logger.warning(
                "PostProcessNode: %d off-grid monetary token(s) snapped to the external grid",
                precision_redactions,
            )
            emit_trace_event(
                "post_process_precision_redaction",
                {"redaction_count": precision_redactions},
                state,
            )

        logger.info("PostProcessNode: output gate passed — length=%d", len(sanitised_output))
        emit_trace_event(
            "post_process_complete",
            {"output_length": len(sanitised_output)},
            state,
        )

        return {
            "formatted_output": sanitised_output,
            "result": sanitised_output,
            "status": AgentStatus.SUCCESS.value,
        }
