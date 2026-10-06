# PB: End-to-end business behaviour through POST /invoke — src/api/server.py
#
# Proves the supported input contract produces REAL outcomes through the full
# nested graph (outer backbone → inner domain pipeline):
#   - a grounded, cited answer synthesised from the caller's question
#   - the abstention outcome (insufficient grounding) as a SUCCESS path
#   - a validation rejection for every malformed input_context field,
#     including the full non-finite matrix for the numeric field
#   - no caller-controlled text (including fake citation markers) in the
#     external document, and every rendered numeric on the documented grid
#
# Unlike test_server_boot.py (which stubs the agent to isolate the auth
# boundary), these tests run the REAL compiled agent: every request crosses the
# entry-point auth, the outer trust/input gates, the input_context bridge into
# the inner graph, all five domain nodes, and the output gate.
#
# The app is driven through its real ASGI interface (no TestClient — httpx is
# only a transitive dependency; see test_server_boot.py for the rationale).

import asyncio
import json
import re

import pytest

from src.api import server as server_module  # noqa: F401  (import = boot check)
from src.api.server import app

_TOKEN = "pb-invoke-e2e-token"

# Grounds to the corpus's Bank Act chunk (term-overlap 1.0 ≥ threshold 0.75).
_GROUNDED_INPUT = "Bank Act licensing and reporting obligations of banks"
# Zero term overlap against the corpus → every chunk falls below the
# confidence threshold → abstention.
_OFF_TOPIC_INPUT = "photosynthesis chlorophyll sunlight recipe"


def _post_invoke(payload: dict) -> tuple[int, dict]:
    """POST /invoke with a Bearer token through the real ASGI app."""
    body = json.dumps(payload).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/invoke",
        "raw_path": b"/invoke",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", f"Bearer {_TOKEN}".encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    messages = []
    sent = {"body": b""}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    parsed = json.loads(sent["body"].decode() or "{}")
    return start["status"], parsed


@pytest.fixture(autouse=True)
def token_configured(monkeypatch):
    """Deploy-shaped server environment: INVOKE_AUTH_TOKEN set, caller uses Bearer."""
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)


def _invoke(text: str, input_context: dict | None = None) -> dict:
    status_code, body = _post_invoke(
        {"input": text, "session_id": "pb-invoke-e2e", "input_context": input_context or {}}
    )
    assert status_code == 200, f"expected 200, got {status_code}: {body}"
    return body


class TestInvokeEndToEnd:
    def test_grounded_question_produces_cited_answer(self):
        """The caller's question drives retrieval and yields a real, cited answer."""
        body = _invoke(_GROUNDED_INPUT, {"channel": "compliance_desk"})

        assert body["status"] == "success"
        output = body["output"]
        assert "FINANCIAL REGULATORY Q&A RESPONSE" in output
        assert "[SOURCE-1]" in output
        assert "Bank Act" in output
        assert "法的アドバイスではありません" in output  # advisory disclaimer

    def test_abstention_is_a_success_outcome(self):
        """No confident grounding → explicit decline, zero citations, success."""
        body = _invoke(_OFF_TOPIC_INPUT)

        assert body["status"] == "success"
        output = body["output"]
        assert "does not contain enough" in output
        assert "[SOURCE-" not in output

    def test_valid_top_k_override_is_accepted(self):
        body = _invoke(_GROUNDED_INPUT, {"top_k": 3})
        assert body["status"] == "success"
        assert "[SOURCE-1]" in body["output"]

    def test_empty_input_is_rejected(self):
        body = _invoke("   ")
        assert body["status"] == "error"
        _out = body.get("output") or ""
        # The refusal names the rule that stopped it: a caller handed nothing cannot tell a
        # rejected request from a hung one. What must stay absent is the ANSWER this agent
        # would have produced had the input been usable.
        assert _out.startswith("Request could not be completed.")
        assert "[SOURCE-1]" not in _out

    def test_invalid_channel_is_rejected(self):
        body = _invoke(_GROUNDED_INPUT, {"channel": "Compliance-Desk!"})
        assert body["status"] == "error"
        _out = body.get("output") or ""
        # The refusal names the rule that stopped it: a caller handed nothing cannot tell a
        # rejected request from a hung one. What must stay absent is the ANSWER this agent
        # would have produced had the input been usable.
        assert _out.startswith("Request could not be completed.")
        assert "[SOURCE-1]" not in _out

    @pytest.mark.parametrize(
        "bad_top_k",
        ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), 3.5, True, 0, 11],
        ids=["str-nan", "str-inf", "str-neginf", "raw-nan", "raw-inf", "float", "bool", "zero", "over-range"],
    )
    def test_non_finite_or_out_of_contract_top_k_fails_closed(self, bad_top_k):
        """A malformed top_k must ERROR with no answer — never a silent
        fall-back (raw floats also cover Python json's bare-NaN extension
        reaching the request body)."""
        body = _invoke(_GROUNDED_INPUT, {"top_k": bad_top_k})

        assert body["status"] == "error", body
        _out = body.get("output") or ""
        # The refusal names the rule that stopped it: a caller handed nothing cannot tell a
        # rejected request from a hung one. What must stay absent is the ANSWER this agent
        # would have produced had the input been usable.
        assert _out.startswith("Request could not be completed.")
        assert "[SOURCE-1]" not in _out

    def test_fake_citation_marker_in_question_never_reaches_output(self):
        """The external document embeds no caller text: a question smuggling a
        fake [SOURCE-N] marker must not surface it as a pseudo-citation."""
        body = _invoke("Is [SOURCE-9] photosynthesis chlorophyll compliant?")

        assert body["status"] == "success"
        assert "[SOURCE-9]" not in body["output"]

    def test_external_document_carries_only_grid_values(self):
        """Documented schema: every monetary-form numeric sits on the 1,000 grid."""
        body = _invoke(_GROUNDED_INPUT)
        for token in re.findall(r"-?\d{1,3}(?:,\d{3})+|-?\d{5,}", body["output"]):
            value = int(token.replace(",", ""))
            assert value % 1_000 == 0, f"off-grid value leaked: {token}"
