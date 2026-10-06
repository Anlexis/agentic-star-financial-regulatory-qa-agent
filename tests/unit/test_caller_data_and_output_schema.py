# FIN-C2-001 — Unit tests: caller-data input contract + external output schema.
#
# Covers the hardening behaviours:
#   1) PreProcessNode validates every declared input_context field against
#      explicit bounds (identifier alphabet, integer ranges), fails CLOSED on
#      an invalid field, and never echoes the rejected value.
#   2) RetrieveNode honours a valid per-invocation top_k override and fails
#      CLOSED on an invalid one; malformed operator config falls back safely.
#   3) RerankFilterNode's confidence threshold cannot be weakened by a
#      non-finite configured value (NaN comparisons are always False).
#   4) The declared runtime configuration (config/config.yaml) reaches the
#      inner graph's node constructors — and a caller top_k override reaches
#      inner retrieval through the full nested graph (the context bridge).
#   5) The output gate enforces the documented external schema on every
#      monetary representation (both ways: every leak form snaps; structural
#      tokens stay byte-identical), redacts verbatim caller-text embeddings,
#      and the renderer prints the schema note exactly when money renders.

import re

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.output_format_node import _SCHEMA_NOTE, OutputFormatNode
from src.nodes.post_process_node import (
    PostProcessNode,
    _enforce_precision,
    _redact_blocked_fields,
)
from src.nodes.pre_process_node import PreProcessNode, _validate_input_context
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import _finite_in_range, from_json, to_json

GROUNDED_QUERY = "bank act licensing and reporting obligations of banks"


# ── input_context contract ────────────────────────────────────────────────────


class TestInputContextValidation:
    def test_absent_context_defaults_channel_unknown(self):
        ctx, err = _validate_input_context(None)
        assert err is None
        assert ctx == {"channel": "unknown"}

    def test_valid_full_context(self):
        ctx, err = _validate_input_context({"channel": "compliance_desk", "top_k": 3})
        assert err is None
        assert ctx == {"channel": "compliance_desk", "top_k": 3}

    def test_unknown_keys_are_ignored(self):
        ctx, err = _validate_input_context({"channel": "x1", "someone_elses_key": "value"})
        assert err is None
        assert "someone_elses_key" not in ctx

    def test_non_mapping_context_is_rejected(self):
        _, err = _validate_input_context(["not", "a", "dict"])
        assert err is not None and "input_context" in err

    @pytest.mark.parametrize(
        "bad_channel",
        ["Compliance", "desk-1", "a" * 33, "", "spa ce", 5, ["x"], {"a": 1}],
        ids=["uppercase", "hyphen", "too-long", "empty", "space", "int", "list", "dict"],
    )
    def test_invalid_channel_fails_closed(self, bad_channel):
        _, err = _validate_input_context({"channel": bad_channel})
        assert err is not None and "channel" in err

    def test_rejected_channel_value_is_never_echoed(self):
        _, err = _validate_input_context({"channel": "EVILVALUE"})
        assert err is not None
        assert "EVILVALUE" not in err

    @pytest.mark.parametrize(
        "bad_top_k",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            float("nan"),
            float("inf"),
            float("-inf"),
            3.5,
            True,
            False,
            0,
            11,
            -1,
            "5",
            [],
            {},
        ],
        ids=[
            "str-nan",
            "str-inf",
            "str-neginf",
            "raw-nan",
            "raw-inf",
            "raw-neginf",
            "float",
            "true",
            "false",
            "zero",
            "over-range",
            "negative",
            "numeric-str",
            "list",
            "dict",
        ],
    )
    def test_invalid_top_k_fails_closed(self, bad_top_k):
        _, err = _validate_input_context({"top_k": bad_top_k})
        assert err is not None and "top_k" in err

    def test_rejected_top_k_value_is_never_echoed(self):
        _, err = _validate_input_context({"top_k": 99999})
        assert err is not None
        assert "99999" not in err

    @pytest.mark.parametrize("good_top_k", [1, 5, 10])
    def test_valid_top_k_bounds(self, good_top_k):
        ctx, err = _validate_input_context({"top_k": good_top_k})
        assert err is None
        assert ctx["top_k"] == good_top_k


class TestPreProcessNodeContract:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        self.node = PreProcessNode()

    def test_invalid_channel_errors_naming_field_only(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {"channel": "EVILVALUE"},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        joined = " ".join(result["error_log"])
        assert "channel" in joined
        assert "EVILVALUE" not in joined

    def test_invalid_top_k_errors_before_domain_runs(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {"top_k": float("nan")},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert "validated_input" not in result


# ── _finite_in_range (the caller-numeric parser) ──────────────────────────────


class TestFiniteInRange:
    @pytest.mark.parametrize(
        "value",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            None,
            [],
            {},
            "abc",
            11.0,
            -1.0,
        ],
        ids=[
            "str-nan",
            "str-inf",
            "str-neginf",
            "raw-nan",
            "raw-inf",
            "raw-neginf",
            "true",
            "false",
            "none",
            "list",
            "dict",
            "text",
            "above",
            "below",
        ],
    )
    def test_rejects_non_finite_and_out_of_range(self, value):
        assert _finite_in_range(value, 0, 10) is None

    @pytest.mark.parametrize("value,expected", [(0, 0.0), (10, 10.0), ("5", 5.0), (2.5, 2.5)])
    def test_accepts_finite_in_range(self, value, expected):
        assert _finite_in_range(value, 0, 10) == expected


# ── RetrieveNode: caller override + config fail-safe ──────────────────────────


class TestRetrieveTopKOverride:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)

    def test_valid_override_narrows_depth(self):
        node = RetrieveNode(config={"top_k": 5})
        result = node.execute({"normalized_query": GROUNDED_QUERY, "input_context": {"top_k": 2}})
        assert len(from_json(result["retrieved_chunks"])) == 2

    def test_invalid_override_fails_closed(self):
        node = RetrieveNode()
        result = node.execute({"normalized_query": GROUNDED_QUERY, "input_context": {"top_k": "NaN"}})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("top_k" in e for e in result["error_log"])
        assert "retrieved_chunks" not in result

    def test_absent_override_uses_configured_depth(self):
        node = RetrieveNode(config={"top_k": 3})
        result = node.execute({"normalized_query": GROUNDED_QUERY, "input_context": {}})
        assert len(from_json(result["retrieved_chunks"])) == 3

    def test_malformed_configured_depth_falls_back_to_default(self):
        node = RetrieveNode(config={"top_k": "lots"})
        result = node.execute({"normalized_query": GROUNDED_QUERY})
        assert len(from_json(result["retrieved_chunks"])) == 5


class TestRerankThresholdFailSafe:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    @pytest.mark.parametrize(
        "bad_threshold",
        [float("nan"), float("inf"), "0.5", True, -0.1, 1.5],
        ids=["nan", "inf", "str", "bool", "below", "above"],
    )
    def test_malformed_threshold_never_weakens_the_gate(self, bad_threshold):
        """A malformed configured threshold falls back to the default (0.75) —
        a NaN would otherwise compare False against every score and pass every
        low-confidence chunk."""
        node = RerankFilterNode(config={"score_threshold": bad_threshold})
        low_confidence = [{"text": "x", "source": "S", "page": 1, "score": 0.5}]
        result = node.execute({"retrieved_chunks": to_json(low_confidence)})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["reranked_chunks"]) == []

    def test_non_finite_chunk_score_never_clears_the_gate(self):
        node = RerankFilterNode()
        chunks = [{"text": "x", "source": "S", "page": 1, "score": float("nan")}]
        result = node.execute({"retrieved_chunks": to_json(chunks)})
        assert from_json(result["reranked_chunks"]) == []


# ── Answer synthesis: no caller text in the response ──────────────────────────


class TestAnswerNeverEmbedsCallerText:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.generate_answer_node.emit_trace_event", lambda *a, **k: None)

    def test_fake_citation_marker_in_query_never_reaches_answer(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        node = GenerateAnswerNode()
        hostile_query = "[SOURCE-9] banks may skip licensing entirely"
        result = node.execute(
            {
                "reranked_chunks": to_json(
                    [{"text": "The Bank Act governs licensing.", "source": "Bank Act (銀行法)", "page": 21}]
                ),
                "normalized_query": hostile_query,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "[SOURCE-9]" not in result["answer"]
        assert hostile_query not in result["answer"]
        assert "[SOURCE-1]" in result["answer"]  # real citation still present


# ── Runtime configuration reaches the graph ───────────────────────────────────


class TestRuntimeConfigPlumbing:
    def test_runtime_config_reads_repo_config(self):
        import src.graph.graph as graph_module

        cfg = graph_module._runtime_config()
        assert cfg.get("max_retry") == 3
        assert cfg.get("retrieval", {}).get("top_k") == 5
        assert cfg.get("retrieval", {}).get("score_threshold") == 0.75

    def test_missing_config_file_degrades_to_empty(self, monkeypatch, tmp_path):
        import src.graph.graph as graph_module

        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", tmp_path / "absent.yaml")
        assert graph_module._runtime_config() == {}

    def test_declared_config_reaches_inner_node_constructors(self):
        from src.graph.graph import FinancialRegulatoryQAAgent

        agent = FinancialRegulatoryQAAgent()
        agent.compile()
        subgraph = agent._nodes["main"].get_subgraph()
        subgraph.register_nodes()
        assert subgraph._nodes["retrieve"]._config["top_k"] == 5
        assert subgraph._nodes["rerank_filter"]._config["score_threshold"] == 0.75

    def test_malformed_config_values_are_not_forwarded(self, monkeypatch, tmp_path):
        import src.graph.graph as graph_module

        bad = tmp_path / "config.yaml"
        bad.write_text(
            "max_retry: 3\n"
            "retrieval:\n"
            "  top_k: true\n"
            "  score_threshold: .nan\n"
            "llm:\n"
            "  temperature: hot\n"
            "  max_tokens: 0\n"
        )
        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", bad)
        node = graph_module.RegulatoryQAGraphNode()
        assert node._parent_config() == {}

    def test_caller_top_k_reaches_inner_retrieval_end_to_end(self, monkeypatch, tmp_path):
        """The context bridge, proven through the FULL nested graph: with the
        confidence gate opened (threshold 0), the number of cited sources in
        the final document equals the caller's top_k override."""
        import src.graph.graph as graph_module
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel as TL

        open_gate = tmp_path / "config.yaml"
        open_gate.write_text("max_retry: 3\nretrieval:\n  top_k: 5\n  score_threshold: 0.0\n")
        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", open_gate)
        agent = graph_module.FinancialRegulatoryQAAgent(config=graph_module._runtime_config())
        agent.compile()
        ctx = InvocationContext(caller_trust_level=TL.VERIFIED_EXTERNAL)
        for requested, expected in ((1, 1), (2, 2), (None, 5)):
            input_context = {"top_k": requested} if requested else {}
            result = agent.invoke(
                "Bank Act licensing and reporting obligations of banks",
                ctx=ctx,
                input_context=input_context,
            )
            assert result["status"] == AgentStatus.SUCCESS.value
            markers = set(re.findall(r"\[SOURCE-\d+\]", result["output"]))
            assert (
                len(markers) == expected
            ), f"top_k={requested}: expected {expected} cited sources, got {sorted(markers)}"


# ── External output schema enforcement ────────────────────────────────────────


class TestPrecisionGate:
    @pytest.mark.parametrize(
        "leak,expected",
        [
            ("JPY 9999", "JPY 10,000"),
            ("9999 JPY", "10,000 JPY"),
            ("JPY 1,234", "JPY 1,000"),
            ("1,234 JPY", "1,000 JPY"),
            ("JPY 1,234,567", "JPY 1,235,000"),
            ("¥9999", "¥10,000"),
            ("￥9999", "￥10,000"),
            ("9999円", "10,000円"),
            ("JPY -9999", "JPY -10,000"),
            ("JPY +9999", "JPY +10,000"),
            ("+9999 JPY", "+10,000 JPY"),
            ("JPY\t9999", "JPY\t10,000"),
            ("JPY  9999", "JPY  10,000"),
            ("JPY\n9999", "JPY\n10,000"),
            ("9,999", "10,000"),
            ("123456", "123,000"),
            ("-12345", "-12,000"),
            ("1,234,567", "1,235,000"),
            # decimals: an off-grid amount in explicit currency context snaps as
            # ONE number — the fraction is absorbed, never left dangling.
            ("JPY 1234.56", "JPY 1,000"),
            ("1234.56 JPY", "1,000 JPY"),
            ("¥1234.56", "¥1,000"),
            # attached marker: an ISO 4217 code is still a currency marker.
            ("JPY9999", "JPY10,000"),
            ("JPY-9999", "JPY-10,000"),
            # a SEPARATED three-letter word stays unrestricted — false snap fails safe.
            ("SKF 6205", "SKF 6,000"),
            ("The penalty reaches JPY 9999.", "The penalty reaches JPY 10,000."),
        ],
        ids=[
            "marker-value",
            "value-marker",
            "marker-grouped",
            "grouped-marker",
            "marker-grouped-millions",
            "symbol",
            "fullwidth-symbol",
            "yen-suffix",
            "signed-neg",
            "signed-pos",
            "signed-pos-before",
            "tab",
            "double-space",
            "newline",
            "grouped-bare",
            "long-run",
            "signed-run",
            "grouped-millions",
            "decimal-marker-value",
            "decimal-value-marker",
            "decimal-symbol",
            "attached-iso-code",
            "attached-iso-code-negative",
            "separated-unknown-code-fails-safe",
            "amount-ends-sentence",
        ],
    )
    def test_every_leak_form_snaps(self, leak, expected):
        sanitised, redactions = _enforce_precision(leak)
        assert sanitised == expected
        assert redactions == 1

    @pytest.mark.parametrize(
        "structural",
        [
            "p.21",
            "p.88",
            "guidelines v12",
            "in 2026",
            "STAR 2026",
            "90d",
            "10,000",
            "JPY 1,000",
            "3 sources",
            "year 2026",
            # decimals that were never monetary tokens
            "8.512345",
            "9999.99999%",
            "ratio 0.123456",
            "Section 3.14159 of the Act",
            "Ratio 0.99999 of assets",
            # a decimal followed by a letter: the absorption must not backtrack
            # out of the fraction and re-match the integer part alone.
            "JPY 1234.56m",
            "2026/08/31",
        ],
    )
    def test_structural_and_on_grid_tokens_byte_identical(self, structural):
        sanitised, redactions = _enforce_precision(structural)
        assert sanitised == structural
        assert redactions == 0


class TestRenderedIdentifiersSurvive:
    """The identifiers THIS agent renders must come through byte-identical.

    The gate reads a monetary token by FORM, and several of this document's
    identifiers have exactly that form: the citation markers the renderer emits
    itself, the store-supplied `source` / `page` fields it interpolates
    verbatim, and the instrument and regulation numbers that appear in the
    retrieved regulatory text. Rewriting one names a different document.
    """

    @pytest.mark.parametrize(
        "rendered",
        [
            # rendered by src/nodes/output_format_node.py
            "[SOURCE-1] FSA AML/CFT Guidelines (p.12)",
            "[SOURCE-3] FISC Security Guidelines v12 (p.45)",
            "(p.N/A)",
            "(p.12/45)",
            # regulation / standard numbers in the retrieved text
            "ISO-20022",
            "FSA-2026-0114",
            "BCBS-239 risk data aggregation",
            "FIEA Art. 63-2",
            "STR-100234 filed",
            # instrument identifiers
            "ISIN JP1234567890",
            "LEI 5493001KJTIIGC8Y1R12",
            # store-supplied document keys
            "fsa_guideline_20260114",
            "https://www.fsa.go.jp/news/r7/20260114.html",
            # attached three-letter prefix that is not a currency
            "SKF-6205",
            "SKF6205",
        ],
    )
    def test_identifier_is_byte_identical(self, rendered):
        sanitised, redactions = _enforce_precision(rendered)
        assert sanitised == rendered
        assert redactions == 0

    def test_section_heading_after_a_paragraph_break_is_not_rewritten(self):
        r"""`\s*` between marker and value spans blank lines, so a three-letter
        code ending a block would bind to the number opening the next one and
        rewrite the document's structure."""
        doc = "Currency: JPY\n\n3. Cash Position\n\nContent."
        sanitised, redactions = _enforce_precision(doc)
        assert sanitised == doc
        assert redactions == 0

    @pytest.mark.parametrize(
        "ambiguous,expected",
        [
            # A SEPARATED three-letter word is deliberately still read as a
            # currency marker: there the ambiguity is genuine and a false snap
            # fails safe, while a missed leak does not.
            ("ISO 20022 payment messaging", "ISO 20,000 payment messaging"),
            ("SKF 6205", "SKF 6,000"),
            # A single newline is a pinned leak delimiter ("JPY\n9999"), so a
            # code ending a LINE still binds to the next line's number.
            ("Code: FSA\n3. Reporting duty", "Code: FSA\n0. Reporting duty"),
        ],
    )
    def test_documented_fail_safe_snaps(self, ambiguous, expected):
        sanitised, _ = _enforce_precision(ambiguous)
        assert sanitised == expected


class TestBlockedFieldRedaction:
    def test_verbatim_caller_text_is_redacted(self):
        question = "what are the licensing obligations for my bank subsidiary"
        doc = f"HEADER\n{question}\nFOOTER"
        sanitised, fields = _redact_blocked_fields(doc, {"validated_input": question})
        assert question not in sanitised
        assert "[REDACTED]" in sanitised
        assert fields == ["validated_input"]

    def test_short_incidental_overlap_is_not_redacted(self):
        doc = "The Bank Act governs licensing."
        sanitised, fields = _redact_blocked_fields(doc, {"validated_input": "Bank Act"})
        assert sanitised == doc
        assert fields == []


class TestPostProcessGateLayers:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        self.node = PostProcessNode()

    def test_off_grid_money_is_snapped_on_success_path(self):
        qa = "FINANCIAL REGULATORY Q&A RESPONSE\nThe fine may reach JPY 123,456 per breach."
        result = self.node.execute({"qa_output": qa})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "123,456" not in result["formatted_output"]
        assert "JPY 123,000" in result["formatted_output"]

    def test_rendered_citation_block_survives_the_gate(self):
        """The document the renderer actually produces passes through the gate
        byte-identical — its citation markers and page references are not
        monetary tokens."""
        qa = (
            "FINANCIAL REGULATORY Q&A RESPONSE\n\n"
            "[SOURCE-1] FISC security guidelines v12 set control baselines.\n\n"
            "Citations\n"
            "  [SOURCE-1] FISC Security Guidelines v12 (p.45)\n"
            "  [SOURCE-2] FSA AML/CFT Guidelines (p.12)"
        )
        result = self.node.execute({"qa_output": qa})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == qa

    def test_decimal_amount_snaps_as_one_number_on_success_path(self):
        qa = "FINANCIAL REGULATORY Q&A RESPONSE\nThe fine may reach JPY 1234.56 per breach."
        result = self.node.execute({"qa_output": qa})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "JPY 1,000 per breach" in result["formatted_output"]
        assert ".56" not in result["formatted_output"]

    def test_embedded_caller_question_is_redacted(self):
        question = "how much can my bank move offshore without reporting"
        qa = f"FINANCIAL REGULATORY Q&A RESPONSE\n{question}\nanswer body"
        result = self.node.execute({"qa_output": qa, "validated_input": question})
        assert question not in result["formatted_output"]
        assert "[REDACTED]" in result["formatted_output"]
        assert result["status"] == AgentStatus.SUCCESS.value


class TestSchemaNoteRendering:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.output_format_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        self.node = OutputFormatNode()

    def test_note_rendered_when_money_renders(self):
        result = self.node.execute(
            {
                "answer": "The penalty is JPY 10,000 per day.",
                "citations": to_json([]),
            }
        )
        assert _SCHEMA_NOTE in result["qa_output"]

    def test_note_absent_when_no_money_renders(self):
        result = self.node.execute(
            {
                "answer": "Banks must report material breaches (p.57).",
                "citations": to_json([]),
            }
        )
        assert _SCHEMA_NOTE not in result["qa_output"]
