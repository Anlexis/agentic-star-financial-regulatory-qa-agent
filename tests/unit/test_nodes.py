# FIN-C2-001 — Unit Tests: domain nodes + graph wiring
#
# Real, non-stub unit tests. They import the REAL modules and assert real
# behaviour: the financial-regulatory RAG pipeline
# (query validation, term-overlap retrieval, score-threshold rerank, grounded
# answer synthesis with [SOURCE-N] citations, abstention on insufficient
# grounding, the output gate) and the Cat 2 two-layer nested graph.
#
# Audit events are patched at the node MODULE level (not via a sys.modules
# stub, which would break the real `shared` package the framework loads at import
# time). Patch pattern per node:
#     monkeypatch.setattr("src.nodes.<mod>.emit_trace_event", lambda *a, **k: None)

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.schemas.state import from_json, to_json


# A financial-regulatory compliance query that grounds to the bundled corpus's
# Bank Act chunk (term-overlap score 1.0, above the 0.75 rerank threshold).
# Lower-cased so it carries no PII "name" bigram: routed through BaseNode.__call__
# the framework input gate would otherwise mask a Title-Case bigram ("Bank Act") in the
# user_input / validated_input fields. Retrieval tokenises case-insensitively, so
# the term-overlap score (and every grounding assertion) is unchanged.
GROUNDED_QUERY = "bank act licensing and reporting obligations of banks"

# An off-topic query with zero term overlap against the corpus → all chunks
# fall below the score threshold → abstention (insufficient grounding).
OFF_TOPIC_QUERY = "photosynthesis chlorophyll sunlight recipe"


def _chunk(
    text="The Bank Act governs licensing and reporting obligations of banks.",
    source="Bank Act (銀行法)",
    page=21,
    score=1.0,
) -> dict:
    return {"text": text, "source": source, "page": page, "score": score}


# ── PreProcessNode (outer pre_process, VERIFIED_EXTERNAL) ──────────────────


class TestPreProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_valid_query_returns_success(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUERY

    def test_normalises_control_chars_and_whitespace(self):
        messy = "  Bank\x00Act   licensing\t\treporting  "
        result = self.node(
            {"user_input": messy, "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        # control char stripped, runs of whitespace collapsed to single spaces, trimmed
        assert result["validated_input"] == "BankAct licensing reporting"

    def test_enriched_context_carries_channel(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {"channel": "compliance_desk"},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        ctx = from_json(result["enriched_context"])
        assert ctx["source"] == "FinancialRegulatoryQAAgent"
        assert ctx["channel"] == "compliance_desk"

    def test_enriched_context_defaults_channel_unknown(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert from_json(result["enriched_context"])["channel"] == "unknown"

    def test_empty_input_returns_error(self):
        result = self.node(
            {"user_input": "", "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e for e in result["error_log"])

    def test_whitespace_only_input_returns_error(self):
        result = self.node(
            {"user_input": "   \t  ", "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.ERROR.value

    def test_over_length_query_returns_error(self):
        result = self.node(
            {"user_input": "a" * 2001, "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("exceeds" in e or "2000" in e for e in result["error_log"])

    def test_trust_level_is_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_execute_signature_is_state_first(self):
        import inspect
        from src.nodes.pre_process_node import PreProcessNode

        params = list(inspect.signature(PreProcessNode.execute).parameters.keys())
        assert params[0] == "self" and params[1] == "state"
        assert "_invoke_impl" not in PreProcessNode.__dict__


# ── InputValidateNode (inner domain node 1, ANONYMOUS) ─────────────────────────


class TestInputValidateNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.input_validate_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def test_plain_query_passes_through(self):
        result = self.node({"validated_input": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        # no known acronym → query is returned unchanged (whitespace-collapsed)
        assert result["normalized_query"] == GROUNDED_QUERY

    def test_expands_known_acronyms(self):
        result = self.node(
            {
                "validated_input": "What are the FSA and AML obligations?",
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        nq = result["normalized_query"]
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "Financial Services Agency" in nq
        assert "Anti-Money Laundering" in nq

    def test_does_not_expand_unknown_acronym(self):
        result = self.node(
            {"validated_input": "quarterly disclosure timeline", "caller_trust_level": TrustLevel.ANONYMOUS.value}
        )
        assert result["normalized_query"] == "quarterly disclosure timeline"

    def test_falls_back_to_user_input(self):
        result = self.node({"user_input": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["normalized_query"] == GROUNDED_QUERY

    def test_empty_query_returns_error(self):
        result = self.node({"validated_input": "   ", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RetrieveNode (inner domain node 2, ANONYMOUS) ──────────────────────────────


class TestRetrieveNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.retrieve_node import RetrieveNode

        self.node = RetrieveNode()

    def test_grounded_query_scores_bank_act_top(self):
        result = self.node({"normalized_query": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        chunks = from_json(result["retrieved_chunks"])
        # default top_k = 5; results ordered by score desc
        assert len(chunks) == 5
        top = chunks[0]
        assert top["source"] == "Bank Act (銀行法)"
        assert top["score"] == 1.0
        # each chunk carries the full contract shape
        assert set(top.keys()) == {"text", "source", "page", "score"}
        assert 0.0 <= top["score"] <= 1.0

    def test_scores_are_bounded_and_monotonic(self):
        chunks = from_json(
            self.node({"normalized_query": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})[
                "retrieved_chunks"
            ]
        )
        scores = [c["score"] for c in chunks]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_respects_top_k_config(self):
        """top_k arrives through the constructor — execute() takes state only."""
        from src.nodes.retrieve_node import RetrieveNode

        node = RetrieveNode(config={"top_k": 2})
        result = node.execute({"normalized_query": GROUNDED_QUERY})
        assert len(from_json(result["retrieved_chunks"])) == 2

    def test_defaults_to_module_top_k_without_config(self):
        result = self.node.execute({"normalized_query": GROUNDED_QUERY})
        assert len(from_json(result["retrieved_chunks"])) == 5

    def test_falls_back_to_validated_input(self):
        result = self.node({"validated_input": GROUNDED_QUERY, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["retrieved_chunks"])[0]["source"] == "Bank Act (銀行法)"

    def test_empty_query_returns_error(self):
        result = self.node({"normalized_query": "", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("query" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RerankFilterNode (inner domain node 3, ANONYMOUS) ──────────────────────────


class TestRerankFilterNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        self.node = RerankFilterNode()

    def _state(self, chunks):
        return {"retrieved_chunks": to_json(chunks), "caller_trust_level": TrustLevel.ANONYMOUS.value}

    def test_filters_below_threshold_and_dedupes(self):
        retrieved = [
            {"text": "A", "source": "S1", "page": 1, "score": 0.90},
            {"text": "B", "source": "S2", "page": 2, "score": 0.50},  # below 0.75 → dropped
            {"text": "A-dup", "source": "S1", "page": 1, "score": 0.80},  # dup (S1,1) → dropped
            {"text": "C", "source": "S3", "page": 3, "score": 0.76},
        ]
        result = self.node(self._state(retrieved))
        assert result["status"] == AgentStatus.SUCCESS.value
        reranked = from_json(result["reranked_chunks"])
        assert [c["source"] for c in reranked] == ["S1", "S3"]
        # reranked by score descending
        assert [c["score"] for c in reranked] == [0.90, 0.76]

    def test_all_below_threshold_yields_empty_success(self):
        retrieved = [{"text": "x", "source": "S", "page": 1, "score": 0.10}]
        result = self.node(self._state(retrieved))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["reranked_chunks"]) == []

    def test_custom_threshold_is_honoured(self):
        """score_threshold arrives through the constructor — execute() takes state only."""
        from src.nodes.rerank_filter_node import RerankFilterNode

        retrieved = [{"text": "x", "source": "S", "page": 1, "score": 0.60}]
        node = RerankFilterNode(config={"score_threshold": 0.5})
        result = node.execute(self._state(retrieved))
        assert len(from_json(result["reranked_chunks"])) == 1

    def test_missing_retrieved_chunks_returns_error(self):
        result = self.node({"caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("retrieved_chunks" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── GenerateAnswerNode (inner domain node 4, ANONYMOUS) ────────────────────────


class TestGenerateAnswerNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.generate_answer_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        self.node = GenerateAnswerNode()

    def test_grounded_answer_cites_sources(self):
        state = {
            "reranked_chunks": to_json([_chunk()]),
            "normalized_query": GROUNDED_QUERY,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["answer"]
        assert "[SOURCE-1]" in answer
        assert "The Bank Act governs licensing" in answer
        # Profile A mandatory advisory disclaimer present
        assert "法的アドバイスではありません" in answer
        citations = from_json(result["citations"])
        assert len(citations) == 1
        assert citations[0] == {"marker": "SOURCE-1", "source": "Bank Act (銀行法)", "page": 21}

    def test_multiple_chunks_produce_multiple_citations(self):
        state = {
            "reranked_chunks": to_json(
                [
                    _chunk(text="Chunk one.", source="Src A", page=1),
                    _chunk(text="Chunk two.", source="Src B", page=2),
                ]
            ),
            "normalized_query": GROUNDED_QUERY,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert "[SOURCE-1]" in result["answer"]
        assert "[SOURCE-2]" in result["answer"]
        assert len(from_json(result["citations"])) == 2

    def test_abstains_when_no_grounding(self):
        result = self.node(
            {
                "reranked_chunks": to_json([]),
                "normalized_query": OFF_TOPIC_QUERY,
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        # abstention is still a SUCCESS path — it just declines to fabricate
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain enough" in result["answer"]
        assert "法的アドバイスではありません" in result["answer"]
        # NO fabricated citations when grounding is insufficient
        assert from_json(result["citations"]) == []

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── OutputFormatNode (inner domain node 5, ANONYMOUS) ──────────────────────────


class TestOutputFormatNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.output_format_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.output_format_node import OutputFormatNode

        self.node = OutputFormatNode()

    def test_assembles_full_qa_document(self):
        state = {
            "answer": "A grounded regulatory answer.\n\n※ disclaimer",
            "citations": to_json([{"marker": "SOURCE-1", "source": "Bank Act (銀行法)", "page": 21}]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        qa = result["qa_output"]
        assert result["result"] == qa
        assert "FINANCIAL REGULATORY Q&A RESPONSE" in qa
        assert "Citations" in qa
        assert "[SOURCE-1] Bank Act (銀行法) (p.21)" in qa

    def test_no_citations_omits_citation_block(self):
        state = {
            "answer": "An answer with no sources.",
            "citations": to_json([]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        qa = self.node(state)["qa_output"]
        assert "FINANCIAL REGULATORY Q&A RESPONSE" in qa
        assert "Citations" not in qa

    def test_missing_answer_returns_error(self):
        result = self.node({"answer": "", "citations": to_json([]), "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("answer" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── PostProcessNode (outer post_process, output gate, ANONYMOUS) ───────────


class TestPostProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    def test_clean_output_passes_gate(self):
        qa = "FINANCIAL REGULATORY Q&A RESPONSE\nA clean grounded compliance answer."
        result = self.node({"qa_output": qa, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == qa
        assert result["result"] == qa

    def test_empty_output_uses_fallback(self):
        result = self.node({"qa_output": "", "answer": "", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No answer content generated" in result["formatted_output"]

    def test_gate_redacts_credential_leak(self):
        leaky = "FINANCIAL REGULATORY Q&A RESPONSE\ntoken=sk-abcdefghij0123456789ABCDEF"
        result = self.node({"qa_output": leaky, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert "REDACTED" in result["formatted_output"]
        assert result["formatted_output"] == result["result"]
        assert any("credential pattern" in e for e in result["error_log"])

    def test_security_gate_output_helper_detects_and_clears(self):
        from src.nodes.post_process_node import _security_gate_output

        assert _security_gate_output("sk-abcdefghij0123456789ABCDEF") is not None
        assert _security_gate_output("Bearer abcdefgh12345678") is not None
        assert _security_gate_output("password = supersecret123") is not None
        assert _security_gate_output("A perfectly clean regulatory answer.") is None

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── Graph wiring: outer AgentBaseGraph + inner BaseGraph (Cat 2 nested) ─────────


class TestOuterGraphComposition:
    def test_registers_five_backbone_slots(self):
        from src.graph.graph import FinancialRegulatoryQAAgent, RegulatoryQAGraphNode
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode

        agent = FinancialRegulatoryQAAgent()
        agent.compile()
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], RegulatoryQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.graph import FinancialRegulatoryQAAgent

        agent = FinancialRegulatoryQAAgent()
        assert agent.name == "FinancialRegulatoryQAAgent"
        assert agent.state_schema is State

    def test_main_slot_graphnode_contracts(self):
        from src.graph.graph import RegulatoryQAGraphNode

        node = RegulatoryQAGraphNode()
        assert node.error_strategy == "propagate"
        assert node.propagate_hitl is False
        # extract_input prefers validated_input, falls back to user_input
        assert node.extract_input({"validated_input": "V", "user_input": "U"}) == "V"
        assert node.extract_input({"user_input": "U"}) == "U"

    def test_merge_output_maps_subresult_keys(self):
        from src.graph.graph import RegulatoryQAGraphNode

        node = RegulatoryQAGraphNode()
        sub_result = {
            "answer": "ANSWER",
            "citations": "[]",
            "qa_output": "QA-OUTPUT",
            "status": AgentStatus.SUCCESS.value,
            "node_history": ["x"],  # not forwarded by merge_output
            "correlation_id": "cid-1",  # not forwarded by merge_output
        }
        delta = node.merge_output({}, sub_result)
        assert delta["answer"] == "ANSWER"
        assert delta["qa_output"] == "QA-OUTPUT"
        assert delta["status"] == AgentStatus.SUCCESS.value
        assert set(delta.keys()) == {"answer", "citations", "qa_output", "status"}

    def test_agent_class_output_gate(self):
        from src.graph.graph import FinancialRegulatoryQAAgent

        agent = FinancialRegulatoryQAAgent()
        assert agent._security_gate_output("token=sk-abcdefghij0123456789ABCDEF") is not None
        assert agent._security_gate_output("A perfectly clean regulatory answer.") is None


class TestInnerDomainGraph:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        for mod in (
            "input_validate_node",
            "retrieve_node",
            "rerank_filter_node",
            "generate_answer_node",
            "output_format_node",
        ):
            monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)

    def test_registers_five_domain_nodes(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.register_nodes()
        assert set(g._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        assert g.name == "fin_c2_001_regulatory_qa_workflow"
        assert g.state_schema is State

    def test_inner_graph_grounds_answer(self):
        """Standalone inner-graph invoke (ANONYMOUS caller) runs the RAG pipeline
        and grounds the answer to a retrieved source."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(GROUNDED_QUERY, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "FINANCIAL REGULATORY Q&A RESPONSE" in result["qa_output"]
        assert "[SOURCE-1]" in result["answer"]
        assert "Bank Act" in result["qa_output"]

    def test_inner_graph_abstains_off_topic(self):
        """An off-topic query grounds to nothing → insufficient-grounding answer,
        no fabricated citations (still a SUCCESS path)."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(OFF_TOPIC_QUERY, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain enough" in result["answer"]
        assert from_json(result["citations"]) == []

    def test_manifest_config_reaches_node_constructors(self):
        """Graph-level config is injected into the config-driven node ctors.

        The node contract forbids a config parameter on execute(), so the
        settings forwarded by RegulatoryQAGraphNode._parent_config() must arrive
        through the constructors instead.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph(config={"top_k": 2, "score_threshold": 0.5})
        g.register_nodes()
        assert g._nodes["retrieve"]._config["top_k"] == 2
        assert g._nodes["rerank_filter"]._config["score_threshold"] == 0.5


# ── execute() contract (state-only signature) ─────────────────────────────


class TestExecuteContract:
    """Every FunctionNode must expose exactly `execute(self, state) -> dict`.

    The node contract permits no extra parameters: BaseNode.__call__ invokes
    execute(state) with a single argument, so a `config` parameter is both a
    contract violation and dead at runtime. Static configuration belongs in the
    node constructor (see DomainWorkflowGraph.register_nodes).
    """

    def test_all_nodes_take_state_only(self):
        import inspect

        from src.nodes.generate_answer_node import GenerateAnswerNode
        from src.nodes.input_validate_node import InputValidateNode
        from src.nodes.main_node import MainNode
        from src.nodes.output_format_node import OutputFormatNode
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode
        from src.nodes.rerank_filter_node import RerankFilterNode
        from src.nodes.retrieve_node import RetrieveNode

        for node_cls in (
            PreProcessNode,
            MainNode,
            PostProcessNode,
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            params = list(inspect.signature(node_cls.execute).parameters.keys())
            assert params == [
                "self",
                "state",
            ], f"{node_cls.__name__}.execute must be execute(self, state), got {params}"
            assert "_invoke_impl" not in node_cls.__dict__


# ── Trust gate (BaseNode.__call__ enforcement) ────────────────────────────


class TestTrustGate:
    """The trust gate lives in BaseNode.__call__ and runs BEFORE execute().

    Unit tests that call node.execute(state) directly bypass this
    gate. Every node in this suite is now invoked through __call__ (node(state))
    with an explicit caller_trust_level so the gate is exercised. These two tests
    assert the gate's denial and admission behaviour directly on the only node
    that requires elevated trust — PreProcessNode (VERIFIED_EXTERNAL).
    """

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def test_anonymous_caller_denied_before_execute(self):
        """An ANONYMOUS caller is denied by the __call__ trust gate before execute()
        runs (fail-closed; no exception raised)."""
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any("trust gate denied" in e.lower() for e in result["error_log"])
        # execute() never ran, so its output key is absent from the denial result
        assert "validated_input" not in result

    def test_verified_external_caller_admitted(self):
        """A VERIFIED_EXTERNAL caller clears the trust gate and execute() runs to SUCCESS."""
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "user_input": GROUNDED_QUERY,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUERY
