# Test Specification — FIN-C2-001 Financial Regulatory Q&A Agent

## 1. Test Strategy

- **Agent:** FIN-C2-001 — Financial Regulatory Q&A Agent (Cat 2, RAG pattern,
  two-layer nested graph: outer `AgentBaseGraph` backbone + inner
  `DomainWorkflowGraph` `BaseGraph`).
- **Coverage target:** ≥ 90% of `src/nodes/` + `src/graph/` branches.
- **Test types:** Unit (per node + graph wiring + caller-data contract +
  output schema) · Proof-of-Boundary (framework security / serialization
  contracts + entry-point auth + end-to-end `/invoke`) · Backbone invoke
  (full `FinancialRegulatoryQAAgent().invoke()`).
- **Framework provisioning:** `framework` (agenticstar-agentcore) is supplied
  by the CI environment — the published wheel from the package registry. Tests
  import the REAL modules; there are no stub nodes.
- **Audit events:** `emit_trace_event` is patched at the node module level in
  unit tests to avoid audit-backend calls, never via a `sys.modules` stub
  (which would break the real `shared` package the framework loads at import
  time).

### RAG grounding contract (what "correct" means here)

FIN-C2-001 is a regulated-domain retrieval-augmented Q&A agent. Two
behaviours are the core acceptance criteria:

1. **Grounding** — when the query overlaps the regulatory corpus, the answer is
   synthesised ONLY from retrieved chunks that clear the rerank score threshold,
   carries `[SOURCE-N]` citation markers, and returns a matching `citations`
   list. The agent never emits regulatory text it cannot cite — and never
   embeds caller-supplied text in the response, so a request cannot smuggle
   pseudo-citations into the document.
2. **Abstention (anti-hallucination)** — when no retrieved chunk clears the
   score threshold (`retrieval.score_threshold = 0.75`), `GenerateAnswerNode`
   returns an explicit *insufficient grounding* message plus the mandatory
   advisory disclaimer, and emits **zero** citations rather than fabricating a
   regulatory answer. Abstention is a SUCCESS path — declining to answer is the
   correct outcome, not an error.

The bundled retriever is a deterministic term-overlap scorer over a sample
corpus (FSA / FISC / AML / FIEA / Bank Act material); production swaps in a real
vector store via `config retrieval.vector_store` with the same node contract.

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_nodes.py` | All 7 domain/backbone nodes + outer & inner graph wiring; grounding + abstention |
| `tests/unit/test_caller_data_and_output_schema.py` | input_context contract (bounds, non-finite matrix, no value echo) · runtime-config plumbing + context bridge · output-gate layers (precision grid both ways, caller-text redaction, schema note) |
| `tests/unit/test_main_node.py` | Cat 1 reference `MainNode` — `execute()` contract kept green |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | TC-06/TC-07 — framework input/output gates are non-bypassable |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6 per-node + backbone invoke order (VERIFIED_EXTERNAL) + trust gate + payload alignment |
| `tests/proof_of_boundary/test_server_boot.py` | Entry-point auth boundary (Bearer token, trust elevation, adapter size cap) |
| `tests/proof_of_boundary/test_invoke_e2e.py` | End-to-end business behaviour through the real ASGI `POST /invoke` |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4 platform-SDK import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5 State msgpack/credential safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7 HITL interrupt-propagation (skip stub — no cross-boundary HITL) |

### Canonical valid payload (PB-6 `_VALID_PAYLOAD`)

The RAG input is a plain compliance-question **string** (not a JSON object). The
backbone invoke test and `deploy/invoke_payload.json` MUST use the identical
string (asserted by `test_invoke_payload_matches_pb6`):

```
Bank Act licensing and reporting obligations of banks
```

This query is term-aligned with the corpus's Bank Act chunk (overlap score
`1.0`, above the `0.75` rerank threshold), so retrieval grounds the answer to a
single cited source (`[SOURCE-1] Bank Act (銀行法) (p.21)`) and the full
end-to-end invoke returns a `FINANCIAL REGULATORY Q&A RESPONSE` document with
`status = success`.

## 2. Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected Result | Where |
|-------|------|----------------|-------|
| TC-01 | State contract: flat `TypedDict`, domain fields optional `Optional[str]`, no Pydantic/dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Empty / whitespace / over-length (`> 2000`) query rejected at PreProcessNode | `status=error`, error_log populated | `TestPreProcessNode` |
| TC-03 | No JWT/credential in State | credential scan: 0 violations | CI + `test_state_safety.py` |
| TC-04 | `execute(self, state)` contract — no extra parameters, no `_invoke_impl` | Signature is exactly `(self, state)` for all 8 nodes, `_invoke_impl` absent | `TestExecuteContract`, `test_execute_signature_is_state_first`, `test_main_node.py` |
| TC-05 | Audit: `emit_trace_event()` called inside each node `execute()` (positional form) | ≥1 domain event per node | code inspection + patched-emit fixtures |
| TC-06 | Framework input gate cannot be overridden by a node subclass | TypeError at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-07 | Framework output gate cannot be overridden by a node subclass | TypeError at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | Trust gate: `required_trust_level` enforced in `__call__` before `execute()` | ANONYMOUS caller → refused; VERIFIED_EXTERNAL → admitted | `TestTrustGate` |
| TC-08a | Outer `PreProcessNode` = VERIFIED_EXTERNAL; inner nodes + post_process = ANONYMOUS | trust levels asserted per node | `test_trust_level_*` |
| TC-11 | Output gate on post_process (+ agent-class canonical gate) | credential pattern → redacted + `status=error`; clean → pass | `TestPostProcessNode`, `test_agent_class_output_gate` |

## 3. Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Test | Expected Result | Where |
|-------|----------|------|----------------|-------|
| PB-2 | State serialization | AST scan of `src/schemas/state.py` | primitives only; no Pydantic/dataclass | `test_state_safety.py` |
| PB-4 | Import isolation | AST scan of `src/` | 0 platform-SDK (`agenticstar` / platform) imports | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named fields / prohibited types in State | inspection pass | `test_state_safety.py` |
| PB-6 | Invoke execution order (per node) | `__call__`: node_start audit → trust gate → input gate → `execute()` → output gate → node_complete audit | order verified for every `src/nodes/` class | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | full `FinancialRegulatoryQAAgent().invoke(_VALID_PAYLOAD, ctx=VERIFIED_EXTERNAL)` | `status=success`; node_history = `[Initialize, PreProcess, RegulatoryQAGraphNode, PostProcess, Finalize]` | `TestBackboneInvokeOrder` |
| PB-6c | Real external caller | `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` — **never** `for_internal()` | inner ANONYMOUS nodes accept the passthrough trust; grounded SUCCESS end-to-end | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json["input"] == _VALID_PAYLOAD` | the deployment evidence invoke exercises the PB-6 payload | `test_invoke_payload_matches_pb6` |
| PB-7 | HITL interrupt propagation | skip stub — `propagate_hitl=False`, no cross-boundary `interrupt()` checkpoint | skipped with reason (real assertion when HITL wired) | `test_pb7_hitl_interrupt_propagation.py` |
| PB-E2E | Entry-point auth | Bearer boundary: missing/wrong/malformed/non-ASCII token; middleware trust preserved; adapter size cap | 401 generic / trust elevation / 413 | `test_server_boot.py` |
| PB-E2E | End-to-end `/invoke` | grounded answer, abstention, validation rejections, non-finite matrix, no caller text, grid scan | see §4 | `test_invoke_e2e.py` |

## 4. Business Logic Tests

| BL-ID | Test | Input | Expected Result | Where |
|-------|------|-------|----------------|-------|
| BL-01 | Happy-path grounded Q&A | `_VALID_PAYLOAD` | `FINANCIAL REGULATORY Q&A RESPONSE` + `[SOURCE-1]` + `Bank Act`; `status=success` | `test_backbone_invoke_succeeds_and_returns_output`, `test_inner_graph_grounds_answer`, `test_grounded_question_produces_cited_answer` |
| BL-02 | Acronym expansion for retrieval | `"... FSA and AML obligations"` | query augmented with `Financial Services Agency` / `Anti-Money Laundering` | `test_expands_known_acronyms` |
| BL-03 | Query normalisation | control chars + tabs + runs of spaces | control chars stripped, whitespace collapsed | `test_normalises_control_chars_and_whitespace` |
| BL-04 | Term-overlap retrieval ranking | grounded query | Bank Act chunk ranked top (score `1.0`); results bounded `[0,1]`, monotonic desc | `TestRetrieveNode` |
| BL-05 | Score-threshold rerank + dedup | mixed-score chunks | drop `< 0.75`, dedup `(source,page)`, rerank desc | `test_filters_below_threshold_and_dedupes` |
| BL-06 | Grounded synthesis + citations | reranked chunk(s) | `[SOURCE-N]` markers + matching `citations`; advisory disclaimer present | `test_grounded_answer_cites_sources` |
| BL-07 | **Abstention** (anti-hallucination) | no chunk clears threshold / off-topic query | insufficient-grounding message + disclaimer; `citations = []`; `status=success` | `test_abstains_when_no_grounding`, `test_inner_graph_abstains_off_topic`, `test_abstention_is_a_success_outcome` |
| BL-08 | Output document assembly | answer + citations | `FINANCIAL REGULATORY Q&A RESPONSE` header + `Citations` block; `result == qa_output` | `TestOutputFormatNode` |
| BL-09 | Graph key coupling | inner `get_output` ↔ outer `merge_output` | 4 coupled keys (`answer`, `citations`, `qa_output`, `status`) mapped; `merge_output` returns changed keys only | `TestOuterGraphComposition`, `TestInnerDomainGraph` |
| BL-10 | Caller-data contract | `input_context` fields | bounds enforced per field; non-finite/out-of-contract values fail CLOSED; rejected values never echoed | `TestInputContextValidation`, `TestPreProcessNodeContract`, `test_non_finite_or_out_of_contract_top_k_fails_closed` |
| BL-11 | Retrieval-depth override | `input_context.top_k` | valid override narrows depth end-to-end (context bridge); invalid fails CLOSED | `TestRetrieveTopKOverride`, `test_caller_top_k_reaches_inner_retrieval_end_to_end` |
| BL-12 | Runtime-config plumbing | `config/config.yaml` | declared values reach the outer graph and inner node constructors; malformed values are dropped, never forwarded | `TestRuntimeConfigPlumbing`, `test_runtime_config_is_live_on_the_deployed_agent` |
| BL-13 | External output schema | monetary tokens, caller-text embeddings | every monetary form snaps to the 1,000 grid as ONE number, decimals included (structural tokens, decimals that were never amounts, and rendered identifiers byte-identical); verbatim caller text `[REDACTED]`; schema note exactly when money renders | `TestPrecisionGate`, `TestRenderedIdentifiersSurvive`, `TestBlockedFieldRedaction`, `TestPostProcessGateLayers`, `TestSchemaNoteRendering`, `test_external_document_carries_only_grid_values` |
| BL-14 | No caller text in the document | question smuggling `[SOURCE-9]` | marker never surfaces in answer or output | `TestAnswerNeverEmbedsCallerText`, `test_fake_citation_marker_in_question_never_reaches_output` |

### Negative / boundary cases

| Case | Node | Expected |
|------|------|----------|
| empty / whitespace `user_input` | PreProcessNode | `status=error`, "empty" |
| query `> 2000` chars | PreProcessNode | `status=error`, "exceeds 2000" |
| non-mapping `input_context` | PreProcessNode | `status=error`, field named |
| invalid `channel` (case/charset/length) | PreProcessNode | `status=error`, value never echoed |
| `top_k` non-int / bool / non-finite / out of 1..10 | PreProcessNode, RetrieveNode | `status=error`, value never echoed |
| empty query after normalisation | InputValidateNode | `status=error`, "empty" |
| empty query for retrieval | RetrieveNode | `status=error`, "query" |
| all chunks below threshold | RerankFilterNode | `reranked = []`, `status=success` |
| missing `retrieved_chunks` | RerankFilterNode | `status=error`, "retrieved_chunks" |
| non-finite configured threshold / chunk score | RerankFilterNode | falls back to default / chunk dropped — the gate never weakens |
| no grounded chunks | GenerateAnswerNode | insufficient-grounding answer, `citations=[]`, `status=success` |
| missing `answer` | OutputFormatNode | `status=error`, "answer" |
| empty `qa_output` | PostProcessNode | fallback message, `status=success` |
| credential leak in output | PostProcessNode | redacted, `status=error` |
| verbatim caller text in output | PostProcessNode | `[REDACTED]`, `status=success` |
| off-grid monetary token in output | PostProcessNode | snapped to the 1,000 grid + audit event |
| oversized `input_context` (> 256 KB) | `/invoke` adapter | HTTP 413, agent never reached |
| missing / wrong / non-ASCII Bearer | `/invoke` adapter | HTTP 401, generic body |

## 5. Test Execution Summary

- Execution: `pytest tests/` (unit + proof-of-boundary; mirrors CI).
- PB-7 ships as a skip stub by design (no cross-boundary HITL); all other tests
  execute on both the grounding and abstention paths.
- Repository checks (dependency pinning, stub-test detection, category
  consistency, import isolation, credential scan, trust level, manifest
  schema) run in CI via `scripts/check_*.py` — all PASS.
- Coverage: node + graph modules exercised on success, error, and abstention
  paths, plus the caller-data and output-schema contracts.
