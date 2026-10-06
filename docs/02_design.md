# Template Design Specification — FIN-C2-001 Financial Regulatory Q&A Agent

## Position in AgentCore Architecture

- **Agent Class**: FinancialRegulatoryQAAgent
- **L1 Base (framework base class)**: AgentBaseGraph — direct framework inheritance
- **Category**: Cat 2 (multi-step domain workflow — RAG pattern)
- **Three-Layer Separation**:
  - State: flat TypedDict composition (no Pydantic — msgpack incompatible)
  - Node: framework inheritance (Template Method: `execute(self, state) -> dict` override only)
  - Graph: composition (`register_nodes()` for node substitution; inner BaseGraph)

## Architecture Overview

Cat 2 two-layer nested architecture. The outer `AgentBaseGraph` keeps the fixed
5-node backbone; the `main` slot is a `GraphNode` (RegulatoryQAGraphNode) that
delegates the RAG pipeline to an inner `BaseGraph` (DomainWorkflowGraph).

### Node Configuration — Outer backbone (AgentBaseGraph)

| Node | Responsibility | Input State | Output State | Inherits/Overrides |
|------|---------------|-------------|--------------|-------------------|
| initialize | schema_version, session_id, trust_level | — | session_id | InitializeNode (default) |
| pre_process | **Trust gate (VERIFIED_EXTERNAL)** + query validation + input_context contract | user_input, input_context | validated_input, enriched_context | PreProcessNode |
| main | delegate to inner DomainWorkflowGraph; bridge input_context | validated_input, input_context | answer, citations, qa_output | RegulatoryQAGraphNode (GraphNode) |
| post_process | **Output gate** (credential scan + caller-text redaction + precision grid) + expose result | qa_output | formatted_output, result | PostProcessNode |
| finalize | response_metadata, total_time_ms | — | response_metadata | FinalizeNode (default) |

### Node Configuration — Inner RAG pipeline (DomainWorkflowGraph / BaseGraph)

| Node | Responsibility | Input State | Output State | Trust |
|------|---------------|-------------|--------------|-------|
| input_validate | domain query normalisation + acronym expansion (FSA/FISC/AML/KYC) | validated_input | normalized_query | ANONYMOUS |
| retrieve | top-k regulatory chunk retrieval (config `retrieval.top_k`; caller may narrow via `input_context.top_k`) | normalized_query, input_context | retrieved_chunks | ANONYMOUS |
| rerank_filter | rerank + drop chunks below `retrieval.score_threshold`; dedup | retrieved_chunks | reranked_chunks | ANONYMOUS |
| generate_answer | grounded synthesis with `[SOURCE-N]` citations + disclaimer (deterministic — see note below) | reranked_chunks | answer, citations | ANONYMOUS |
| output_format | assemble final Q&A document (answer + citations block; schema note when money renders) | answer, citations | qa_output, result | ANONYMOUS |

### Implementation Note — LLM synthesis

The current build uses **deterministic grounded synthesis**: `generate_answer`
assembles the answer directly from the reranked regulatory passages with `[SOURCE-N]`
citation markers and the mandatory advisory disclaimer, and abstains with an explicit
"insufficient grounding" answer when no chunk clears the score gate. This keeps the
grounding / anti-hallucination contract fully verifiable offline. This build reads
**no** LLM configuration — the framework ships no LLM client for it to call.

**Live-LLM synthesis is a documented next step, not a gap in this build.** When the
platform SDK LLM ships, `generate_answer` wires it in by loading and rendering
`prompts/fin_qa.j2` (declared in `config/config.yaml` as `llm.system_prompt_template`)
and invoking the SDK LLM under the same grounding rules. `config/config.yaml` already
carries the retrieval (`top_k`, `score_threshold`) and LLM (`temperature`,
`max_tokens`, `system_prompt_template`) settings, and
`RegulatoryQAGraphNode._parent_config()` validates and forwards them to the inner
graph, which injects them into the config-driven node constructors (`RetrieveNode`,
`RerankFilterNode`) in `register_nodes()`. Constructor injection is the config route
because the node contract is `execute(self, state) -> dict` — a node takes no
per-invocation config parameter. No configuration-shape change is needed when the
LLM path is enabled.

### Runtime configuration

`config/agent.yaml` is the **flat registration manifest** (root-level keys only —
identity, entry point, trust level, compile-time requires). Every runtime parameter
lives in `config/config.yaml`, which the platform registry loads and passes as
`Graph(config=...)`; the standalone server (`src/api/server.py`) reads the same file
via `_runtime_config()` so both deployments see identical configuration. The outer
`AgentBaseGraph` consumes `max_retry` from that config (retry routing); the
`retrieval`/`llm` blocks are validated (type, finiteness, range) in
`_parent_config()` before being forwarded — a malformed file can neither crash graph
construction nor weaken the retrieval confidence gate.

### Caller-data contract (`input_context`)

`POST /invoke` accepts an optional `input_context` object (adapter-capped at 256 KB
serialized). Every declared field is validated in PreProcessNode before the domain
workflow runs; violations fail CLOSED with an error naming the field — never echoing
the value. Undeclared keys are ignored.

| Field | Type / bounds | Effect |
|-------|---------------|--------|
| `channel` | str, `^[a-z0-9_]{1,32}$` | recorded in `enriched_context` audit metadata (absent → `"unknown"`) |
| `top_k` | int, 1..10 | narrows retrieval depth for this invocation (absent → configured `retrieval.top_k`) |

The confidence threshold (`retrieval.score_threshold`) is deliberately **not**
caller-controllable — a caller can never lower the grounding bar.

**Context bridge.** The framework's `GraphNode.execute()` does not forward the outer
state's `input_context` into `subgraph.invoke()` (SDK 1.0.1), so the outer graph
stashes it in a ContextVar (`RegulatoryQAGraphNode.extract_input`) and the inner
graph re-seeds it (`DomainWorkflowGraph._extra_initial_state`) — see
`src/graph/context_bridge.py`. Verified end-to-end: a caller `top_k` override
changes the number of cited sources through the full nested graph.

### External output schema

The response document is built from retrieved regulatory content only — **no
caller-supplied text is embedded** (a question smuggling fake `[SOURCE-N]` markers
cannot surface them as pseudo-citations). The output gate (PostProcessNode) enforces
three independent layers:

1. **Credential scan** — API-key/JWT/Bearer/assignment patterns anywhere in the
   document withhold the answer entirely (sanitised stub, `status=error`).
2. **Verbatim caller-text redaction** — a verbatim embedding of the caller's raw or
   normalised question (or request metadata) is replaced with `[REDACTED]`.
3. **Monetary precision grid** — monetary figures are expressed in units of 1,000;
   every monetary-form token (comma-grouped, 5+-digit runs, or short values in
   currency context — code or symbol, either side, horizontal whitespace or a
   single newline, signed) is snapped onto the grid, with an audit event. The
   renderer prints the schema note whenever money renders, using the same grammar
   as the gate.

   A decimal amount is absorbed into **one** token, so an off-grid amount in
   currency context snaps as a whole number (`JPY 1234.56` → `JPY 1,000`) instead
   of snapping its integer part and leaving the fraction dangling, and a ratio,
   percentage or section number that was never monetary is left byte-identical
   (`0.123456`, `9999.99999%`, `Section 3.14159`). The absorption is written
   `(?:\.\d+|(?!\.\d))` — take the fraction whole, or assert there is none —
   because an optional `(?:\.\d+)?` lets the engine backtrack out of the fraction
   and re-match the integer alone (`JPY 1234.56m`).

   The gate reads a monetary token by form, and several of the identifiers this
   document renders have that same form. Guards on both ends stop a number being
   **entered** from inside one: the alphabet is `[A-Za-z0-9_/-]` — the citation
   markers the renderer emits itself (`[SOURCE-3]`), the store-supplied `source`
   and `page` fields it interpolates verbatim (`(p.N/A)`, `(p.12/45)`, snake_case
   document keys from the production vector store), and the instrument and
   regulation numbers in the retrieved text (`FSA-2026-0114`, `ISO-20022`, the
   ISIN `JP1234567890`, the LEI `5493001KJTIIGC8Y1R12`). `.` is in the **leading**
   guard only; in the trailing guard an amount ending a sentence would escape.

   `<three uppercase letters><digits>` stays genuinely ambiguous — `JPY-9999` is a
   signed amount and `SKF-6205` is a part number — so where the marker is
   **attached** to the value the gate consults ISO 4217: a currency code snaps, any
   other three-letter word is a name and is left alone. A **separated** marker
   (`SKF 6205`) stays unrestricted, because there a false snap fails safe while a
   missed leak does not; the same reasoning keeps a single newline as a delimiter
   (`JPY\n9999`), at the documented cost that a line ending in a three-letter code
   binds to the number opening the next line.

### Data Flow

```
Outer:  START → initialize → pre_process → main → {route} → post_process → finalize → END
                                            ↓ (retry, max 3)
                                         pre_process

Inner (main slot → RegulatoryQAGraphNode → DomainWorkflowGraph):
        START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

### State Definition

| Field | Type | Purpose | Required |
|-------|------|---------|----------|
| validated_input | Optional[str] | normalised query | producer: pre_process |
| enriched_context | Optional[str] | JSON channel metadata (stored as str) | producer: pre_process |
| normalized_query | Optional[str] | domain-normalised query | producer: input_validate |
| retrieved_chunks | Optional[str] | JSON list of scored chunks | producer: retrieve |
| reranked_chunks | Optional[str] | JSON list of filtered chunks | producer: rerank_filter |
| answer | Optional[str] | grounded answer w/ `[SOURCE-N]` | producer: generate_answer |
| citations | Optional[str] | JSON list of cited sources | producer: generate_answer |
| qa_output | Optional[str] | assembled Q&A document | producer: output_format |
| result | Optional[str] | caller result (post-gate) | producer: post_process |

**State Constraints (mandatory):**
- Flat TypedDict only (primitives + JSON-serializable types); dict/list fields stored as JSON `Optional[str]`
- `formatted_output` is inherited from AgentState — NOT re-declared
- No JWT, API keys, credentials in State (checkpoint DB leakage)
- InvocationContext via `config["configurable"]` only (not in State)
- No Pydantic models, dataclass, arbitrary Python objects (msgpack incompatible)

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (correlation_id, session_id, trust_level)
- [x] Trust gate: `required_trust_level = VERIFIED_EXTERNAL` on PreProcessNode (outer gate); inner nodes ANONYMOUS
- [x] Output gate: `_security_gate_output` — credential scan on the agent class (canonical) + the three-layer gate in PostProcessNode (runtime)
- [x] Audit: `emit_trace_event(event, payload, state)` — one domain event inside every `execute()`

> **Output-gate placement (this template):** the canonical `_security_gate_output`
> is a method on the agent class `FinancialRegulatoryQAAgent` (graph.py), delegating
> to the module-level scanner in `post_process_node.py` that the post_process slot
> applies at runtime — one source of truth for the pattern set. The inner
> FunctionNodes do NOT override `_security_gate_output` (framework `@final`).

## Import Isolation Confirmation
- [x] Template does not import the agenticstar platform SDK
- [x] Import targets: framework/ and shared/ only

## Composition Pattern

- **Pattern**: GraphNode (subgraph) — outer AgentBaseGraph + inner BaseGraph
- **Composition target**: `src/graph/domain_workflow_graph.py::DomainWorkflowGraph`
- **Error propagation strategy**: propagate (RegulatoryQAGraphNode.error_strategy = "propagate")

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG workflow, no autonomous loop |
| Composition pattern | Standalone | GraphNode subgraph | **GraphNode subgraph** | Cat 2 nested: domain complexity isolated in inner BaseGraph |
| Output-gate location | post_process node only | agent class + node | **agent class + node** | Canonical gate on agent class, applied at post_process |
| Query echo in answer | Echo the question | Never embed caller text | **Never embed caller text** | The document's value is that every statement traces to a retrieved source; caller text could smuggle pseudo-citations |
| Caller retrieval tuning | top_k + threshold | top_k only | **top_k only** | The confidence threshold is the anti-hallucination bar — caller must not lower it |
