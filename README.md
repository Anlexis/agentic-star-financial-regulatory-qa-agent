# Financial Regulatory Q&A Agent

AI agent for answering financial regulatory questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Finance
> **Template ID**: FIN-C2-001

## Overview

Answers compliance questions about Japanese financial regulation — FSA guidance, FISC security
guidelines, AML/KYC obligations, the Financial Instruments and Exchange Act (金融商品取引法) and
the Bank Act (銀行法) — with a retrieval-augmented pipeline. Each query is validated and
normalised, matching regulatory passages are retrieved and filtered by a confidence threshold,
and the answer is synthesised **only** from passages that clear that threshold, with `[SOURCE-N]`
citations and an advisory disclaimer. When no passage is confident enough, the agent explicitly
declines to answer instead of fabricating regulatory text — abstention is a first-class outcome.

The bundled retrieval corpus is a small sample of public regulatory material so the pipeline runs
and tests end-to-end out of the box; a real deployment replaces it with its own regulatory
knowledge base behind the same node contract.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent
fails at graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
prompts/      system prompt template for a live-LLM build
docs/         design and operational documentation
```

See `docs/` for the design specification and test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
