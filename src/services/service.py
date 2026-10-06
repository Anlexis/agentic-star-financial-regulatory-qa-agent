"""AgentCore Platform v1.0"""

# Service layer: domain queries, external API wrappers, data aggregation.
# Must NOT contain business logic, routing, or credentials.
# Nodes call this; this calls shared/services/ for external integrations.
#
# Contract: FIN-C2-001 runs a fully offline RAG pipeline — RetrieveNode scores
# a bundled sample regulatory corpus, so no external domain service is called.
# `Service` is the seam where a real deployment wires the regulatory vector
# store / FSA-FISC API client.  `fetch()` therefore has a concrete no-op
# contract: it returns an empty result set rather than raising, so a caller
# that does reach it degrades gracefully instead of erroring.

from __future__ import annotations

from typing import Any


class Service:
    """Domain data-access seam for FIN-C2-001 (offline build; no external service).

    The bundled build uses the corpus in RetrieveNode; a real deployment wires
    a regulatory vector store / FSA-FISC API client here.
    """

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return domain data for the query.

        No-op contract: no external regulatory service is configured, so this
        returns an empty result set (`{"results": []}`).  A real deployment
        replaces the body with the vector-store / API call.
        """
        return {"results": []}
