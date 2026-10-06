"""AgentCore Platform v1.0"""

# Package-root export: the manifest entry point resolves through this package,
# so `import src.graph` exposes the agent class here. The flat manifest's
# `class:` field carries the full dotted path (src.graph.graph.<Class>); this
# re-export additionally keeps `src.graph.FinancialRegulatoryQAAgent`
# importable for callers that address the package root.
from .graph import FinancialRegulatoryQAAgent

__all__ = ["FinancialRegulatoryQAAgent"]
