# The identity the entrypoint provisions secrets under, against the identity the
# manifest declares.
#
# The manifest is the source of truth for who this agent is. The secret provider
# is scoped by that identity: it reads `env/namespaces/{namespace}/.env.{env}`
# and `env/agents/{namespace}/{agent_name}/.env.{env}`
# (shared/secrets/dotenv_provider.py). Under the registry the scope comes from
# `config/agent.yaml`; standalone it comes from whatever `src/api/server.py`
# passes to the factory. When those two disagree, one agent's secrets live in
# two stores and a key provisioned for the registry deployment is simply absent
# in the standalone one — with no error at boot, because a missing tier file is
# ignored by design. Nothing fails until a secret is declared, and then it fails
# in only one of the two deployments.
#
# `namespace` scopes BOTH tiers, so a drifting namespace moves the agent tier
# too — matching `agent_name` alone does not keep the agent-scoped path aligned.
#
# Both sides are READ here, never restated: the manifest is parsed from
# config/agent.yaml, and the entrypoint's identity is taken from the provider
# the module actually provisioned on the agent. A test that spelled the expected
# namespace out twice would keep passing through exactly the drift it exists to
# catch.
#
# The direction of the alignment is pinned too. `namespace:` is
# `lower(industry_code)` — the manifest's own `industry` field, lowercased. Both
# facts are checked, so re-aligning the wrong way (moving the manifest onto the
# entrypoint's value instead of the reverse) fails here rather than passing as a
# fix.
#
# Deterministic — no model, no network, no filesystem beyond the manifest.

from pathlib import Path

from framework.utils.config_loader import load_config

from src.api.server import agent

# tests/integration/<this file> -> parents[2] is the repository root.
_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "agent.yaml"
_MANIFEST = load_config(str(_MANIFEST_PATH))

# The provider the entrypoint bound at import time — the identity that is live
# in a standalone deployment, not a re-derivation of it. `_namespace` and
# `_agent_name` are declared on the SecretProvider base class
# (framework/secrets/base.py), which is also where the framework itself reads
# them from when it reports a missing secret.
_PROVISIONED = agent._secrets_provider


def test_manifest_declares_the_identity_fields() -> None:
    """The two fields the rest of this module compares against must be present.

    Without this, a manifest that lost `namespace:` would make every comparison
    below `None == None` and the file would pass while asserting nothing.
    """
    assert _MANIFEST.get("namespace"), f"{_MANIFEST_PATH} declares no namespace"
    assert _MANIFEST.get("name"), f"{_MANIFEST_PATH} declares no name"
    assert _MANIFEST.get("industry"), f"{_MANIFEST_PATH} declares no industry"


def test_provisioned_namespace_matches_the_manifest() -> None:
    assert _PROVISIONED._namespace == _MANIFEST["namespace"], (
        "src/api/server.py provisions secrets under namespace "
        f"{_PROVISIONED._namespace!r}, but config/agent.yaml declares "
        f"{_MANIFEST['namespace']!r}. A secret would resolve from a different "
        "store standalone than under the registry."
    )


def test_provisioned_agent_name_matches_the_manifest() -> None:
    assert _PROVISIONED._agent_name == _MANIFEST["name"], (
        "src/api/server.py provisions secrets for agent name "
        f"{_PROVISIONED._agent_name!r}, but config/agent.yaml declares "
        f"{_MANIFEST['name']!r}."
    )


def test_manifest_namespace_is_lower_industry() -> None:
    """`namespace:` is `lower(industry_code)` — the fleet-wide convention.

    Checked against the manifest's own `industry` field, so this pins which of
    the two values is the correct one to align on without hard-coding either.
    """
    assert _MANIFEST["namespace"] == _MANIFEST["industry"].lower(), (
        f"config/agent.yaml declares namespace {_MANIFEST['namespace']!r}; the "
        f"convention is lower(industry) = {_MANIFEST['industry'].lower()!r}."
    )
