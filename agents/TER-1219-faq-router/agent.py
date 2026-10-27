"""FAQ Router LangGraph Agent — TER-1219.

Classifies incoming questions into billing / account / technical / fallback
and returns a canned answer. All data is in-code; no external services.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

# AgentRuntime import with try/except guard so the module also works
# in plain Python unit-test environments that lack the runtime package.
try:
    from agent_runtime.runtime import AgentRuntimeApp, AuthContext
except ImportError:  # pragma: no cover — missing only outside AgentOps
    from dataclasses import dataclass

    @dataclass
    class AuthContext:  # type: ignore[no-redef]
        user_id: str = ""
        email: str = ""
        token: str = ""

    class AgentRuntimeApp:  # type: ignore[no-redef]
        """Minimal shim used in test/local environments."""

        def entrypoint(self, fn):  # noqa: D401
            return fn


from agents.TER_1219_faq_router.graph import build_graph

log = logging.getLogger(__name__)

app = AgentRuntimeApp()
_graph = build_graph()


@app.entrypoint
async def invoke(
    message: str,
    session_id: str | None = None,
    streaming: bool = False,  # accepted but ignored — streaming not required
    auth_context: "AuthContext | None" = None,
) -> dict[str, Any]:
    """Agent entrypoint called by AgentOps on every /invoke request.

    Args:
        message:      User question string.
        session_id:   Optional session / conversation ID.
        streaming:    Ignored (streaming not in scope for this agent).
        auth_context: JWT context injected by the runtime; may be None.

    Returns:
        ``{"content": str, "role": "assistant"}``
    """
    if auth_context is not None:
        log.info(
            "Authenticated request from user=%s", getattr(auth_context, "user_id", "?")
        )

    result = await asyncio.to_thread(
        _graph.invoke,
        {"question": message, "topic": "", "confidence": 0.0, "answer": ""},
    )
    return {"content": result["answer"], "role": "assistant"}
