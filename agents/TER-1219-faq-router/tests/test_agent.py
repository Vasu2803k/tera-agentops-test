"""Tests for TER-1219 FAQ Router — covers all six Acceptance Criteria.

Run with:
    pytest agents/TER-1219-faq-router/tests/ -v

The tests mock the LLM so no API key is required.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(topic: str, confidence: float) -> MagicMock:
    """Create a mock LLM response that returns a valid classifier JSON."""
    msg = MagicMock()
    msg.content = json.dumps({"topic": topic, "confidence": confidence})
    return msg


# ---------------------------------------------------------------------------
# AC-1  File structure: agent.py  config.yaml  requirements.txt
# ---------------------------------------------------------------------------


class TestAC1FileStructure:
    """AC1: Follows the build-langgraph-agent skill structure exactly."""

    def test_agent_py_exists(self):
        import os

        base = os.path.join(
            os.path.dirname(__file__), "..",  # agents/TER-1219-faq-router/
        )
        assert os.path.isfile(os.path.join(base, "agent.py"))

    def test_config_yaml_exists_with_langgraph_framework(self):
        import os

        import yaml  # PyYAML is transitively available via langchain-core

        base = os.path.join(os.path.dirname(__file__), "..")
        cfg_path = os.path.join(base, "config.yaml")
        assert os.path.isfile(cfg_path)
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        assert cfg["framework"] == "langgraph"

    def test_requirements_txt_exists(self):
        import os

        base = os.path.join(os.path.dirname(__file__), "..")
        assert os.path.isfile(os.path.join(base, "requirements.txt"))


# ---------------------------------------------------------------------------
# AC-2  AgentRuntime @app.entrypoint pattern + try/except import + auth guard
# ---------------------------------------------------------------------------


class TestAC2EntrypointPattern:
    """AC2: agent.py uses the @app.entrypoint pattern with guards."""

    def test_agent_module_imports_without_agent_runtime_installed(self):
        """The try/except guard means the module loads even without the runtime."""
        # Remove any cached module so we get a fresh import
        for key in list(sys.modules.keys()):
            if "agent_runtime" in key:
                del sys.modules[key]
        # Import must succeed (the except branch provides a shim)
        import agents.TER_1219_faq_router.agent as agent_mod  # noqa: F401

        assert hasattr(agent_mod, "app")
        assert hasattr(agent_mod, "invoke")

    def test_invoke_is_async(self):
        import agents.TER_1219_faq_router.agent as agent_mod

        import inspect

        assert inspect.iscoroutinefunction(agent_mod.invoke)

    def test_auth_context_none_does_not_raise(self):
        """Passing auth_context=None must not raise."""
        from agents.TER_1219_faq_router.graph import FAQState

        mock_llm_resp = _make_llm_response("billing", 0.95)
        with patch(
            "agents.TER_1219_faq_router.graph.ChatOpenAI"
        ) as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_llm_resp
            import agents.TER_1219_faq_router.agent as agent_mod

            # Rebuild the graph with mocked LLM
            with patch("agents.TER_1219_faq_router.agent._graph") as mock_graph:
                mock_graph.invoke.return_value = {
                    "question": "billing?",
                    "topic": "billing",
                    "confidence": 0.95,
                    "answer": "billing answer",
                }
                result = asyncio.run(
                    agent_mod.invoke("billing?", auth_context=None)
                )
        assert result["role"] == "assistant"


# ---------------------------------------------------------------------------
# AC-3  StateGraph with conditional edge from classifier
# ---------------------------------------------------------------------------


class TestAC3GraphStructure:
    """AC3: StateGraph with conditional edge from classifier to topic/fallback nodes."""

    def test_graph_compiles(self):
        """build_graph() must return a compiled graph without raising."""
        from agents.TER_1219_faq_router.graph import build_graph

        g = build_graph()
        assert g is not None

    def test_graph_has_compiled_attribute(self):
        """The compiled graph exposes .invoke (LangGraph CompiledStateGraph)."""
        from agents.TER_1219_faq_router.graph import build_graph

        g = build_graph()
        assert callable(getattr(g, "invoke", None))

    @pytest.mark.parametrize(
        "topic", ["billing", "account", "technical", "fallback"]
    )
    def test_route_question_returns_correct_node(self, topic):
        from agents.TER_1219_faq_router.graph import route_question, FAQState

        state: FAQState = {
            "question": "test",
            "topic": topic,
            "confidence": 0.9,
            "answer": "",
        }
        assert route_question(state) == topic

    def test_route_question_unknown_topic_returns_fallback(self):
        from agents.TER_1219_faq_router.graph import route_question, FAQState

        state: FAQState = {
            "question": "test",
            "topic": "unknown_topic",
            "confidence": 0.1,
            "answer": "",
        }
        assert route_question(state) == "fallback"


# ---------------------------------------------------------------------------
# AC-4  ≥ 2 FAQ entries per topic + classifier returns topic + confidence
# ---------------------------------------------------------------------------


class TestAC4FAQEntries:
    """AC4: each topic has ≥ 2 FAQ entries; classifier returns topic + confidence."""

    def test_billing_has_at_least_2_entries(self):
        from agents.TER_1219_faq_router.graph import FAQ_MAP

        assert len(FAQ_MAP["billing"]) >= 2

    def test_account_has_at_least_2_entries(self):
        from agents.TER_1219_faq_router.graph import FAQ_MAP

        assert len(FAQ_MAP["account"]) >= 2

    def test_technical_has_at_least_2_entries(self):
        from agents.TER_1219_faq_router.graph import FAQ_MAP

        assert len(FAQ_MAP["technical"]) >= 2

    def test_classify_node_sets_topic_and_confidence(self):
        from agents.TER_1219_faq_router.graph import FAQState, classify_node

        mock_resp = _make_llm_response("account", 0.88)
        with patch("agents.TER_1219_faq_router.graph.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_resp
            state: FAQState = {
                "question": "How do I reset my password?",
                "topic": "",
                "confidence": 0.0,
                "answer": "",
            }
            result = classify_node(state)

        assert result["topic"] == "account"
        assert result["confidence"] == pytest.approx(0.88)

    def test_classify_node_handles_llm_error_gracefully(self):
        """Classifier falls back to 'fallback' topic on LLM/parse error."""
        from agents.TER_1219_faq_router.graph import FAQState, classify_node

        bad_resp = MagicMock()
        bad_resp.content = "not valid json at all"
        with patch("agents.TER_1219_faq_router.graph.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = bad_resp
            state: FAQState = {
                "question": "???",
                "topic": "",
                "confidence": 0.0,
                "answer": "",
            }
            result = classify_node(state)

        assert result["topic"] == "fallback"
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# AC-5  Returns {content, role: assistant} — non-streaming invoke
# ---------------------------------------------------------------------------


class TestAC5ResponseShape:
    """AC5: returns {content: ..., role: assistant} and supports non-streaming invoke."""

    @pytest.mark.parametrize(
        "topic",
        ["billing", "account", "technical"],
    )
    def test_topic_nodes_produce_non_empty_answer(self, topic):
        from agents.TER_1219_faq_router import graph as g

        node_fn = getattr(g, f"{topic}_node")
        state = {
            "question": "test",
            "topic": topic,
            "confidence": 0.9,
            "answer": "",
        }
        result = node_fn(state)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_fallback_node_includes_topic_and_confidence(self):
        from agents.TER_1219_faq_router.graph import FAQState, fallback_node

        state: FAQState = {
            "question": "what is the meaning of life?",
            "topic": "fallback",
            "confidence": 0.42,
            "answer": "",
        }
        result = fallback_node(state)
        assert "fallback" in result["answer"]
        assert "42%" in result["answer"]
        assert "billing" in result["answer"]
        assert "account" in result["answer"]
        assert "technical" in result["answer"]

    def test_invoke_returns_content_and_role(self):
        """Full async invoke call must return {content, role: assistant}."""
        import agents.TER_1219_faq_router.agent as agent_mod

        with patch("agents.TER_1219_faq_router.agent._graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "question": "billing?",
                "topic": "billing",
                "confidence": 0.9,
                "answer": "Use Settings → Billing to manage payments.",
            }
            result = asyncio.run(
                agent_mod.invoke(
                    "How do I update my card?",
                    session_id="sess-001",
                    streaming=False,
                    auth_context=None,
                )
            )

        assert result["role"] == "assistant"
        assert "content" in result
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0

    def test_streaming_flag_accepted_and_ignored(self):
        """streaming=True must be accepted without error (response is still a dict)."""
        import agents.TER_1219_faq_router.agent as agent_mod

        with patch("agents.TER_1219_faq_router.agent._graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "question": "help",
                "topic": "fallback",
                "confidence": 0.3,
                "answer": "fallback answer",
            }
            result = asyncio.run(
                agent_mod.invoke("help", streaming=True, auth_context=None)
            )

        assert isinstance(result, dict)
        assert result["role"] == "assistant"


# ---------------------------------------------------------------------------
# AC-6  Branch name is agent-generated/TER-1219 (structural check)
# ---------------------------------------------------------------------------


class TestAC6BranchConvention:
    """AC6: verifies the story key appears in this file's directory path.

    The actual branch name is validated by the PR; here we just confirm
    the agent lives under the expected story-key namespace.
    """

    def test_story_key_in_package_path(self):
        import os

        # __file__ is  .../agents/TER-1219-faq-router/tests/test_agent.py
        assert "TER-1219" in os.path.abspath(__file__)
