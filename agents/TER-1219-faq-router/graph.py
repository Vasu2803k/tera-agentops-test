"""LangGraph StateGraph for the FAQ router — TER-1219.

Graph topology
--------------
    [START]
       │
  ┌────▼─────┐
  │ classify │  LLM node: returns topic + confidence
  └────┬─────┘
       │  conditional_edge (route_question)
       ├──► billing
       ├──► account
       ├──► technical
       └──► fallback
       │
    [END]  (all topic/fallback nodes go straight to END)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class FAQState(TypedDict):
    """Shared state passed between all nodes."""

    question: str
    topic: str        # classifier output: billing | account | technical | fallback
    confidence: float  # 0.0 – 1.0
    answer: str


# ---------------------------------------------------------------------------
# In-code FAQ maps (no external services)
# ---------------------------------------------------------------------------

FAQ_MAP: dict[str, list[dict[str, str]]] = {
    "billing": [
        {
            "q": "How do I update my payment method?",
            "a": "Go to Settings → Billing → Payment Methods and click 'Update'.",
        },
        {
            "q": "Why was I charged twice?",
            "a": "Duplicate charges are usually reversed within 3–5 business days. "
                 "Contact billing@example.com with your transaction ID for faster resolution.",
        },
        {
            "q": "How do I download my invoice?",
            "a": "Navigate to Settings → Billing → Invoice History and click 'Download PDF'.",
        },
    ],
    "account": [
        {
            "q": "How do I reset my password?",
            "a": "Click 'Forgot Password' on the login page. A reset link will be emailed to you.",
        },
        {
            "q": "How do I update my email address?",
            "a": "Go to Settings → Profile → Contact Info and enter your new email address.",
        },
        {
            "q": "How do I delete my account?",
            "a": "Account deletion is permanent. Submit a request via Settings → Account → Delete Account.",
        },
    ],
    "technical": [
        {
            "q": "Why is the agent returning a 500 error?",
            "a": "A 500 error usually means the agent container crashed. Check the deployment "
                 "logs in AgentOps → Deployments → Logs.",
        },
        {
            "q": "How do I increase the agent memory limit?",
            "a": "Edit the deployment's resource config: set 'memory' to your desired value (e.g. '4Gi') "
                 "under engine_type_config.resources.",
        },
        {
            "q": "The agent is timing out. What should I do?",
            "a": "Increase the TIMEOUT_SECONDS environment variable and redeploy. "
                 "Also check if downstream services are slow.",
        },
    ],
}

_CANNED_ANSWERS: dict[str, str] = {
    topic: "\n".join(f"• {entry['q']}\n  → {entry['a']}" for entry in entries)
    for topic, entries in FAQ_MAP.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_llm() -> ChatOpenAI:
    """Build the LLM client from environment variables.

    In production these are injected by AgentOps from config.yaml.
    In tests they can be overridden with pytest monkeypatch.
    """
    return ChatOpenAI(
        model=os.environ.get("LITELLM_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("LITELLM_API_KEY", "sk-placeholder"),
        base_url=os.environ.get("LITELLM_BASE_URL"),  # None → default OpenAI
        temperature=0.0,
    )


_CLASSIFY_SYSTEM = """\
You are a question classifier for a customer-support FAQ router.
Classify the user question into exactly one of these topics:
  billing   - payment, invoices, charges, subscriptions
  account   - login, password, profile, account settings
  technical - errors, performance, configuration, deployments
  fallback  - anything that does not clearly fit the above

Respond with a JSON object and nothing else:
{"topic": "<billing|account|technical|fallback>", "confidence": <0.0-1.0>}
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def classify_node(state: FAQState) -> FAQState:
    """Call the LLM to classify the question and return topic + confidence."""
    llm = _build_llm()
    messages = [
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=state["question"]),
    ]
    try:
        raw = llm.invoke(messages).content
        parsed: dict[str, Any] = json.loads(raw)
        topic = str(parsed.get("topic", "fallback")).lower()
        if topic not in FAQ_MAP and topic != "fallback":
            topic = "fallback"
        confidence = float(parsed.get("confidence", 0.5))
    except Exception as exc:  # noqa: BLE001
        log.warning("Classifier error (%s); routing to fallback.", exc)
        topic = "fallback"
        confidence = 0.0

    return {**state, "topic": topic, "confidence": confidence}


def billing_node(state: FAQState) -> FAQState:
    """Return billing FAQ answers."""
    answer = (
        "Here are answers to common billing questions:\n\n"
        + _CANNED_ANSWERS["billing"]
    )
    return {**state, "answer": answer}


def account_node(state: FAQState) -> FAQState:
    """Return account FAQ answers."""
    answer = (
        "Here are answers to common account questions:\n\n"
        + _CANNED_ANSWERS["account"]
    )
    return {**state, "answer": answer}


def technical_node(state: FAQState) -> FAQState:
    """Return technical FAQ answers."""
    answer = (
        "Here are answers to common technical questions:\n\n"
        + _CANNED_ANSWERS["technical"]
    )
    return {**state, "answer": answer}


def fallback_node(state: FAQState) -> FAQState:
    """Return a polite fallback that includes the detected topic + confidence."""
    answer = (
        f"I can help with billing, account, or technical questions. "
        f"Your question was classified as '{state['topic']}' "
        f"(confidence: {state['confidence']:.0%}). "
        "Please rephrase or choose one of the supported topics."
    )
    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_question(state: FAQState) -> str:
    """Conditional-edge function: return the next node name."""
    topic = state.get("topic", "fallback")
    return topic if topic in FAQ_MAP else "fallback"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Construct and compile the FAQ router StateGraph."""
    graph: StateGraph = StateGraph(FAQState)

    # Nodes
    graph.add_node("classify", classify_node)
    graph.add_node("billing", billing_node)
    graph.add_node("account", account_node)
    graph.add_node("technical", technical_node)
    graph.add_node("fallback", fallback_node)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional edge from classifier to topic / fallback nodes
    graph.add_conditional_edges(
        "classify",
        route_question,
        {
            "billing": "billing",
            "account": "account",
            "technical": "technical",
            "fallback": "fallback",
        },
    )

    # All leaf nodes go straight to END
    graph.add_edge("billing", END)
    graph.add_edge("account", END)
    graph.add_edge("technical", END)
    graph.add_edge("fallback", END)

    return graph.compile()
