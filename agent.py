# agent.py — generated from Jira story TER-1219
# FAQ Router LangGraph agent for AgentRuntime
from __future__ import annotations

import json
import logging
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

try:
    from .agent_runtime.runtime import AgentRuntimeApp
except ImportError:
    from agent_runtime.runtime import AgentRuntimeApp

logger = logging.getLogger(__name__)
app = AgentRuntimeApp()

# ── LLM setup ────────────────────────────────────────────────────────────────
LITELLM_BASE = os.getenv("LITELLM_BASE_URL")
LITELLM_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")
LITELLM_MODEL = os.getenv("LITELLM_MODEL")

model_name = LITELLM_MODEL or os.getenv("LLM_MODEL", "gpt-4o-mini")
llm_kwargs: dict = {"model": model_name}
if LITELLM_BASE:
    llm_kwargs.update({"base_url": LITELLM_BASE, "api_key": LITELLM_KEY})
llm = ChatOpenAI(**llm_kwargs)

# ── FAQ knowledge base ────────────────────────────────────────────────────────
FAQ_MAP: dict[str, list[dict[str, str]]] = {
    "billing": [
        {
            "q": "How do I update my payment method?",
            "a": "You can update your payment method by navigating to Account Settings > Billing > "
                 "Payment Methods and clicking 'Edit'. Changes take effect on your next billing cycle.",
        },
        {
            "q": "Why was I charged twice this month?",
            "a": "Double charges are usually caused by a failed transaction retry. Please check your "
                 "Billing History in your account dashboard. If the duplicate charge is confirmed, "
                 "contact support@example.com and we will issue a refund within 3–5 business days.",
        },
        {
            "q": "How do I cancel my subscription?",
            "a": "To cancel your subscription, go to Account Settings > Billing > Subscription and click "
                 "'Cancel Plan'. Your access continues until the end of the current billing period.",
        },
    ],
    "account": [
        {
            "q": "How do I reset my password?",
            "a": "Click 'Forgot Password' on the login page and enter your registered email address. "
                 "You will receive a reset link within 2 minutes. The link expires after 30 minutes.",
        },
        {
            "q": "How do I change my email address?",
            "a": "Go to Account Settings > Profile > Contact Information. Enter your new email and "
                 "click 'Verify'. A confirmation link will be sent to the new address before the change is applied.",
        },
        {
            "q": "How do I delete my account?",
            "a": "Account deletion is permanent. Go to Account Settings > Security > Delete Account. "
                 "You will be asked to confirm your password. All data is removed within 30 days per our retention policy.",
        },
    ],
    "technical": [
        {
            "q": "The app is not loading — what should I do?",
            "a": "First, clear your browser cache and cookies, then try a hard refresh (Ctrl+Shift+R / Cmd+Shift+R). "
                 "If the problem persists, try a different browser or disable browser extensions. "
                 "Check our status page at status.example.com for any ongoing incidents.",
        },
        {
            "q": "How do I integrate the API with my application?",
            "a": "Our REST API uses Bearer token authentication. Generate an API key in "
                 "Account Settings > Developer > API Keys. Full documentation, SDKs, and "
                 "code samples are available at docs.example.com/api.",
        },
        {
            "q": "Why is my data export failing?",
            "a": "Data exports larger than 100 MB are split into multiple files. Ensure you have "
                 "sufficient disk space and a stable connection. If the export job shows 'Failed' "
                 "in your dashboard, retry once; if it fails again, contact support with your export job ID.",
        },
    ],
}

FALLBACK_TOPICS = "billing, account, or technical"

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str           # classified topic: billing | account | technical | fallback
    confidence: float    # classifier confidence 0.0–1.0
    answer: str          # assembled answer text


# ── Message conversion ────────────────────────────────────────────────────────
def to_lc_messages(messages: list) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for m in messages:
        if not isinstance(m, dict):
            result.append(HumanMessage(content=str(m)))
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


# ── Nodes ─────────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are a question classifier for a customer support FAQ system.
Given a user question, classify it into exactly one of these topics:
  - billing   (payments, invoices, subscriptions, charges, refunds)
  - account   (login, password, email, profile, account deletion)
  - technical (app issues, API, integrations, data exports, errors)
  - fallback  (anything that does not clearly fit the above)

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{"topic": "<topic>", "confidence": <0.0-1.0>}"""


def classifier_node(state: AgentState) -> dict:
    """Classify the user question into a topic with a confidence score."""
    # Extract the latest user message
    user_msg = ""
    for m in reversed(state["messages"]):
        content = m.content if isinstance(m, BaseMessage) else m.get("content", "")
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else "human")
        if role in ("human", "user"):
            user_msg = content
            break

    classification_messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM),
        HumanMessage(content=f"User question: {user_msg}"),
    ]

    response = llm.invoke(classification_messages)
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
        topic = parsed.get("topic", "fallback").lower()
        confidence = float(parsed.get("confidence", 0.5))
        if topic not in FAQ_MAP and topic != "fallback":
            topic = "fallback"
    except (json.JSONDecodeError, ValueError):
        logger.warning("Classifier returned non-JSON: %s", raw)
        topic = "fallback"
        confidence = 0.0

    return {"topic": topic, "confidence": confidence}


def _find_best_faq_answer(topic: str, user_msg: str) -> str:
    """Return the single most relevant FAQ answer for the topic.

    Uses the LLM to pick the best match from the topic's FAQ entries.
    """
    entries = FAQ_MAP[topic]
    options = "\n".join(
        f"{i+1}. Q: {e['q']}\n   A: {e['a']}" for i, e in enumerate(entries)
    )
    pick_prompt = (
        f"The user asked: \"{user_msg}\"\n\n"
        f"Choose the most relevant FAQ answer from the list below and reply with ONLY the answer text:\n\n"
        f"{options}"
    )
    response = llm.invoke([HumanMessage(content=pick_prompt)])
    return response.content.strip()


def billing_node(state: AgentState) -> dict:
    """Return the most relevant billing FAQ answer."""
    user_msg = _get_last_user_message(state)
    answer = _find_best_faq_answer("billing", user_msg)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def account_node(state: AgentState) -> dict:
    """Return the most relevant account FAQ answer."""
    user_msg = _get_last_user_message(state)
    answer = _find_best_faq_answer("account", user_msg)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def technical_node(state: AgentState) -> dict:
    """Return the most relevant technical FAQ answer."""
    user_msg = _get_last_user_message(state)
    answer = _find_best_faq_answer("technical", user_msg)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def fallback_node(state: AgentState) -> dict:
    """Return a polite fallback response including the confidence score."""
    confidence = state.get("confidence", 0.0)
    answer = (
        f"I\'m sorry, I wasn\'t able to match your question to a known topic "
        f"(confidence: {confidence:.0%}). "
        f"I can help with {FALLBACK_TOPICS} questions. "
        f"Could you please rephrase your question or let me know which area you need help with?"
    )
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def _get_last_user_message(state: AgentState) -> str:
    """Extract the latest user message text from state."""
    for m in reversed(state["messages"]):
        if isinstance(m, BaseMessage):
            if m.type in ("human",):
                return m.content
        elif isinstance(m, dict):
            if m.get("role") in ("user", "human"):
                return m.get("content", "")
    return ""


# ── Routing ───────────────────────────────────────────────────────────────────
def route_topic(state: AgentState) -> str:
    """Route to the appropriate topic node based on the classifier output."""
    topic = state.get("topic", "fallback")
    if topic in ("billing", "account", "technical"):
        return topic
    return "fallback"


# ── Graph ─────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("classifier", classifier_node)
    g.add_node("billing", billing_node)
    g.add_node("account", account_node)
    g.add_node("technical", technical_node)
    g.add_node("fallback", fallback_node)

    g.set_entry_point("classifier")

    g.add_conditional_edges(
        "classifier",
        route_topic,
        {
            "billing": "billing",
            "account": "account",
            "technical": "technical",
            "fallback": "fallback",
        },
    )

    g.add_edge("billing", END)
    g.add_edge("account", END)
    g.add_edge("technical", END)
    g.add_edge("fallback", END)

    return g.compile()


_graph = None


@app.on_startup
async def init():
    global _graph
    _graph = build_graph()
    logger.info("FAQ router graph compiled")


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.entrypoint
async def invoke(
    messages: list,
    auth_context=None,
    session_id: str | None = None,
    streaming: bool = False,
) -> dict:
    """Route a user question to the appropriate FAQ topic and return a canned answer."""
    # Guard auth_context for None
    if auth_context:
        logger.debug("Request from user_id=%s", auth_context.user_id)

    global _graph
    if _graph is None:
        _graph = build_graph()

    lc_messages = to_lc_messages(messages)
    initial_state: AgentState = {
        "messages": lc_messages,
        "topic": "",
        "confidence": 0.0,
        "answer": "",
    }

    # streaming parameter accepted but not used (out of scope per story)
    result = await _graph.ainvoke(initial_state)

    answer = result.get("answer") or ""
    if not answer and result.get("messages"):
        last = result["messages"][-1]
        answer = last.content if isinstance(last, BaseMessage) else last.get("content", "")

    return {"content": answer, "role": "assistant"}
