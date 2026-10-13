# agent.py — generated from Jira story TER-1219
from __future__ import annotations

import os
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

try:
    from .agent_runtime.runtime import AgentRuntimeApp
except ImportError:
    from agent_runtime.runtime import AgentRuntimeApp

logger = logging.getLogger(__name__)
app = AgentRuntimeApp()

# ── LLM setup ──────────────────────────────────────────────────────────────
LITELLM_BASE  = os.getenv("LITELLM_BASE_URL")
LITELLM_KEY   = os.getenv("LITELLM_API_KEY", "sk-1234")
LITELLM_MODEL = os.getenv("LITELLM_MODEL")
model_name    = LITELLM_MODEL or os.getenv("LLM_MODEL", "gpt-4o-mini")

llm_kwargs: dict = {"model": model_name}
if LITELLM_BASE:
    llm_kwargs.update({"base_url": LITELLM_BASE, "api_key": LITELLM_KEY})

llm = ChatOpenAI(**llm_kwargs)

# ── FAQ data (in-code, no external services) ────────────────────────────────
FAQ_MAP: dict[str, list[dict[str, str]]] = {
    "billing": [
        {
            "q": "How do I update my payment method?",
            "a": "Go to Account Settings → Billing → Payment Methods. "
                 "Click 'Edit' next to your current method and enter the new details.",
        },
        {
            "q": "Why was I charged twice this month?",
            "a": "Double charges are usually caused by a failed payment retry. "
                 "Please contact billing@support.example.com with your invoice number "
                 "and we will issue a refund within 3–5 business days.",
        },
        {
            "q": "How do I cancel my subscription?",
            "a": "You can cancel anytime from Account Settings → Billing → Subscription. "
                 "Your access continues until the end of the current billing period.",
        },
    ],
    "account": [
        {
            "q": "How do I reset my password?",
            "a": "Click 'Forgot Password' on the login page. "
                 "A reset link will be sent to your registered email address.",
        },
        {
            "q": "How do I change my email address?",
            "a": "Navigate to Account Settings → Profile → Email. "
                 "Enter your new email and verify it via the confirmation link we send.",
        },
        {
            "q": "How do I enable two-factor authentication?",
            "a": "Go to Account Settings → Security → Two-Factor Authentication "
                 "and follow the setup wizard. We support authenticator apps and SMS.",
        },
    ],
    "technical": [
        {
            "q": "Why is the application running slowly?",
            "a": "Try clearing your browser cache and cookies. "
                 "If the issue persists, check our status page at status.example.com "
                 "or contact technical support.",
        },
        {
            "q": "How do I integrate with your API?",
            "a": "Full API documentation is available at docs.example.com/api. "
                 "You will need an API key from Account Settings → Developer → API Keys.",
        },
        {
            "q": "What browsers are supported?",
            "a": "We support the two most recent versions of Chrome, Firefox, Edge, and Safari. "
                 "Internet Explorer is not supported.",
        },
    ],
}

KNOWN_TOPICS = list(FAQ_MAP.keys())  # ["billing", "account", "technical"]

# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:   Annotated[list, add_messages]
    topic:      str   # classified topic or "fallback"
    confidence: float # classifier confidence 0.0–1.0


# ── Helper: convert runtime dicts to LangChain messages ───────────────────
def to_lc_messages(messages: list) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for m in messages:
        if not isinstance(m, dict):
            result.append(HumanMessage(content=str(m)))
            continue
        role    = m.get("role", "user")
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


# ── Node 1: Classifier ─────────────────────────────────────────────────────
CLASSIFIER_SYSTEM = """You are an intent classifier for a customer support FAQ system.
Given a user question, determine which topic it belongs to from this fixed list:
  - billing   : payment, invoices, charges, subscriptions, refunds
  - account   : passwords, email, profile, security, two-factor authentication
  - technical : application errors, API, integrations, browser support, performance

Respond with EXACTLY one line in this format (no other text):
<topic> <confidence>

Where:
  <topic>      is one of: billing, account, technical, fallback
  <confidence> is a float between 0.0 and 1.0

If the question does not fit any topic well, use 'fallback'."""


def classify_question(state: AgentState) -> dict:
    """LLM-powered classifier: routes to billing / account / technical / fallback."""
    user_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=CLASSIFIER_SYSTEM),
        HumanMessage(content=f"Question: {user_message}"),
    ])

    raw = response.content.strip().lower()
    parts = raw.split()

    topic      = "fallback"
    confidence = 0.0

    if parts:
        candidate = parts[0]
        if candidate in KNOWN_TOPICS:
            topic = candidate
        elif candidate == "fallback":
            topic = "fallback"

    if len(parts) >= 2:
        try:
            confidence = float(parts[1])
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.8 if topic != "fallback" else 0.0
    elif topic != "fallback":
        confidence = 0.8  # default when classifier omits the score

    logger.info("Classified question as topic=%s confidence=%.2f", topic, confidence)
    return {"topic": topic, "confidence": confidence}


# ── Node 2-4: Topic answer nodes ───────────────────────────────────────────
def _answer_from_faq(topic: str, question: str, confidence: float) -> str:
    """Find the closest FAQ entry for a topic using the LLM."""
    faq_entries = FAQ_MAP[topic]
    faq_text = "\n".join(
        f"{i+1}. Q: {e['q']}\n   A: {e['a']}"
        for i, e in enumerate(faq_entries)
    )
    prompt = (
        f"The user asked: \"{question}\"\n\n"
        f"Below are FAQ entries for the '{topic}' topic. "
        f"Select the most relevant answer and return it verbatim, "
        f"or combine entries if the question spans multiple items. "
        f"If none match well, say so politely.\n\n"
        f"{faq_text}"
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip()


def answer_billing(state: AgentState) -> dict:
    """Return an answer from the billing FAQ."""
    question = state["messages"][-1].content
    answer   = _answer_from_faq("billing", question, state["confidence"])
    return {"messages": [AIMessage(content=answer)]}


def answer_account(state: AgentState) -> dict:
    """Return an answer from the account FAQ."""
    question = state["messages"][-1].content
    answer   = _answer_from_faq("account", question, state["confidence"])
    return {"messages": [AIMessage(content=answer)]}


def answer_technical(state: AgentState) -> dict:
    """Return an answer from the technical FAQ."""
    question = state["messages"][-1].content
    answer   = _answer_from_faq("technical", question, state["confidence"])
    return {"messages": [AIMessage(content=answer)]}


# ── Node 5: Fallback ───────────────────────────────────────────────────────
def answer_fallback(state: AgentState) -> dict:
    """Politely decline and hint at supported topics."""
    confidence = state.get("confidence", 0.0)
    response = (
        f"I'm sorry, I wasn't able to match your question to a known topic "
        f"(confidence: {confidence:.0%}). "
        "I can help with **billing**, **account**, or **technical** questions. "
        "Could you rephrase your question within one of those areas?"
    )
    return {"messages": [AIMessage(content=response)]}


# ── Conditional router ─────────────────────────────────────────────────────
def route_by_topic(state: AgentState) -> str:
    """Return the node name to route to based on the classified topic."""
    topic = state.get("topic", "fallback")
    routes = {
        "billing":   "billing",
        "account":   "account",
        "technical": "technical",
    }
    return routes.get(topic, "fallback")


# ── Graph ──────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("classifier", classify_question)
    g.add_node("billing",    answer_billing)
    g.add_node("account",    answer_account)
    g.add_node("technical",  answer_technical)
    g.add_node("fallback",   answer_fallback)

    # Entry point
    g.set_entry_point("classifier")

    # Conditional routing after classification
    g.add_conditional_edges(
        "classifier",
        route_by_topic,
        {
            "billing":   "billing",
            "account":   "account",
            "technical": "technical",
            "fallback":  "fallback",
        },
    )

    # All topic/fallback nodes terminate
    g.add_edge("billing",   END)
    g.add_edge("account",   END)
    g.add_edge("technical", END)
    g.add_edge("fallback",  END)

    return g.compile()


graph = build_graph()


# ── Entrypoint ─────────────────────────────────────────────────────────────
@app.entrypoint
async def invoke(
    messages: list,
    auth_context=None,
    session_id: str | None = None,
    streaming: bool = False,
) -> dict:
    """FAQ router agent — classifies the question and routes to the right answer."""
    # Auth guard (no role requirement for FAQ, but guard for None)
    if auth_context:
        logger.info(
            "Request from user_id=%s customer_id=%s",
            auth_context.user_id,
            auth_context.customer_id,
        )

    lc_messages = to_lc_messages(messages)
    config      = {"configurable": {"thread_id": session_id or "default"}}

    # Streaming parameter accepted but not required per story AC
    result = await graph.ainvoke(
        {"messages": lc_messages, "topic": "", "confidence": 0.0},
        config=config,
    )

    last_message = result["messages"][-1]
    return {"content": last_message.content, "role": "assistant"}
