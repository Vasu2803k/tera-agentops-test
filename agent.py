# agent.py  — generated from Jira story TER-1219
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

# ── LLM setup ────────────────────────────────────────────────────────────────
LITELLM_BASE  = os.getenv("LITELLM_BASE_URL")
LITELLM_KEY   = os.getenv("LITELLM_API_KEY", "sk-1234")
LITELLM_MODEL = os.getenv("LITELLM_MODEL")

model_name = LITELLM_MODEL or os.getenv("LLM_MODEL", "gpt-4o-mini")
llm_kwargs: dict = {"model": model_name}
if LITELLM_BASE:
    llm_kwargs.update({"base_url": LITELLM_BASE, "api_key": LITELLM_KEY})
llm = ChatOpenAI(**llm_kwargs)

# ── FAQ knowledge base ───────────────────────────────────────────────────────
FAQ: dict[str, dict[str, str]] = {
    "billing": {
        "how do i update my payment method": (
            "To update your payment method, go to Account Settings → Billing → "
            "Payment Methods and click 'Edit'."
        ),
        "why was i charged twice": (
            "Duplicate charges are usually caused by a failed retry. Please "
            "contact support@example.com with your invoice number and we'll "
            "resolve it within 1 business day."
        ),
        "how do i get a refund": (
            "Refunds for unused time are processed within 5-7 business days. "
            "Submit a request at support.example.com/refunds."
        ),
    },
    "account": {
        "how do i reset my password": (
            "Click 'Forgot Password' on the login page and enter your email. "
            "You'll receive a reset link within 5 minutes."
        ),
        "how do i change my email address": (
            "Go to Account Settings → Profile → Contact Info and update your "
            "email. A verification link will be sent to the new address."
        ),
        "how do i delete my account": (
            "To permanently delete your account, navigate to Account Settings → "
            "Security → Delete Account. This action is irreversible."
        ),
    },
    "technical": {
        "why is the app slow": (
            "App slowness is often caused by a poor network connection or "
            "an outdated app version. Try refreshing or updating to the latest release."
        ),
        "how do i report a bug": (
            "Please report bugs via our GitHub Issues page or email "
            "bugs@example.com with steps to reproduce and a screenshot."
        ),
        "the app wont load": (
            "Try clearing your browser cache (Ctrl+Shift+Delete), disabling "
            "browser extensions, or switching to an incognito window."
        ),
    },
}

TOPICS = list(FAQ.keys())  # ["billing", "account", "technical"]

# ── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str          # classified topic name
    confidence: float   # classifier confidence 0.0–1.0
    answer: str         # final answer text


# ── Message conversion utility ───────────────────────────────────────────────
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


# ── Nodes ────────────────────────────────────────────────────────────────────
def classify(state: AgentState) -> dict:
    """Use the LLM to classify the user's question into a topic + confidence."""
    user_question = state["messages"][-1].content

    system_prompt = (
        "You are a question classifier. Given a user question, decide which support "
        "topic it belongs to: billing, account, or technical.\n"
        "If none fits, output 'unknown'.\n"
        "Respond with EXACTLY two lines:\n"
        "TOPIC: <billing|account|technical|unknown>\n"
        "CONFIDENCE: <float between 0.0 and 1.0>\n"
        "No other text."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question),
    ])

    topic      = "unknown"
    confidence = 0.5
    for line in response.content.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("TOPIC:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw in TOPICS:
                topic = raw
            else:
                topic = "unknown"
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.5

    logger.info("Classified question as topic=%s confidence=%.2f", topic, confidence)
    return {"topic": topic, "confidence": confidence}


def _faq_answer(topic: str, question: str) -> str:
    """Look up the best FAQ answer for a given topic and question."""
    faq_map = FAQ.get(topic, {})
    q_lower = question.lower()
    # Exact keyword match first
    for key, answer in faq_map.items():
        if key in q_lower or any(word in q_lower for word in key.split()):
            return answer
    # Default to first answer in the topic if no close match
    if faq_map:
        return next(iter(faq_map.values()))
    return "I'm sorry, I don't have a specific answer for that."


def handle_billing(state: AgentState) -> dict:
    """Return a billing-related FAQ answer."""
    question = state["messages"][-1].content
    answer   = _faq_answer("billing", question)
    return {"answer": answer}


def handle_account(state: AgentState) -> dict:
    """Return an account-related FAQ answer."""
    question = state["messages"][-1].content
    answer   = _faq_answer("account", question)
    return {"answer": answer}


def handle_technical(state: AgentState) -> dict:
    """Return a technical-related FAQ answer."""
    question = state["messages"][-1].content
    answer   = _faq_answer("technical", question)
    return {"answer": answer}


def handle_fallback(state: AgentState) -> dict:
    """Polite fallback for unrecognized questions."""
    confidence = state.get("confidence", 0.0)
    answer = (
        f"I can help with billing, account, or technical questions. "
        f"Your question didn't clearly match any of those topics "
        f"(detected intent confidence: {confidence:.0%}). "
        f"Could you please rephrase or choose one of the supported topics?"
    )
    return {"answer": answer}


def format_response(state: AgentState) -> dict:
    """Wrap the answer as an AIMessage for the messages list."""
    return {"messages": [AIMessage(content=state["answer"])]}


# ── Routing function ──────────────────────────────────────────────────────────
def route_topic(state: AgentState) -> str:
    """Conditional edge: route to the appropriate topic handler."""
    return state.get("topic", "unknown")


# ── Graph ────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("classify",         classify)
    g.add_node("billing",          handle_billing)
    g.add_node("account",          handle_account)
    g.add_node("technical",        handle_technical)
    g.add_node("fallback",         handle_fallback)
    g.add_node("format_response",  format_response)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        route_topic,
        {
            "billing":   "billing",
            "account":   "account",
            "technical": "technical",
            "unknown":   "fallback",
        },
    )

    g.add_edge("billing",   "format_response")
    g.add_edge("account",   "format_response")
    g.add_edge("technical", "format_response")
    g.add_edge("fallback",  "format_response")
    g.add_edge("format_response", END)

    return g.compile()


graph = build_graph()


# ── Entrypoint ───────────────────────────────────────────────────────────────
@app.entrypoint
async def invoke(
    messages: list,
    auth_context=None,
    session_id: str | None = None,
    streaming: bool = False,
) -> dict:
    """FAQ router entrypoint — classifies and routes user questions."""
    if auth_context and auth_context.has_role("admin"):
        logger.info("Request from admin user: %s", auth_context.user_id)

    lc_messages = to_lc_messages(messages)
    initial_state: AgentState = {
        "messages":   lc_messages,
        "topic":      "",
        "confidence": 0.0,
        "answer":     "",
    }

    config = {"configurable": {"thread_id": session_id or "default"}}
    result = await graph.ainvoke(initial_state, config=config)

    final_message = result["messages"][-1]
    return {"content": final_message.content, "role": "assistant"}
