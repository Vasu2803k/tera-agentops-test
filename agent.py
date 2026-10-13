# agent.py — generated from Jira story TER-1219
from __future__ import annotations

import os
import logging
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
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
model_name    = LITELLM_MODEL or os.getenv("LLM_MODEL", "gpt-4o-mini")

llm_kwargs: dict = {"model": model_name}
if LITELLM_BASE:
    llm_kwargs.update({"base_url": LITELLM_BASE, "api_key": LITELLM_KEY})
llm = ChatOpenAI(**llm_kwargs)

# ── FAQ maps (in-code, no external services) ─────────────────────────────────
FAQ: dict[str, dict[str, str]] = {
    "billing": {
        "How do I update my payment method?": (
            "Log in to your account, navigate to Billing > Payment Methods, "
            "and click 'Add / Replace Card'."
        ),
        "Why was I charged twice?": (
            "Duplicate charges are usually reversed within 3-5 business days. "
            "If not, please contact billing support with your invoice number."
        ),
        "How do I download my invoice?": (
            "Go to Billing > Invoices, select the period you need, "
            "and click 'Download PDF'."
        ),
    },
    "account": {
        "How do I reset my password?": (
            "Click 'Forgot Password' on the login page and follow the "
            "email instructions to set a new password."
        ),
        "How do I update my email address?": (
            "Go to Account Settings > Profile and change the email field. "
            "A verification link will be sent to the new address."
        ),
        "How do I delete my account?": (
            "Visit Account Settings > Danger Zone and click 'Delete Account'. "
            "This action is irreversible — all data will be removed within 30 days."
        ),
    },
    "technical": {
        "How do I reset my API key?": (
            "Navigate to Developer Settings > API Keys, revoke the current key, "
            "and generate a new one. Update your integrations with the new key."
        ),
        "Why is my integration returning 401 errors?": (
            "A 401 error means your API key is invalid or expired. "
            "Regenerate your key in Developer Settings and redeploy your integration."
        ),
        "How do I enable SSO?": (
            "SSO is available on Enterprise plans. Contact your account manager "
            "or open a support ticket to initiate the SSO setup workflow."
        ),
    },
}

FALLBACK_RESPONSE = (
    "I can help with **billing**, **account**, or **technical** questions. "
    "Could you rephrase your question under one of those topics?"
)

TOPICS = list(FAQ.keys())  # ["billing", "account", "technical"]

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str          # classified topic name (or "fallback")
    confidence: float   # classifier confidence 0.0–1.0


# ── Helper: convert dicts → LangChain messages ───────────────────────────────
def to_lc(messages: list) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for m in messages:
        role    = m.get("role", "user") if isinstance(m, dict) else "user"
        content = m.get("content", "")  if isinstance(m, dict) else str(m)
        if role == "assistant":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


# ── Node 1: classifier ────────────────────────────────────────────────────────
_CLASSIFIER_SYSTEM = """You are a FAQ router. Classify the user's question into exactly one
of the following topics: billing, account, technical.
If it does not fit any topic, reply with: fallback

You MUST reply in this exact JSON format and nothing else:
{"topic": "<billing|account|technical|fallback>", "confidence": <0.0-1.0>}"""


def classify(state: AgentState) -> dict:
    """Classify the latest user message into a topic + confidence."""
    import json

    user_msg = state["messages"][-1].content if state["messages"] else ""
    response = llm.invoke([
        SystemMessage(content=_CLASSIFIER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    try:
        parsed = json.loads(response.content.strip())
        topic      = parsed.get("topic", "fallback")
        confidence = float(parsed.get("confidence", 0.0))
        if topic not in TOPICS:
            topic = "fallback"
    except (json.JSONDecodeError, ValueError, KeyError):
        topic      = "fallback"
        confidence = 0.0

    logger.info("Classified as '%s' (confidence=%.2f)", topic, confidence)
    return {"topic": topic, "confidence": confidence}


# ── Node 2a-c: topic answer nodes ─────────────────────────────────────────────
def _answer_from_faq(topic: str, user_msg: str, confidence: float) -> str:
    """Return the best-matching FAQ answer for the given topic."""
    faq_map = FAQ[topic]
    # Simple keyword overlap to pick the best entry
    user_lower = user_msg.lower()
    best_key   = max(faq_map, key=lambda q: sum(w in user_lower for w in q.lower().split()))
    return (
        f"{faq_map[best_key]}\n\n"
        f"_(Topic: **{topic}** — confidence {confidence:.0%})_"
    )


def billing_node(state: AgentState) -> dict:
    """Return a billing FAQ answer."""
    user_msg = state["messages"][-1].content
    answer   = _answer_from_faq("billing", user_msg, state["confidence"])
    return {"messages": [AIMessage(content=answer)]}


def account_node(state: AgentState) -> dict:
    """Return an account FAQ answer."""
    user_msg = state["messages"][-1].content
    answer   = _answer_from_faq("account", user_msg, state["confidence"])
    return {"messages": [AIMessage(content=answer)]}


def technical_node(state: AgentState) -> dict:
    """Return a technical FAQ answer."""
    user_msg = state["messages"][-1].content
    answer   = _answer_from_faq("technical", user_msg, state["confidence"])
    return {"messages": [AIMessage(content=answer)]}


# ── Node 3: fallback ──────────────────────────────────────────────────────────
def fallback_node(state: AgentState) -> dict:
    """Return a polite fallback with detected confidence."""
    confidence = state.get("confidence", 0.0)
    answer = (
        f"{FALLBACK_RESPONSE}\n\n"
        f"_(Detected intent confidence: {confidence:.0%})_"
    )
    return {"messages": [AIMessage(content=answer)]}


# ── Routing function ──────────────────────────────────────────────────────────
def route_topic(state: AgentState) -> str:
    """Conditional edge: return the node name to route to."""
    return state.get("topic", "fallback")


# ── Graph assembly ────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("classify",  classify)
    g.add_node("billing",   billing_node)
    g.add_node("account",   account_node)
    g.add_node("technical", technical_node)
    g.add_node("fallback",  fallback_node)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        route_topic,
        {
            "billing":   "billing",
            "account":   "account",
            "technical": "technical",
            "fallback":  "fallback",
        },
    )

    g.add_edge("billing",   END)
    g.add_edge("account",   END)
    g.add_edge("technical", END)
    g.add_edge("fallback",  END)

    return g.compile()


graph = build_graph()


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.entrypoint
async def invoke(
    messages: list,
    auth_context=None,
    session_id: str | None = None,
    streaming: bool = False,
) -> dict:
    """AgentRuntime entrypoint for the FAQ router agent."""
    # Guard auth_context (may be None for unauthenticated requests)
    if auth_context:
        logger.info("Request from user_id=%s", auth_context.user_id)

    lc_messages = to_lc(messages)
    config = {"configurable": {"thread_id": session_id or "default"}}

    result = await graph.ainvoke(
        {"messages": lc_messages, "topic": "", "confidence": 0.0},
        config=config,
    )
    return {"content": result["messages"][-1].content, "role": "assistant"}
