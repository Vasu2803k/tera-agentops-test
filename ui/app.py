# ui/app.py  — Streamlit chat UI for the TER-1219 FAQ Router agent
"""Streamlit front-end that wraps the FAQ Router LangGraph agent.

Run:
    streamlit run ui/app.py

The UI:
- Renders a chat interface (user bubbles on the right, assistant on the left).
- Calls the agent's graph.invoke() directly (no HTTP server required).
- Displays the routing metadata (topic + confidence) in a sidebar panel.
- Keeps the full conversation history in Streamlit session state.
"""
from __future__ import annotations

import sys
import os

# Make sure the repo root is on the path so `agent` can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAQ Router — AgentRuntime Demo",
    page_icon="🤖",
    layout="centered",
)

# ── Load the agent graph (cached once per session) ───────────────────────────
@st.cache_resource(show_spinner="Loading FAQ Router agent…")
def load_graph():
    """Import and compile the LangGraph graph exactly once."""
    from agent import build_graph  # noqa: PLC0415
    return build_graph()

graph = load_graph()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []   # [{role, content, topic, confidence}]

# ── Sidebar — metadata & controls ────────────────────────────────────────────
with st.sidebar:
    st.title("🗂️ FAQ Router")
    st.markdown(
        """
        **Topics handled**
        - 💳 `billing`  — invoices, payments, refunds
        - 👤 `account`  — passwords, profiles, 2FA
        - 🔧 `technical` — bugs, errors, integrations

        Any other question gets a **fallback** response.
        """
    )
    st.divider()

    if st.session_state.history:
        last = st.session_state.history[-1]
        if last["role"] == "assistant":
            st.subheader("Last routing decision")
            topic = last.get("topic", "—")
            confidence = last.get("confidence", None)
            topic_emoji = {
                "billing": "💳",
                "account": "👤",
                "technical": "🔧",
                "fallback": "🔁",
            }.get(topic, "❓")
            st.metric("Topic", f"{topic_emoji} {topic}")
            if confidence is not None:
                try:
                    st.metric("Confidence", f"{float(confidence):.0%}")
                except (TypeError, ValueError):
                    st.metric("Confidence", str(confidence))
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🤖 FAQ Router Chat")
st.caption("Ask anything about billing, account management, or technical issues.")

# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question…"):
    # 1. Show user message immediately
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build the messages list for the agent
    agent_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.history
    ]

    # 3. Invoke the graph
    with st.chat_message("assistant"):
        with st.spinner("Routing your question…"):
            try:
                result = graph.invoke({"messages": agent_messages})
                answer = result.get("answer") or result.get("messages", [{}])[-1].get("content", "")
                topic = result.get("topic", "unknown")
                confidence = result.get("confidence", None)
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Agent error: {exc}"
                topic = "error"
                confidence = None

        st.markdown(answer)

    # 4. Store assistant reply
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
            "topic": topic,
            "confidence": confidence,
        }
    )
    st.rerun()
