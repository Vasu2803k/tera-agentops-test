# dashboard/dashboard.py  — generated from Jira story TER-1219
# FAQ Router Agent — Analytics Dashboard
# Run: streamlit run dashboard/dashboard.py

from __future__ import annotations

import sys
import os
import re
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Add project root so agent.py can be imported ─────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from agent import graph, FAQ_TOPICS  # type: ignore
    AGENT_AVAILABLE = True
except Exception:
    AGENT_AVAILABLE = False
    FAQ_TOPICS = [
        "Billing", "Technical Support", "Account Management",
        "Product Features", "Shipping & Delivery", "Returns & Refunds",
        "Privacy & Security", "General Inquiry",
    ]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAQ Router — Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .topic-badge {
    display: inline-block;
    background: #667eea;
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .confidence-high   { color: #22c55e; font-weight: 700; }
  .confidence-medium { color: #f59e0b; font-weight: 700; }
  .confidence-low    { color: #ef4444; font-weight: 700; }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
DEFAULT_STATE: dict = {
    "messages": [],
    "query_log": [],
    "total_queries": 0,
    "topic_counts": Counter(),
    "confidence_scores": [],
    "session_start": datetime.now().isoformat(),
    "demo_loaded": False,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Colour palette for topics ─────────────────────────────────────────────────
TOPIC_COLOURS = {
    "Billing":              "#6366f1",
    "Technical Support":    "#3b82f6",
    "Account Management":   "#8b5cf6",
    "Product Features":     "#06b6d4",
    "Shipping & Delivery":  "#10b981",
    "Returns & Refunds":    "#f59e0b",
    "Privacy & Security":   "#ef4444",
    "General Inquiry":      "#64748b",
}


def topic_colour(topic: str) -> str:
    return TOPIC_COLOURS.get(topic, "#94a3b8")


# ── Agent invocation helper ───────────────────────────────────────────────────
def call_agent(user_input: str) -> dict:
    if not AGENT_AVAILABLE:
        topic = random.choice(FAQ_TOPICS)
        confidence = round(random.uniform(0.55, 0.98), 2)
        return {
            "content": (
                f"[Demo] This question is about **{topic}** "
                f"(confidence {confidence:.0%}).\n\n"
                f"Running in demo mode — agent not connected."
            ),
            "topic": topic,
            "confidence": confidence,
        }
    try:
        result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
        last = result["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)
        topic = result.get("topic", "General Inquiry")
        confidence = float(result.get("confidence", 0.75))
        tm = re.search(r"(?i)topic[:\s]+([A-Za-z &]+)", content)
        cm = re.search(r"(?i)confidence[:\s]+([0-9.]+)", content)
        if tm:
            topic = tm.group(1).strip()
        if cm:
            try:
                val = float(cm.group(1))
                confidence = val / 100.0 if val > 1.0 else val
            except ValueError:
                pass
        return {"content": content, "topic": topic, "confidence": confidence}
    except Exception as exc:
        return {
            "content": f"⚠️ Agent error: {exc}",
            "topic": "General Inquiry",
            "confidence": 0.0,
        }


def record_interaction(user_msg: str, response: dict) -> None:
    ts = datetime.now()
    topic = response.get("topic", "General Inquiry")
    conf = float(response.get("confidence", 0.75))
    st.session_state.messages.append({
        "role": "user", "content": user_msg,
        "topic": topic, "confidence": conf, "ts": ts.isoformat(),
    })
    st.session_state.messages.append({
        "role": "assistant", "content": response["content"],
        "topic": topic, "confidence": conf, "ts": ts.isoformat(),
    })
    st.session_state.total_queries += 1
    st.session_state.topic_counts[topic] += 1
    st.session_state.confidence_scores.append(conf)
    st.session_state.query_log.append({
        "#": st.session_state.total_queries,
        "Timestamp": ts.strftime("%H:%M:%S"),
        "Question": user_msg[:80] + ("…" if len(user_msg) > 80 else ""),
        "Topic": topic,
        "Confidence": conf,
        "Status": "✅" if conf >= 0.75 else ("⚠️" if conf >= 0.50 else "❌"),
    })


# ── Seed demo data ────────────────────────────────────────────────────────────
DEMO_QUERIES = [
    ("How do I reset my password?",              "Account Management",  0.93),
    ("When will my order arrive?",               "Shipping & Delivery", 0.89),
    ("I was charged twice for my subscription.", "Billing",             0.96),
    ("The app keeps crashing on iOS 17.",        "Technical Support",   0.91),
    ("Can I export my data?",                    "Privacy & Security",  0.78),
    ("What's included in the Pro plan?",         "Product Features",    0.85),
    ("How do I return a damaged item?",          "Returns & Refunds",   0.88),
    ("Do you support SSO with Okta?",            "Technical Support",   0.72),
    ("Update my billing address.",               "Billing",             0.94),
    ("Track my shipment.",                       "Shipping & Delivery", 0.90),
    ("Delete my account permanently.",           "Account Management",  0.87),
    ("I need a refund for order #4521.",         "Returns & Refunds",   0.95),
    ("What is your privacy policy?",             "Privacy & Security",  0.80),
    ("Compare Starter vs Enterprise plans.",     "Product Features",    0.82),
    ("General question about your company.",     "General Inquiry",     0.61),
]

if not st.session_state.demo_loaded:
    base_ts = datetime.now() - timedelta(minutes=len(DEMO_QUERIES))
    for i, (q, topic, conf) in enumerate(DEMO_QUERIES):
        ts = base_ts + timedelta(minutes=i)
        st.session_state.messages.append({
            "role": "user", "content": q,
            "topic": topic, "confidence": conf, "ts": ts.isoformat(),
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Routing to **{topic}** (confidence {conf:.0%}).",
            "topic": topic, "confidence": conf, "ts": ts.isoformat(),
        })
        st.session_state.total_queries += 1
        st.session_state.topic_counts[topic] += 1
        st.session_state.confidence_scores.append(conf)
        st.session_state.query_log.append({
            "#": st.session_state.total_queries,
            "Timestamp": ts.strftime("%H:%M:%S"),
            "Question": q,
            "Topic": topic,
            "Confidence": conf,
            "Status": "✅" if conf >= 0.75 else ("⚠️" if conf >= 0.50 else "❌"),
        })
    st.session_state.demo_loaded = True


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 FAQ Router Agent")
    st.caption("TER-1219 — Analytics Dashboard")
    st.divider()

    st.subheader("⚙️ Settings")
    conf_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.75, 0.05,
        help="Queries below this are flagged for review",
    )
    chart_type = st.selectbox("Topic chart style", ["Donut", "Bar", "Treemap"])
    show_raw = st.checkbox("Show raw JSON log", value=False)

    st.divider()
    st.subheader("🔗 Status")
    if AGENT_AVAILABLE:
        st.success("Agent connected")
    else:
        st.warning("Demo mode (agent not imported)")

    st.caption(f"Session started: {st.session_state.session_start[:19]}")
    st.divider()
    if st.button("🗑️ Clear session", use_container_width=True):
        for k, v in DEFAULT_STATE.items():
            st.session_state[k] = v
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 FAQ Router — Analytics Dashboard")
st.caption("Real-time monitoring for the TER-1219 LangGraph FAQ Router Agent")
st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
total     = st.session_state.total_queries
scores    = st.session_state.confidence_scores
avg_conf  = (sum(scores) / len(scores)) if scores else 0.0
high_conf = sum(1 for s in scores if s >= conf_threshold)
low_conf  = total - high_conf
top_topic = (
    st.session_state.topic_counts.most_common(1)[0][0]
    if st.session_state.topic_counts else "—"
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Queries",   f"{total:,}")
col2.metric("Avg Confidence",  f"{avg_conf:.1%}")
col3.metric(
    "High Confidence", f"{high_conf:,}",
    delta=f"{high_conf/total:.0%} of total" if total else None,
)
col4.metric(
    "Needs Review", f"{low_conf:,}",
    delta=f"-{low_conf/total:.0%}" if total else None,
    delta_color="inverse",
)
col5.metric("Top Topic", top_topic)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_analytics, tab_log = st.tabs([
    "💬 Live Chat",
    "📈 Analytics",
    "📋 Query Log",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Live Chat Interface")
    st.caption("Ask a question — the agent classifies it in real time.")

    chat_area = st.container(height=420)
    with chat_area:
        for msg in st.session_state.messages[-40:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("topic"):
                    conf = msg["confidence"]
                    cc = (
                        "confidence-high"   if conf >= 0.75 else
                        "confidence-medium" if conf >= 0.50 else
                        "confidence-low"
                    )
                    st.markdown(
                        f'<span class="topic-badge">{msg["topic"]}</span> '
                        f'<span class="{cc}">{conf:.0%}</span>',
                        unsafe_allow_html=True,
                    )

    user_input = st.chat_input("Type your FAQ question here…")
    if user_input:
        with st.spinner("Routing question…"):
            response = call_agent(user_input)
        record_interaction(user_input, response)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    if total == 0:
        st.info("No queries yet. Send a message in the Live Chat tab.")
    else:
        c1, c2 = st.columns(2)

        # ── Topic distribution ─────────────────────────────────────────────
        with c1:
            st.subheader("Topic Distribution")
            topic_df = pd.DataFrame(
                [(t, c) for t, c in st.session_state.topic_counts.items()],
                columns=["Topic", "Count"],
            ).sort_values("Count", ascending=False)
            colour_map = {t: topic_colour(t) for t in topic_df["Topic"]}

            if chart_type == "Donut":
                fig = px.pie(
                    topic_df, names="Topic", values="Count", hole=0.5,
                    color="Topic", color_discrete_map=colour_map,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))

            elif chart_type == "Bar":
                fig = px.bar(
                    topic_df, x="Count", y="Topic", orientation="h",
                    color="Topic", text="Count", color_discrete_map=colour_map,
                )
                fig.update_layout(
                    showlegend=False,
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(t=10, b=10, l=10, r=10),
                )

            else:  # Treemap
                fig = px.treemap(
                    topic_df, path=["Topic"], values="Count",
                    color="Count", color_continuous_scale="Blues",
                )
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))

            st.plotly_chart(fig, use_container_width=True)

        # ── Confidence histogram ───────────────────────────────────────────
        with c2:
            st.subheader("Confidence Score Distribution")
            conf_df = pd.DataFrame({"Confidence": scores})
            fig2 = px.histogram(
                conf_df, x="Confidence", nbins=20, range_x=[0, 1],
                color_discrete_sequence=["#6366f1"],
                labels={"Confidence": "Confidence Score"},
            )
            fig2.add_vline(
                x=conf_threshold, line_dash="dash", line_color="#ef4444",
                annotation_text=f"Threshold ({conf_threshold:.0%})",
                annotation_position="top right",
            )
            fig2.add_vline(
                x=avg_conf, line_dash="dot", line_color="#22c55e",
                annotation_text=f"Avg ({avg_conf:.0%})",
                annotation_position="top left",
            )
            fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Confidence over time ───────────────────────────────────────────
        st.subheader("Confidence Over Time")
        log_df = pd.DataFrame(st.session_state.query_log)
        if not log_df.empty:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=log_df["#"],
                y=log_df["Confidence"],
                mode="lines+markers",
                line=dict(color="#6366f1", width=2),
                marker=dict(
                    size=7,
                    color=log_df["Confidence"],
                    colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
                    cmin=0, cmax=1,
                    showscale=True,
                    colorbar=dict(title="Conf", len=0.5),
                ),
                hovertemplate="<b>Q%{x}</b><br>Confidence: %{y:.0%}<extra></extra>",
            ))
            fig3.add_hline(
                y=conf_threshold, line_dash="dash", line_color="#ef4444",
                annotation_text="Threshold",
            )
            fig3.update_layout(
                xaxis_title="Query #",
                yaxis_title="Confidence",
                yaxis_range=[0, 1.05],
                height=260,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── Per-topic box plots ────────────────────────────────────────────
        st.subheader("Confidence by Topic")
        if not log_df.empty:
            fig4 = px.box(
                log_df, x="Topic", y="Confidence",
                color="Topic",
                color_discrete_map={t: topic_colour(t) for t in log_df["Topic"].unique()},
                points="all",
            )
            fig4.add_hline(y=conf_threshold, line_dash="dash", line_color="#ef4444")
            fig4.update_layout(
                showlegend=False,
                xaxis_tickangle=-30,
                height=320,
                margin=dict(t=10, b=60, l=10, r=10),
            )
            st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUERY LOG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.subheader("Query Log")
    if not st.session_state.query_log:
        st.info("No queries logged yet.")
    else:
        log_df = pd.DataFrame(st.session_state.query_log)

        f1, f2, f3 = st.columns([2, 2, 1])
        with f1:
            topic_filter = st.multiselect(
                "Filter by topic",
                options=sorted(log_df["Topic"].unique().tolist()),
                default=[],
            )
        with f2:
            status_filter = st.multiselect(
                "Filter by status",
                options=["✅", "⚠️", "❌"],
                default=[],
            )
        with f3:
            sort_by = st.selectbox(
                "Sort by", ["# (newest)", "Confidence ↓", "Confidence ↑"],
            )

        view = log_df.copy()
        if topic_filter:
            view = view[view["Topic"].isin(topic_filter)]
        if status_filter:
            view = view[view["Status"].isin(status_filter)]
        if sort_by == "# (newest)":
            view = view.sort_values("#", ascending=False)
        elif sort_by == "Confidence ↓":
            view = view.sort_values("Confidence", ascending=False)
        else:
            view = view.sort_values("Confidence", ascending=True)

        def colour_conf(val: float) -> str:
            if val >= 0.75:
                return "color: #22c55e; font-weight: 700"
            if val >= 0.50:
                return "color: #f59e0b; font-weight: 700"
            return "color: #ef4444; font-weight: 700"

        styled = (
            view.style
            .applymap(colour_conf, subset=["Confidence"])
            .format({"Confidence": "{:.0%}"})
            .hide(axis="index")
        )
        st.dataframe(styled, use_container_width=True, height=420)

        csv = view.drop(columns=["Status"]).to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="faq_router_query_log.csv",
            mime="text/csv",
        )

        if show_raw:
            st.subheader("Raw JSON")
            st.json(st.session_state.query_log)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"FAQ Router Analytics Dashboard · TER-1219 · "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
