# FAQ Router — Streamlit UI

A chat-style web interface for the **TER-1219 FAQ Router LangGraph agent**.

## Screenshot

```
┌─────────────────────────────────────────────────────────────┐
│  🗂️ FAQ Router          │  🤖 FAQ Router Chat               │
│  ─────────────────      │  ─────────────────────────────    │
│  Topics handled:        │                                   │
│  💳 billing             │  You: How do I reset my password?  │
│  👤 account             │                                   │
│  🔧 technical           │  🤖: To reset your password,       │
│                         │      go to Settings → Security…   │
│  Last routing decision  │                                   │
│  Topic:   👤 account    │  [Ask a question…          ]  ➤   │
│  Confidence:  92%       │                                   │
└─────────────────────────────────────────────────────────────┘
```

## Setup

```bash
# 1. Install all dependencies (from repo root)
pip install -r requirements.txt -r ui/requirements-ui.txt

# 2. Set required environment variables
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"          # or any LiteLLM-supported model

# 3. Launch the UI
streamlit run ui/app.py
```

Streamlit will open **http://localhost:8501** in your browser.

## How it works

| Layer | File | Role |
|-------|------|------|
| Agent | `../agent.py` | LangGraph `StateGraph` — classify → route → answer |
| UI | `ui/app.py` | Streamlit chat app — calls `graph.invoke()` directly |
| Config | `../config.yaml` | AgentRuntime manifest |

### Data flow

```
User types question
      │
      ▼
  app.py collects full message history
      │
      ▼
  graph.invoke({"messages": [...]})
      │
      ├─ classify node   → topic + confidence (LLM JSON)
      │
      ├─ billing / account / technical / fallback node
      │
      └─ returns {answer, topic, confidence}
            │
            ▼
  Streamlit renders assistant bubble
  Sidebar shows topic + confidence
```

## Features

- **Chat history** — full multi-turn conversation preserved in `st.session_state`.
- **Routing metadata** — sidebar shows the last detected topic and confidence score.
- **Clear button** — reset the conversation with one click.
- **Error handling** — agent exceptions are caught and displayed gracefully.
- **No HTTP server needed** — the graph is invoked in-process via `graph.invoke()`.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | API key for the LLM backend |
| `LLM_MODEL` | optional | Model name (default: `gpt-4o`) |
