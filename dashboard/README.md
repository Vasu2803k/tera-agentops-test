# FAQ Router — Analytics Dashboard

A Streamlit analytics dashboard for the **TER-1219 FAQ Router LangGraph Agent**.
Displays real-time topic routing, confidence scores, and query history across three tabs.

---

## Tabs

| Tab | Content |
|-----|---------|
| 💬 **Live Chat** | Send questions; agent classifies them live; topic badge + confidence shown inline |
| 📈 **Analytics** | Topic distribution (donut/bar/treemap), confidence histogram, sparkline, per-topic box plots |
| 📋 **Query Log** | Filterable + sortable table of all queries; CSV download |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r dashboard/requirements-dashboard.txt

# 2. Set required environment variable
export OPENAI_API_KEY="sk-..."

# 3. Run from the project root
streamlit run dashboard/dashboard.py
```

Opens at **http://localhost:8501** by default.

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes (live mode) | — | OpenAI key for the agent |
| `LLM_MODEL` | No | `gpt-4o-mini` | LLM model name |
| `LITELLM_BASE_URL` | No | — | LiteLLM proxy URL |
| `LITELLM_API_KEY` | No | — | LiteLLM proxy key |

---

## Modes

### Live mode
When `agent.py` is importable and `OPENAI_API_KEY` is set, every chat question is
routed through the real LangGraph agent.

### Demo mode
When the agent cannot be imported the dashboard seeds itself with 15 pre-built
queries and simulates routing. All charts and the query log are fully functional.

---

## Sidebar controls

| Control | Effect |
|---------|--------|
| **Confidence threshold** slider | Marks queries below threshold as "Needs Review" in KPIs and charts |
| **Topic chart style** | Switches distribution chart between Donut, Bar, and Treemap |
| **Show raw JSON log** | Dumps the full query log as JSON in the Query Log tab |
| **Clear session** | Resets all session state |

---

## Project layout

```
tera-agentops-test/
├── agent.py                    ← LangGraph FAQ Router agent
├── config.yaml
├── requirements.txt
├── dashboard/
│   ├── dashboard.py            ← This dashboard
│   ├── requirements-dashboard.txt
│   └── README.md
└── ui/
    ├── app.py                  ← Simpler chat UI
    ├── requirements-ui.txt
    └── README.md
```
