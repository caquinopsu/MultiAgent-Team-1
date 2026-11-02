# ELP Team 2 — Multi-Agent Research Team

A Streamlit app that answers a business research question with a team of agents:

    Question → Supervisor → Specialists (parallel) → Shared Evidence (SQLite) → Synthesis → Reviewer → Report

## Run it

From this folder:

    python -m pip install -r requirements_team2.txt
    python -m streamlit run multiagent.py

The app needs a `.env` file in this folder containing `TAVILY_API_KEY` and `DEEPSEEK_API_KEY`.

## What is in this folder

| File | What it is |
| --- | --- |
| `multiagent.py` | **The app.** One file holding Weeks 3, 4, 5 and 6, in that order. |
| `requirements_team2.txt` | Python packages the app needs. |
| `.env` | Your API keys. Never commit this. |
| `team2_evidence.db` | SQLite database the app creates: `evidence` table (Week 4) and `reports` table (Week 5). Safe to delete; it is rebuilt on the next run. |

## Where each week lives in `multiagent.py`

| Week | Section | Status |
| --- | --- | --- |
| 3 | Supervisor, specialist agents, parallel execution, first button | Done |
| 4 | Evidence extraction, validation, SQLite store, second button | Done |
| 5 | Report synthesis, `reports` table, third button | Done |
| 6 | Reviewer/critic agent, one controlled revision, fourth button | Done |

Each week starts with a large docstring banner (`ELP Week 4 — ...`, `ELP WEEK 5 — ...`, `ELP WEEK 6 — ...`), so searching the file for `ELP W` jumps between them.

### Week 6 progress

| TODO | What | Status |
| --- | --- | --- |
| 0 | `review_report` — scores the draft on six criteria, treats malformed reviews as a failure, enforces the "all scores ≥ 7" PASS rule in code | Done |
| 1 | `revise_report_once` — one revision from existing evidence only; a section the model drops keeps its draft text | Done |
| 2 | `Week6ResearchState` — adds `review`, `initial_review`, `final_report_markdown`, `revision_performed` | Done |
| 3 | `reviewer_node` — review → revise at most once → review again | Done |
| 4 | `build_week6_graph` — Week 5 graph plus the reviewer node | Done |
| 5 | Week 6 UI — "Run Reviewed Synthesis Pipeline" button | Done |

Use the Week 6 button for the full flow. Only a PASS report is shown as final; a failed report is labelled unapproved and kept in an expander. The Week 5 draft is still saved to SQLite before review, so a saved report is not automatically approved.
