# ELP Team 2 — Multi-Agent Research Team

A Streamlit app that answers a business research question with a team of agents:

    Question → Supervisor → Specialists (parallel) → Shared Evidence (SQLite) → Synthesis → Report

## Run it

From this folder:

    python -m pip install -r requirements_team2.txt
    python -m streamlit run week3_4.py

The app needs a `.env` file in this folder containing `TAVILY_API_KEY` and `DEEPSEEK_API_KEY`.

## What is in this folder

| File | What it is |
| --- | --- |
| `week3_4.py` | **The app.** One file holding Weeks 3, 4 and 5, in that order. |
| `requirements_team2.txt` | Python packages the app needs. |
| `.env` | Your API keys. Never commit this. |
| `team2_evidence.db` | SQLite database the app creates: `evidence` table (Week 4) and `reports` table (Week 5). Safe to delete; it is rebuilt on the next run. |
| `earlier_weeks/week2_supervisor.py` | The standalone Week 2 app (Supervisor → Tavily → Response). Kept for reference. |
| `earlier_weeks/week4_starter.py` | The original, unfinished Week 4 starter. Its completed version is already inside `week3_4.py`. |

## Where each week lives in `week3_4.py`

| Week | Section | Status |
| --- | --- | --- |
| 3 | Supervisor, specialist agents, parallel execution, first button | Done |
| 4 | Evidence extraction, validation, SQLite store, second button | Done |
| 5 | Report synthesis, `reports` table, third button | Done |
| 6 | Citation verification, reviewer/critic agent | Not started |

Each week starts with a large docstring banner (`ELP Week 4 — ...`, `ELP WEEK 5 — ...`), so searching the file for `ELP W` jumps between them.
