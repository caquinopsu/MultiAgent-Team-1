"""ELP Week 3 — Team 2 Student Starter
Agentic Research Team: Supervisor + Specialist Agents + Parallelism

Week 2 already gave you:
    Question -> Supervisor -> first assignment -> Tavily -> Response

Week 3 goal:
    Question -> Supervisor -> Specialist Agents in Parallel -> Specialist Results

Your job is to complete the NEW Week 3 TODOs only.

Run:
    python -m pip install -r requirements_team2.txt
    streamlit run team2_week3_student.py

.env:
    TAVILY_API_KEY=your_key
    DEEPSEEK_API_KEY=your_key
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from tavily import TavilyClient


# ---------------- Configuration ----------------
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not TAVILY_API_KEY or not DEEPSEEK_API_KEY:
    st.error(
        "Missing API key. Add TAVILY_API_KEY and DEEPSEEK_API_KEY "
        "to a .env file beside this script."
    )
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# ---------------- Team 2 Supervisor ----------------
# This carries forward Week 2 homework: each assignment includes output_fields.
TEAM_2_SYSTEM_PROMPT = """
You are the Supervisor for a multi-agent competitive-intelligence research system.

Divide the user's business research question across exactly three specialist roles:
Competitor, Market, and Tech/Regulatory.

Each specialist must receive a distinct, non-overlapping objective. Include a short
output_fields list describing what that specialist should return.

Return only valid JSON in exactly this shape:
{
  "tasks": [
    {
      "agent": "Competitor",
      "objective": "...",
      "output_fields": ["...", "..."]
    },
    {
      "agent": "Market",
      "objective": "...",
      "output_fields": ["...", "..."]
    },
    {
      "agent": "Tech/Regulatory",
      "objective": "...",
      "output_fields": ["...", "..."]
    }
  ]
}

Use exactly these three agent names.
Do not include markdown or text outside the JSON object.
""".strip()

SPECIALIST_PROMPTS = {
    "Competitor": """
You are the Competitor specialist. Research competitors, products, positioning,
partnerships, and differentiators relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Market or Tech/Regulatory agent's job.
""".strip(),
    "Market": """
You are the Market specialist. Research market direction, customers, adoption,
industry dynamics, and business context relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Competitor or Tech/Regulatory agent's job.
""".strip(),
    "Tech/Regulatory": """
You are the Tech/Regulatory specialist. Research technology developments,
technical capabilities, emerging technology, regulation, and official policy
context relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Competitor or Market agent's job.
""".strip(),
}


class ResearchState(TypedDict):
    question: str
    assignments: list[dict]
    specialist_results: list[dict]


def call_model_json(system_prompt: str, user_text: str) -> dict:
    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def supervisor_node(state: ResearchState) -> dict:
    output = call_model_json(TEAM_2_SYSTEM_PROMPT, state["question"])
    tasks = output.get("tasks")

    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("Supervisor must return exactly three assignments.")

    expected_agents = {"Competitor", "Market", "Tech/Regulatory"}
    seen_agents = set()
    cleaned = []

    for assignment in tasks:
        if not isinstance(assignment, dict):
            raise ValueError("Each assignment must be a JSON object.")

        agent = str(assignment.get("agent", "")).strip()
        objective = str(assignment.get("objective", "")).strip()
        output_fields = assignment.get("output_fields", [])

        if agent not in expected_agents:
            raise ValueError(f"Unexpected specialist role: {agent}")
        if not objective:
            raise ValueError(f"{agent} needs a non-empty objective.")
        if not isinstance(output_fields, list):
            raise ValueError(f"{agent} output_fields must be a list.")

        seen_agents.add(agent)
        cleaned.append(
            {
                "agent": agent,
                "objective": objective,
                "output_fields": [str(field) for field in output_fields],
            }
        )

    if seen_agents != expected_agents:
        raise ValueError(
            "Assignments must include Competitor, Market, and Tech/Regulatory exactly once."
        )

    return {"assignments": cleaned}


# ---------------- Week 3 specialist execution ----------------
def research_assignment(assignment: dict, original_question: str) -> dict:
    """Execute ONE specialist's assignment."""

    # TODO 1:
    # 1. Read:
    #       agent = assignment["agent"]
    #       objective = assignment["objective"]
    #       output_fields = assignment["output_fields"]
    #
    # 2. Search Tavily using the specialist's objective.
    #
    # 3. Format the returned Tavily results into readable source text containing
    #    title, URL, and content.
    #
    # 4. Call DeepSeek with SPECIALIST_PROMPTS[agent].
    #    Give the model:
    #       - the original question
    #       - this specialist's objective
    #       - requested output_fields
    #       - Tavily source text
    #
    # 5. Return ONE dictionary in exactly this shape:
    #
    #    {
    #        "agent": agent,
    #        "objective": objective,
    #        "output_fields": output_fields,
    #        "research": model_response_text,
    #        "sources": tavily_results,
    #    }
    #
    # This function should execute only ONE specialist. Do not loop over all
    # assignments here.

    agent = assignment["agent"]
    objective = assignment["objective"]
    output_fields = assignment["output_fields"]

    tavily_answer = tavily.search(
        query=objective,
        search_depth="advanced",
        max_results=5,
    )
    tavily_results = tavily_answer.get("results", [])

    source_text = "\n\n".join(
        f"Title: {result.get('title', '')}\n"
        f"URL: {result.get('url', '')}\n"
        f"Content: {result.get('content', '')}"
        for result in tavily_results
    )

    user_text = (
        f"Original question: {original_question}\n\n"
        f"Your objective: {objective}\n\n"
        f"Requested output fields: {', '.join(output_fields)}\n\n"
        f"Tavily sources:\n{source_text or 'No sources returned.'}"
    )

    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": SPECIALIST_PROMPTS[agent]},
            {"role": "user", "content": user_text},
        ],
    )
    model_response_text = response.choices[0].message.content

    return {
        "agent": agent,
        "objective": objective,
        "output_fields": output_fields,
        "research": model_response_text,
        "sources": tavily_results,
    }


def specialists_node(state: ResearchState) -> dict:
    """Run all three independent specialist assignments in parallel."""
    specialist_results = []

    # TODO 2:
    # Use ThreadPoolExecutor(max_workers=3).
    #
    # Submit research_assignment(assignment, state["question"]) once for each
    # assignment in state["assignments"].
    #
    # Use as_completed(...) to collect the finished results and append each one
    # to specialist_results.
    #
    # Important:
    # - The specialists should execute independently.
    # - Do not call them one-by-one in a normal sequential for-loop.
    # - Week 4 will add the shared evidence store; do not build it here.
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(research_assignment, assignment, state["question"])
            for assignment in state["assignments"]
        ]
        for future in as_completed(futures):
            specialist_results.append(future.result())

    return {"specialist_results": specialist_results}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", END)

    return graph.compile()


research_graph = build_graph()


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ELP Team 2 — Week 3", page_icon="🧩")
st.title("Team 2 — Agentic Research Team")
st.caption("Supervisor → Specialist Agents → Parallel Research")

question = st.text_input(
    "Enter a business research question",
    value="How is NVIDIA positioning itself against AMD in the AI chip market?",
)

if st.button("Research", type="primary") and question.strip():
    initial_state: ResearchState = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
    }

    try:
        with st.spinner("Supervising and running specialist research..."):
            result = research_graph.invoke(initial_state)
    except Exception as error:
        st.error(f"The workflow could not finish: {error}")
    else:
        st.subheader("Supervisor assignments")
        st.json(result["assignments"])

        st.subheader("Independent specialist results")
        for item in sorted(
            result["specialist_results"],
            key=lambda x: x.get("agent", ""),
        ):
            with st.expander(item.get("agent", "Specialist"), expanded=True):
                st.write(f"**Objective:** {item.get('objective', '')}")
                st.write(item.get("research", ""))
                st.caption(f"{len(item.get('sources', []))} Tavily sources returned")

"""
ELP Week 4 — Team 2 Student Extension
Evidence & Traceability + SQLite

PASTE THIS FILE BELOW THE EXISTING Week 3 Team 2 CODE.

Do not replace your Week 3 code. This file adds the Week 4 shared evidence
layer and stores the shared evidence store in a small SQLite database.
"""

# ---------------- Week 4: Shared Evidence Store ----------------

import sqlite3

EVIDENCE_DB = "team2_evidence.db"


def init_evidence_db():
    """Create the Week 4 shared evidence table if it does not exist."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                publication_date TEXT,
                source_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence TEXT,
                related_company TEXT,
                report_section TEXT,
                UNIQUE(claim, url)
            )
        """)
        conn.commit()


def save_evidence_records(records: list[dict]):
    """Save validated specialist evidence to SQLite."""
    # TODO 0 — SAVE TO SQLITE
    #
    # Insert each record into the evidence table using parameterized SQL.
    # Use INSERT OR IGNORE so the same claim + URL is not stored twice.
    columns = [
        "claim",
        "source",
        "url",
        "publication_date",
        "source_type",
        "evidence",
        "confidence",
        "related_company",
        "report_section",
    ]

    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO evidence
                (claim, source, url, publication_date, source_type, evidence,
                 confidence, related_company, report_section)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(record.get(column) for column in columns) for record in records],
        )
        conn.commit()


def load_evidence_records() -> list[dict]:
    """Read the shared evidence store back from SQLite."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM evidence ORDER BY id").fetchall()
        return [dict(row) for row in rows]


init_evidence_db()

# TODO 1 — WEEK 4 STATE
#
# Extend the existing Week 3 ResearchState with:
#     evidence_records: list[dict]
#
# Hint:
#     class Week4ResearchState(ResearchState):
#         ...


class Week4ResearchState(ResearchState):
    evidence_records: list[dict]


EVIDENCE_SYSTEM_PROMPT = """
You are the evidence extraction step in a business research workflow.

Convert useful specialist research sources into structured Evidence Records.

Return ONLY valid JSON in exactly this shape:
{
  "records": [
    {
      "claim": "...",
      "source": "...",
      "url": "...",
      "publication_date": null,
      "source_type": "web",
      "evidence": "...",
      "confidence": "high",
      "related_company": null,
      "report_section": null
    }
  ]
}

Rules:
- claim = a factual statement directly supported by the source.
- evidence = the passage that supports the claim.
- source and url must come from the supplied Tavily result.
- Do not guess a publication date.
- confidence must be high, medium, or low.
- Do not turn analysis or recommendations into facts.
""".strip()


def extract_specialist_evidence(specialist_results: list[dict]) -> list[dict]:
    """Convert specialist source results into the shared Evidence Record format."""
    records = []

    # TODO 2 — EVIDENCE EXTRACTION
    #
    # For each specialist in specialist_results:
    #   1. Read agent and objective.
    #   2. Loop through specialist["sources"].
    #   3. Build source_payload with:
    #        agent, objective, title, url, content, published_date
    #   4. Call call_model_json(EVIDENCE_SYSTEM_PROMPT, prompt).
    #   5. Add every returned record to records.
    #
    # Every specialist must produce the SAME Evidence Record structure.
    for specialist in specialist_results:
        agent = specialist["agent"]
        objective = specialist["objective"]

        for source in specialist["sources"]:
            source_payload = {
                "agent": agent,
                "objective": objective,
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "content": source.get("content", ""),
                "published_date": source.get("published_date"),
            }
            prompt = (
                "Extract Evidence Records from this Tavily result:\n"
                + json.dumps(source_payload, indent=2)
            )

            output = call_model_json(EVIDENCE_SYSTEM_PROMPT, prompt)
            returned = output.get("records", [])
            if isinstance(returned, list):
                records.extend(returned)

    return records


def validate_shared_evidence(records: list[dict]) -> list[dict]:
    """Validate records and remove exact duplicate claim/source pairs."""
    required = ["claim", "source", "url", "publication_date", "source_type", "evidence"]

    validated = []

    # TODO 3 — VALIDATION + DUPLICATES
    #
    # Keep only dictionaries with non-empty:
    #   claim, source, url, source_type, evidence
    #
    # Remove exact duplicates using:
    #   (claim.lower(), url)
    #
    # Hint: use a set called seen.
    optional = ["confidence", "related_company", "report_section"]
    seen = set()

    for record in records:
        if not isinstance(record, dict):
            continue

        # Every stored record gets the same keys, as stripped text or None.
        cleaned = {}
        for field in required + optional:
            value = record.get(field)
            text = "" if value is None else str(value).strip()
            cleaned[field] = text or None

        if any(
            cleaned[field] is None
            for field in required
            if field != "publication_date"
        ):
            continue

        key = (cleaned["claim"].lower(), cleaned["url"])
        if key in seen:
            continue

        seen.add(key)
        validated.append(cleaned)

    return validated


def evidence_store_node(state: ResearchState) -> dict:
    """Week 4: build and persist the shared evidence store."""
    # TODO 4 — CONNECT SPECIALISTS TO SHARED STORE + DATABASE
    #
    # 1. Extract records from state["specialist_results"].
    # 2. Validate them.
    # 3. Save validated records to SQLite.
    # 4. Return {"evidence_records": ...}.
    records = extract_specialist_evidence(state["specialist_results"])
    validated = validate_shared_evidence(records)
    save_evidence_records(validated)
    return {"evidence_records": validated}


def build_week4_graph():
    """
    Extend the existing Week 3 Team 2 flow:

    Question -> Supervisor -> Specialist Agents -> Evidence Records -> SQLite Store
    """
    graph = StateGraph(Week4ResearchState)

    # TODO 5 — GRAPH CONNECTION
    #
    # Reuse:
    #   supervisor_node
    #   specialists_node
    #
    # Add:
    #   evidence_store_node
    #
    # Connect:
    #   START -> supervise -> specialists -> evidence_store -> END
    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", END)

    return graph.compile()


week4_graph = build_week4_graph()


# ---------------- Week 4 UI ----------------

st.divider()
st.subheader("Week 4 — Shared Evidence Store")
st.caption("Specialist research → Evidence Records → SQLite Shared Store")

if st.button("Build Shared Evidence Store", type="primary") and question.strip():
    week4_state = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
    }

    try:
        with st.spinner("Running specialists and building the shared evidence store..."):
            evidence_result = week4_graph.invoke(week4_state)
    except Exception as error:
        st.error(f"The Week 4 evidence workflow could not finish: {error}")
    else:
        st.subheader("New Evidence Records")
        st.json(evidence_result["evidence_records"])

        stored_records = load_evidence_records()
        st.subheader("Shared Evidence Store (SQLite)")
        st.dataframe(stored_records, use_container_width=True)
        st.caption(f"{len(stored_records)} evidence records stored in {EVIDENCE_DB}")



"""
ELP WEEK 5 — TEAM 2 — STUDENT CODE
Multi-agent synthesis and report generation

==================== EXACT SETUP INSTRUCTIONS ====================
1. OPEN the SAME Streamlit Python file used for Weeks 3 and 4, for YOUR team.
2. Make sure it already contains, IN THIS ORDER:
     (a) the completed Week 3 application, imports, tools and graphs;
     (b) the completed Week 4 evidence extension, including its Streamlit UI.
3. Scroll to the VERY BOTTOM of that combined file, below the final Week 4
   st.caption(...) / st.dataframe(...) line. Leave two blank lines.
4. COPY EVERYTHING IN THIS WEEK 5 FILE, including its imports, and PASTE there.
   Alternatively, if you already pasted a previous Week 5 extension, REPLACE
   that old Week 5 extension; do NOT append two Week 5 copies.
5. SAVE the combined file. Run it from a terminal with:
     streamlit run YOUR_EXISTING_APP_FILE.py
6. Enter a research question in the existing question input, then click
   Run Full Synthesis Pipeline in the new Week 5 section near the bottom.
7. Check that the output displays a complete 10-section report AND that SQLite
   contains both evidence and a saved report in the 'reports' table.

IMPORTANT DEPENDENCIES:
- This extension is NOT a standalone Python program. It expects Week 3/4 to
  provide sqlite3, json, st, EVIDENCE_DB, question, call_model_json,
  Week4ResearchState, StateGraph, START, END, and the nodes listed below.
- Team 1 reuses: plan_node, research_node, evidence_node.
- Team 2 reuses: supervisor_node, specialists_node, evidence_store_node.
- Complete ALL Week 4 student TODOs first; unfinished Week 4 code prevents
  Week 5 from obtaining evidence.
- DO NOT mix Team 1 and Team 2 files: their graph state and node names differ.
- This week keeps existing SQLite storage. It creates a SECOND table 'reports'
  inside the SAME database file (not a separate database).
- Existing Week 3 and Week 4 UI buttons are preserved. The Week 5 button is
  the one that runs the complete new pipeline end-to-end.
- You will need your existing LLM and search API configuration to make a live run.

STUDENT WORK: Complete TODO 0 through TODO 5 in order. Functions marked pass are exercises and will not work until completed. Use the instructor file only for checking or teaching solutions.

WEEK 5 TARGET: multi-agent Supervisor → Specialists → Shared Evidence → Synthesis → Report
Deliverable: the first complete DRAFT report; automated citation verification and
quality review are deliberately reserved for Week 6.
==================================================================
"""

# UTC timestamp is used when each report is stored in SQLite.
from datetime import datetime, timezone

# ---------------- Week 5: Save reports in SQLite ----------------

# WHAT THIS DOES — DATABASE SETUP: Creates reports table if missing; preserves all Week 4 evidence rows.
def init_report_db():
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                report_markdown TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


# WHAT THIS DOES — DATABASE WRITE: Saves question, generated markdown, evidence count, timestamp; returns report ID.
def save_report(question: str, report_markdown: str, evidence_count: int) -> int:
    """Save the synthesized report and return its database id."""
    # STUDENT: Replace the `pass` in this function with real code.
    # TODO 0 — REPORT PERSISTENCE
    # Insert question, report_markdown, evidence_count, and a current UTC timestamp
    # into reports using parameterized SQL. Return cursor.lastrowid.
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(EVIDENCE_DB) as conn:
        cursor = conn.execute(
            """
            INSERT INTO reports (question, report_markdown, evidence_count, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (question, report_markdown, evidence_count, created_at),
        )
        conn.commit()
        return cursor.lastrowid


# WHAT THIS DOES — DATABASE READ: Loads previously generated reports, most recent first, for the UI.
def load_reports() -> list[dict]:
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


init_report_db()  # Ensure reports table exists before any report is saved.


# SHARED OUTPUT CONTRACT: Both teams use the SAME nine model-written sections;
# Sources / References is appended in Python as section number ten.
REPORT_SECTION_KEYS = [
    ("executive_summary", "Executive Summary"),
    ("key_findings", "Key Findings"),
    ("competitor_comparison", "Competitor Comparison"),
    ("market_industry_trends", "Market / Industry Trends"),
    ("emerging_technologies", "Emerging Technologies"),
    ("regulatory_developments", "Regulatory Developments"),
    ("risks_opportunities", "Risks / Opportunities"),
    ("recommended_next_steps", "Recommended Next Steps"),
    ("limitations_information_gaps", "Limitations / Information Gaps"),
]


# MODEL INSTRUCTIONS: The model must work from evidence only and return structured JSON.
# Week 6, not this file, handles independent citation validation and report review.
SYNTHESIS_SYSTEM_PROMPT = """
You are the SYNTHESIS AGENT in a multi-agent business research system.

Use ONLY the supplied Shared Evidence Store to create ONE coherent report.
Return ONLY valid JSON with exactly these keys:
{
  "executive_summary": "...",
  "key_findings": "...",
  "competitor_comparison": "...",
  "market_industry_trends": "...",
  "emerging_technologies": "...",
  "regulatory_developments": "...",
  "risks_opportunities": "...",
  "recommended_next_steps": "...",
  "limitations_information_gaps": "..."
}

Rules:
- Cite factual statements using evidence labels such as [E1].
- Reorganize findings by business topic, not agent name.
- The report should read as if one coordinated team produced it.
- Do not invent facts, dates, metrics, source names, URLs, or citations.
- Clearly distinguish facts, analysis, recommendations, and uncertainty.
- If evidence is insufficient, say so.
- Do NOT add the Reviewer/Critic today. That is Week 6.
""".strip()


# WHAT THIS DOES — EVIDENCE PREP: Assigns [E1], [E2] labels to this run’s evidence, keeps sources and excerpts.
def build_evidence_packet(evidence_records: list[dict]) -> list[dict]:
    packet = []
    for index, record in enumerate(evidence_records, start=1):
        packet.append({
            "evidence_id": f"E{index}",
            "claim": record.get("claim"),
            "source": record.get("source"),
            "url": record.get("url"),
            "publication_date": record.get("publication_date"),
            "source_type": record.get("source_type"),
            "evidence": record.get("evidence"),
            "confidence": record.get("confidence"),
            "related_company": record.get("related_company"),
            "report_section": record.get("report_section"),
        })
    return packet


# WHAT THIS DOES — TEAM 2 SYNTHESIS AGENT: Combines all specialist evidence into ONE unified report.
def synthesize_report(question: str, evidence_records: list[dict]) -> dict:
    """Merge all specialist evidence into the common report structure."""
    # STUDENT: Replace the `pass` below with the report-generation implementation.
    # TODO 1 — SYNTHESIS AGENT
    # 1. Build the numbered evidence packet.
    # 2. Put the research question + shared evidence into the prompt.
    # 3. Call call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt).
    # 4. Make sure all REPORT_SECTION_KEYS exist.
    # 5. Return the report dict.
    packet = build_evidence_packet(evidence_records)

    prompt = (
        f"Research question: {question}\n\n"
        "Shared Evidence Store:\n"
        + json.dumps(packet, indent=2)
    )

    output = call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt)

    report = {}
    for key, _title in REPORT_SECTION_KEYS:
        value = output.get(key)
        if isinstance(value, (list, dict)):
            value = json.dumps(value, indent=2)
        text = "" if value is None else str(value).strip()
        report[key] = text or "Insufficient evidence to complete this section."

    return report


# WHAT THIS DOES — REFERENCES: Creates citation legend from original URLs, not invented model references.
def build_references(evidence_records: list[dict]) -> str:
    lines = []
    for index, record in enumerate(evidence_records, start=1):
        source = record.get("source") or "Unknown source"
        url = record.get("url") or "No URL"
        date = record.get("publication_date") or "Date unavailable"
        lines.append(f"- [E{index}] {source} — {date} — {url}")
    return "\n".join(lines) if lines else "No source records were available."


# WHAT THIS DOES — REPORT FORMAT: Assembles nine model sections and the tenth Sources / References section.
def render_report_markdown(report: dict, evidence_records: list[dict]) -> str:
    blocks = []
    for key, title in REPORT_SECTION_KEYS:
        blocks.append(f"## {title}\n\n{report.get(key, '')}")
    blocks.append("## Sources / References\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)


# TODO 2 — WEEK 5 STATE
# Extend Week4ResearchState with:
#     report: dict
#     report_markdown: str
#     report_id: int


class Week5ResearchState(Week4ResearchState):
    report: dict
    report_markdown: str
    report_id: int


# WHAT THIS DOES — TEAM 2 GRAPH NODE: Runs synthesis, formats it, saves to SQLite.
def synthesis_node(state):
    """Synthesize only after all shared evidence has been produced."""
    # TODO 3 — CONNECT SHARED EVIDENCE TO SYNTHESIS
    # 1. Read state["evidence_records"].
    # 2. Stop with a helpful error if empty.
    # 3. Call synthesize_report(...).
    # 4. Render markdown.
    # 5. Save the report to SQLite.
    # 6. Return report, report_markdown, and report_id.
    evidence_records = state["evidence_records"]
    if not evidence_records:
        raise ValueError(
            "No evidence records were produced, so there is nothing to synthesize. "
            "Try a more specific research question."
        )

    report = synthesize_report(state["question"], evidence_records)
    report_markdown = render_report_markdown(report, evidence_records)
    report_id = save_report(state["question"], report_markdown, len(evidence_records))

    return {
        "report": report,
        "report_markdown": report_markdown,
        "report_id": report_id,
    }


# WHAT THIS DOES — WORKFLOW WIRING: Reuses Week 3/4 nodes and adds Week 5 report/synthesis at the end.
def build_week5_graph():
    """Question -> Supervisor -> Specialists -> Shared Evidence -> Synthesis -> END"""
    # TODO 4 — EXTEND THE MULTI-AGENT GRAPH
    # Reuse supervisor_node, specialists_node, and evidence_store_node.
    # Add synthesis_node after the shared evidence store.
    # Use Week5ResearchState.
    graph = StateGraph(Week5ResearchState)

    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


week5_graph = build_week5_graph()


# ---------------- Week 5 UI ----------------
# UI: Adds a separate Week 5 button and shows the report + evidence + saved history.
# A full run invokes Week 3 planning/research, Week 4 evidence, THEN Week 5 writing.

st.divider()
st.subheader("Week 5 — Multi-Agent Synthesis Pipeline")
st.caption("Question → Supervisor → Specialists → Shared Evidence → Synthesis Agent → Report")

if st.button("Run Full Synthesis Pipeline", type="primary") and question.strip():
    # TODO 5 — RUN THE COMPLETE PIPELINE
    # Create the initial state, invoke week5_graph, and display:
    #   - the synthesized report with st.markdown(...)
    #   - the shared evidence used
    #   - saved reports from SQLite
    week5_state = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
        "report": {},
        "report_markdown": "",
        "report_id": 0,
    }

    try:
        with st.spinner("Running specialists, building evidence, and synthesizing the report..."):
            week5_result = week5_graph.invoke(week5_state)
    except Exception as error:
        st.error(f"The Week 5 synthesis workflow could not finish: {error}")
    else:
        st.subheader("Synthesized Report")
        st.markdown(week5_result["report_markdown"])
        st.caption(f"Saved to SQLite as report #{week5_result['report_id']}")

        st.subheader("Shared Evidence Used")
        st.dataframe(
            build_evidence_packet(week5_result["evidence_records"]),
            use_container_width=True,
        )
        st.caption(f"{len(week5_result['evidence_records'])} evidence records used")

        saved_reports = load_reports()
        st.subheader("Saved Reports (SQLite)")
        for saved in saved_reports:
            label = f"#{saved['id']} — {saved['question']} ({saved['created_at']})"
            with st.expander(label):
                st.caption(f"{saved['evidence_count']} evidence records")
                st.markdown(saved["report_markdown"])
        st.caption(f"{len(saved_reports)} reports stored in {EVIDENCE_DB}")


"""
ELP WEEK 6 — TEAM 2 — STUDENT EXERCISE
Multi-agent reviewer / quality control

==================== EXACT SETUP INSTRUCTIONS ====================
1. OPEN the existing, combined TEAM 2 Streamlit Python app that contains your
   completed Week 3, Week 4, and Week 5 work IN THAT ORDER.
2. CONFIRM the Week 5 button runs and produces a 10-section draft report.
   Complete Week 5 TODOs before moving on; Week 6 cannot fix missing Week 5 logic.
3. SCROLL TO THE VERY END of that SAME .py app file, AFTER the last line of Week 5
   (including the Week 5 Streamlit UI). Leave two blank lines.
4. COPY THIS ENTIRE Week 6 extension (including imports, functions, graph, UI)
   and PASTE IT BELOW Week 5. Do not paste into a new standalone .py script.
5. DO NOT DELETE or replace Week 3/4/5. DO NOT mix Team 1 and Team 2 code.
6. If you previously pasted a Week 6 extension, REPLACE that Week 6 block;
   do NOT append duplicate Week 6 copies.
7. SAVE your combined app, then run from a terminal in its directory:
       streamlit run YOUR_EXISTING_APP_FILE.py
8. Enter your research question in the EXISTING question input. Click:
       Run Reviewed Synthesis Pipeline
9. Inspect the quality result. PASS permits final display; failures must be
   clearly flagged, with the draft accessible only for inspection.

IMPORTANT:
- Week 6 is NOT standalone; it calls existing Week 5 functions, graph nodes,
  shared state, report/evidence helpers, model client, SQLite, and Streamlit.
- You need the SAME working API keys and Python dependencies as Week 5.
- Week 5 may still save a DRAFT to SQLite before Week 6 checks it;
  that saved draft is NOT automatically an approved final report.
- Use the new Week 6 button, not the older Week 5 button, for the full flow.
- This is a live end-to-end prototype; it is not guaranteed to identify every
  unsupported statement or broken URL. Inspect the verdict yourself.
- NO follow-up Q&A or automatic extra research until Week 7.
- Team 2: reviewer is an agent after synthesis; at most ONE revision.

STUDENT: Complete all numbered TODOs before the NEXT meeting, at your own pace.
Existing `pass` placeholders are exercises and the extension WILL NOT work until finished.
END-OF-WEEK GOAL: The report is checked before being presented as trusted.
===================================================================
"""


REVIEW_SYSTEM_PROMPT = """
You are the REVIEWER / CRITIC AGENT in a multi-agent business research system.
Evaluate the draft against the research question and Shared Evidence Store.
Do not conduct new research and do not silently rewrite the report.

Return ONLY valid JSON:
{
  "overall_status": "PASS" or "NEEDS_REVISION",
  "scores": {
    "completeness": 0,
    "citation_support": 0,
    "source_quality": 0,
    "consistency": 0,
    "business_usefulness": 0,
    "clarity": 0
  },
  "issues": [
    {
      "category": "completeness | citation_support | source_quality | consistency | business_usefulness | clarity",
      "section": "report section or overall",
      "explanation": "specific problem",
      "suggested_fix": "specific bounded fix"
    }
  ],
  "research_gaps": ["gap that may require more research later"]
}

PASS only when all six scores are at least 7 and there are no material citation-support issues.
""".strip()


REVISION_SYSTEM_PROMPT = """
You are the SYNTHESIS AGENT performing ONE controlled revision after reviewer feedback.
Use ONLY the supplied shared evidence. Do not research new facts.
Keep the same nine-section JSON report contract used in Week 5.
If a true research gap cannot be fixed from existing evidence, acknowledge it in limitations.
""".strip()


# TODO 0 — REVIEWER AGENT
# Give it: question + draft report + build_evidence_packet(evidence_records).
# Call call_model_json(REVIEW_SYSTEM_PROMPT, prompt).
# Normalize six scores to 0-10 integers and return:
# overall_status, scores, issues, research_gaps.
# Treat invalid model JSON shapes as a failed review, not a PASS.
REVIEW_SCORE_KEYS = [
    "completeness",
    "citation_support",
    "source_quality",
    "consistency",
    "business_usefulness",
    "clarity",
]


def review_report(question: str, report: dict, evidence_records: list[dict]) -> dict:
    """Score the draft report against the shared evidence; never rewrite it."""
    packet = build_evidence_packet(evidence_records)

    prompt = (
        f"Research question: {question}\n\n"
        "Draft report:\n"
        + json.dumps(report, indent=2)
        + "\n\nShared Evidence Store:\n"
        + json.dumps(packet, indent=2)
    )

    try:
        output = call_model_json(REVIEW_SYSTEM_PROMPT, prompt)
    except json.JSONDecodeError:
        output = {}
    if not isinstance(output, dict):
        output = {}

    status = str(output.get("overall_status", "")).strip().upper()
    raw_scores = output.get("scores")
    issues = output.get("issues")
    research_gaps = output.get("research_gaps")

    valid_shape = (
        status in {"PASS", "NEEDS_REVISION"}
        and isinstance(raw_scores, dict)
        and isinstance(issues, list)
        and isinstance(research_gaps, list)
    )

    scores = {}
    for key in REVIEW_SCORE_KEYS:
        value = raw_scores.get(key) if isinstance(raw_scores, dict) else None
        try:
            score = round(float(value))
        except (TypeError, ValueError):
            score = 0
            valid_shape = False
        scores[key] = max(0, min(10, score))

    issues = [issue for issue in issues if isinstance(issue, dict)] if isinstance(issues, list) else []
    research_gaps = [str(gap) for gap in research_gaps] if isinstance(research_gaps, list) else []

    if not valid_shape:
        status = "NEEDS_REVISION"
        issues.append({
            "category": "review_error",
            "section": "overall",
            "explanation": "The reviewer returned an invalid response, so the report could not be approved.",
            "suggested_fix": "Run the review again.",
        })
    elif status == "PASS" and min(scores.values()) < 7:
        # The prompt's PASS rule, enforced in code rather than trusted to the model.
        status = "NEEDS_REVISION"

    return {
        "overall_status": status,
        "scores": scores,
        "issues": issues,
        "research_gaps": research_gaps,
    }


# TODO 1 — ONE CONTROLLED REVISION
# Give the synthesis model the current report, reviewer feedback, and existing evidence.
# It may revise wording/organization using existing evidence only.
# It may NOT search or invent facts.
# Ensure every REPORT_SECTION_KEYS key still exists.
def revise_report_once(question: str, report: dict, evidence_records: list[dict], review: dict) -> dict:
    """Apply the reviewer's feedback once, using only the existing evidence."""
    packet = build_evidence_packet(evidence_records)
    section_keys = [key for key, _title in REPORT_SECTION_KEYS]

    prompt = (
        f"Research question: {question}\n\n"
        "Current draft report:\n"
        + json.dumps(report, indent=2)
        + "\n\nReviewer feedback:\n"
        + json.dumps(review, indent=2)
        + "\n\nShared Evidence Store:\n"
        + json.dumps(packet, indent=2)
        + "\n\nReturn the revised report as a JSON object with exactly these keys: "
        + ", ".join(section_keys)
    )

    try:
        output = call_model_json(REVISION_SYSTEM_PROMPT, prompt)
    except json.JSONDecodeError:
        output = {}
    if not isinstance(output, dict):
        output = {}

    revised = {}
    for key in section_keys:
        value = output.get(key)
        if isinstance(value, (list, dict)):
            value = json.dumps(value, indent=2)
        text = "" if value is None else str(value).strip()
        # A section the revision dropped keeps its draft text rather than vanishing.
        revised[key] = text or report.get(key, "Insufficient evidence to complete this section.")

    return revised


# TODO 2 — WEEK 6 STATE
# Extend Week5ResearchState with:
#   review: dict
#   initial_review: dict
#   final_report_markdown: str
#   revision_performed: bool


class Week6ResearchState(Week5ResearchState):
    review: dict
    initial_review: dict
    final_report_markdown: str
    revision_performed: bool


# TODO 3 — REVIEW GATE NODE
# 1. Review the Week 5 synthesized report.
# 2. If PASS, keep it.
# 3. If NEEDS_REVISION, revise ONCE using existing evidence.
# 4. Review the revised report exactly once more.
# 5. Render final markdown using render_report_markdown(...).
def reviewer_node(state):
    """Review the draft, revise it at most once, and review the revision."""
    question = state["question"]
    report = state["report"]
    evidence_records = state["evidence_records"]

    review = review_report(question, report, evidence_records)
    initial_review = review
    revision_performed = False

    if review["overall_status"] == "NEEDS_REVISION":
        report = revise_report_once(question, report, evidence_records, review)
        review = review_report(question, report, evidence_records)
        revision_performed = True

    return {
        "report": report,
        "review": review,
        "initial_review": initial_review,
        "final_report_markdown": render_report_markdown(report, evidence_records),
        "revision_performed": revision_performed,
    }


# TODO 4 — EXTEND THE MULTI-AGENT GRAPH
# Reuse supervisor_node -> specialists_node -> evidence_store_node -> synthesis_node.
# Add reviewer_node after synthesis.
def build_week6_graph():
    """Question -> Supervisor -> Specialists -> Shared Evidence -> Synthesis -> Reviewer -> END"""
    graph = StateGraph(Week6ResearchState)

    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", "synthesis")
    graph.add_edge("synthesis", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


week6_graph = build_week6_graph()


# TODO 5 — WEEK 6 UI
# Add a "Run Reviewed Synthesis Pipeline" button.
# Show:
# - final report ONLY if overall_status == PASS
# - if review fails, label the report unapproved and put it in an expander
# - reviewer scorecard
# - issues/research gaps
# - initial review if a revision happened
# - shared evidence

# WHAT THIS DOES — SCORECARD: Shows the six reviewer scores as a one-row table.
def show_review_scorecard(review: dict):
    st.dataframe([review["scores"]], use_container_width=True)


st.divider()
st.subheader("Week 6 — Reviewed Synthesis Pipeline")
st.caption("Question → Supervisor → Specialists → Shared Evidence → Synthesis → Reviewer → Report")

if st.button("Run Reviewed Synthesis Pipeline", type="primary") and question.strip():
    week6_state = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
        "report": {},
        "report_markdown": "",
        "report_id": 0,
        "review": {},
        "initial_review": {},
        "final_report_markdown": "",
        "revision_performed": False,
    }

    try:
        with st.spinner("Researching, synthesizing, and reviewing the report..."):
            week6_result = week6_graph.invoke(week6_state)
    except Exception as error:
        st.error(f"The Week 6 reviewed workflow could not finish: {error}")
    else:
        review = week6_result["review"]

        if review["overall_status"] == "PASS":
            st.success("Reviewer verdict: PASS")
            st.subheader("Final Report")
            st.markdown(week6_result["final_report_markdown"])
        else:
            st.error(
                "Reviewer verdict: NEEDS_REVISION. This report is NOT approved "
                "and should not be treated as trusted."
            )
            with st.expander("Unapproved report (for inspection only)"):
                st.markdown(week6_result["final_report_markdown"])

        if week6_result["revision_performed"]:
            st.caption("One revision was performed after the initial review.")

        st.subheader("Reviewer Scorecard")
        show_review_scorecard(review)

        st.subheader("Issues")
        if review["issues"]:
            st.dataframe(review["issues"], use_container_width=True)
        else:
            st.write("No issues reported.")

        st.subheader("Research Gaps")
        if review["research_gaps"]:
            for gap in review["research_gaps"]:
                st.markdown(f"- {gap}")
        else:
            st.write("No research gaps reported.")

        if week6_result["revision_performed"]:
            initial_review = week6_result["initial_review"]
            with st.expander(f"Initial review (before revision): {initial_review['overall_status']}"):
                show_review_scorecard(initial_review)
                if initial_review["issues"]:
                    st.dataframe(initial_review["issues"], use_container_width=True)
                for gap in initial_review["research_gaps"]:
                    st.markdown(f"- {gap}")

        st.subheader("Shared Evidence Used")
        st.dataframe(
            build_evidence_packet(week6_result["evidence_records"]),
            use_container_width=True,
        )
        st.caption(
            f"{len(week6_result['evidence_records'])} evidence records used. "
            f"The Week 5 draft was saved to SQLite as report #{week6_result['report_id']}; "
            "that saved draft is not an approved report."
        )

"""
ELP WEEK 7 — TEAM 2 — STUDENT EXERCISE
Interactive Research: gap/conflict detection + bounded additional research

EXACT SETUP:
1. Open the EXISTING, combined TEAM 2 Streamlit .py app (Weeks 3–6, in order).
2. Finish Week 6 first; confirm 'Run Reviewed Synthesis Pipeline' works.
3. Paste this ENTIRE Week 7 block at the END of that SAME file, after Week 6 UI.
4. Do not create a standalone app. Do not delete, duplicate, or mix team blocks.
5. Run: streamlit run YOUR_EXISTING_APP_FILE.py
6. Click 'Run Week 7 Research Loop'.

WEEK 7 SCOPE FOR TEAM 2: detect material gaps and conflicts, assign one bounded
additional research pass to existing specialists, merge evidence, then synthesize
and review. DO NOT add a new agent or an unbounded retry loop.
STUDENT: Complete all TODOs before the next meeting.
"""

import json

W7_T2_GAP_PROMPT = """
You are the existing supervisor's GAP AND CONFLICT CHECK, not a new agent.
Given the user's research question and evidence store, identify at most ONE
material research gap or meaningful contradiction that targeted research could fix.
Return JSON ONLY:
  {"needs_research": boolean, "reason": string,
   "targeted_question": string, "conflicts": [string], "gaps": [string]}
Missing evidence and genuinely conflicting facts matter; duplicate coverage
alone does not require searching. Do not invent missing facts or resolve a
conflict by guessing. Recommend NO further research when already adequate.
""".strip()

# TODO 0 — GAP / CONFLICT DETECTION
# Pass the original question and build_evidence_packet(state["evidence_records"])
# to call_model_json(W7_T2_GAP_PROMPT, ...).
# Return {"gap_check": {needs_research: bool, reason: str,
#          targeted_question: str, conflicts: list, gaps: list},
#         "additional_research_performed": False}.
# Invalid model output must NOT accidentally trigger research.
def week7_t2_find_gaps(state: dict) -> dict:
    """Ask the supervisor whether ONE targeted search would fix a gap or conflict."""
    packet = build_evidence_packet(state["evidence_records"])

    prompt = (
        f"Research question: {state['question']}\n\n"
        "Shared Evidence Store:\n"
        + json.dumps(packet, indent=2)
    )

    try:
        output = call_model_json(W7_T2_GAP_PROMPT, prompt)
    except json.JSONDecodeError:
        output = {}
    if not isinstance(output, dict):
        output = {}

    reason = output.get("reason")
    targeted_question = output.get("targeted_question")
    conflicts = output.get("conflicts")
    gaps = output.get("gaps")

    # Only a literal True plus a usable question counts; anything else means no search.
    needs_research = (
        output.get("needs_research") is True
        and isinstance(targeted_question, str)
        and bool(targeted_question.strip())
    )

    return {
        "gap_check": {
            "needs_research": needs_research,
            "reason": str(reason).strip() if reason else "No reason given.",
            "targeted_question": targeted_question.strip() if needs_research else "",
            "conflicts": [str(item) for item in conflicts] if isinstance(conflicts, list) else [],
            "gaps": [str(item) for item in gaps] if isinstance(gaps, list) else [],
        },
        "additional_research_performed": False,
    }

# TODO 1 — CONDITIONAL ROUTING
# Return 'additional' only if a valid, material gap/conflict warrants searching;
# otherwise return 'synthesis'.
def week7_t2_route(state: dict) -> str:
    gap_check = state.get("gap_check") or {}
    if (
        gap_check.get("needs_research") is True
        and gap_check.get("targeted_question")
        and not state.get("additional_research_performed")
    ):
        return "additional"
    return "synthesis"

# TODO 2 — ONE BOUNDED EXTRA RESEARCH PASS
# Use the original existing supervisor_node, specialists_node,
# evidence_store_node on the gap check's targeted_question.
# Merge new records with previous evidence; preserve old evidence.
# Do not recursively call week7_graph; no infinite loops.
# Return evidence_records, specialist_results, additional_research_performed,
# new_evidence_count.
def week7_t2_additional_research(state: dict) -> dict:
    """Run the existing supervisor and specialists once on the targeted question."""
    targeted_question = state["gap_check"]["targeted_question"]
    previous_records = state["evidence_records"]

    # A separate mini-state, so the original question still drives synthesis.
    extra_state = {
        "question": targeted_question,
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
    }
    extra_state.update(supervisor_node(extra_state))
    extra_state.update(specialists_node(extra_state))
    extra_state.update(evidence_store_node(extra_state))

    # Old records come first, so validation keeps them and drops only repeat claims.
    merged = validate_shared_evidence(previous_records + extra_state["evidence_records"])

    return {
        "evidence_records": merged,
        "specialist_results": state["specialist_results"] + extra_state["specialist_results"],
        "additional_research_performed": True,
        "new_evidence_count": len(merged) - len(previous_records),
    }

# TODO 3 — STATE
# Extend Week6ResearchState, adding:
# gap_check: dict
# additional_research_performed: bool
# new_evidence_count: int
class Week7ResearchState(Week6ResearchState):
    gap_check: dict
    additional_research_performed: bool
    new_evidence_count: int


# TODO 4 — WEEK 7 GRAPH
# START -> supervise -> specialists -> evidence_store -> gap_check
# gap_check -> synthesis (no missing evidence)
# gap_check -> additional -> synthesis (at most one extra pass)
# synthesis -> review -> END
# Reuse EXISTING Team 2 nodes. Do not invent agents.
def build_week7_graph():
    graph = StateGraph(Week7ResearchState)

    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)
    graph.add_node("gap_check", week7_t2_find_gaps)
    graph.add_node("additional", week7_t2_additional_research)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("review", reviewer_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", "gap_check")
    graph.add_conditional_edges(
        "gap_check",
        week7_t2_route,
        {"additional": "additional", "synthesis": "synthesis"},
    )
    # The extra pass always goes straight on to synthesis, never back to gap_check.
    graph.add_edge("additional", "synthesis")
    graph.add_edge("synthesis", "review")
    graph.add_edge("review", END)

    return graph.compile()

# TODO 5 — ASSIGN THE COMPILED GRAPH (after TODO 3 and TODO 4 work).
week7_graph = build_week7_graph()

# TODO 6 — STREAMLIT UI
# Add 'Run Week 7 Research Loop'.
# Run the graph with the same Week 6 initial state plus Week 7 fields.
# Show gap/conflict notes and whether additional research occurred.
# Present final report ONLY if reviewer status == PASS.
# Otherwise place draft in expander with unapproved label.
# Show reviewer details + evidence for debugging.
st.divider()
st.subheader("Week 7 — Research Loop with Gap Check")
st.caption(
    "Question → Supervisor → Specialists → Shared Evidence → Gap Check "
    "→ (one extra pass) → Synthesis → Reviewer → Report"
)

if st.button("Run Week 7 Research Loop", type="primary") and question.strip():
    week7_state = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
        "report": {},
        "report_markdown": "",
        "report_id": 0,
        "review": {},
        "initial_review": {},
        "final_report_markdown": "",
        "revision_performed": False,
        "gap_check": {},
        "additional_research_performed": False,
        "new_evidence_count": 0,
    }

    try:
        with st.spinner("Researching, checking for gaps, synthesizing, and reviewing..."):
            week7_result = week7_graph.invoke(week7_state)
    except Exception as error:
        st.error(f"The Week 7 research loop could not finish: {error}")
    else:
        gap_check = week7_result["gap_check"]
        st.subheader("Gap / Conflict Check")
        st.write(f"**Reason:** {gap_check['reason']}")
        if gap_check["gaps"]:
            st.write("**Gaps:**")
            for gap in gap_check["gaps"]:
                st.markdown(f"- {gap}")
        if gap_check["conflicts"]:
            st.write("**Conflicts:**")
            for conflict in gap_check["conflicts"]:
                st.markdown(f"- {conflict}")

        if week7_result["additional_research_performed"]:
            st.info(
                f"One additional research pass ran on: \"{gap_check['targeted_question']}\" "
                f"({week7_result['new_evidence_count']} new evidence records)."
            )
        else:
            st.info("No additional research was needed.")

        review = week7_result["review"]
        if review["overall_status"] == "PASS":
            st.success("Reviewer verdict: PASS")
            st.subheader("Final Report")
            st.markdown(week7_result["final_report_markdown"])
        else:
            st.error(
                "Reviewer verdict: NEEDS_REVISION. This report is NOT approved "
                "and should not be treated as trusted."
            )
            with st.expander("Unapproved report (for inspection only)"):
                st.markdown(week7_result["final_report_markdown"])

        with st.expander("Reviewer details"):
            show_review_scorecard(review)
            if review["issues"]:
                st.dataframe(review["issues"], use_container_width=True)
            for gap in review["research_gaps"]:
                st.markdown(f"- {gap}")
            if week7_result["revision_performed"]:
                st.caption(
                    "One revision was performed; initial verdict was "
                    f"{week7_result['initial_review']['overall_status']}."
                )

        with st.expander("Shared evidence used"):
            st.dataframe(
                build_evidence_packet(week7_result["evidence_records"]),
                use_container_width=True,
            )
            st.caption(f"{len(week7_result['evidence_records'])} evidence records used")







"""STUDENT VERSION — complete the marked TODOs with your team.
Paste ONCE after completed Week 7 code in your existing Team 2 Streamlit app.
This is NOT a standalone script. The approval gate deliberately fails closed
until you complete TODO 1. Ask instructors to help with failing earlier weeks.
"""
"""
ELP WEEK 8 — TEAM 2 (MULTI-AGENT) — INTEGRATION EXTENSION

WHERE TO PASTE
1. Open YOUR team's combined Streamlit app with completed Weeks 3–7.
2. Finish ALL prior TODOs, back up/commit the file, then paste this file ONCE
   at the very bottom AFTER the Week 7 UI block.
3. Do NOT run standalone; do NOT mix Team 1 and Team 2 modules.
4. Launch: streamlit run YOUR_COMBINED_APP.py
5. Use the new Week 8 final run button. Earlier-week buttons are historical;
   hide/remove obsolete sections after confirming this new workspace works.

REQUIRES PREVIOUS CODE TO DEFINE: st, week7_graph, question.
Uses existing Supervisor -> 3 specialists -> shared evidence -> gap check ->
optional ONE extra pass -> synthesis -> Reviewer. No new agents or loops.

STUDENT TODOs
[ ] Verify all three specialist agents execute and their labels are visible.
[ ] Confirm gap check triggers <= ONE extra pass and preserves evidence.
[ ] Confirm the Reviewer quality gate prevents approved download on failure.
[ ] Check sources manually and that all 10 report headings are present.
[ ] Run three repeated trials with the same prompt and note errors/timing.
[ ] Consolidate obsolete prior-week UI after this works.
[ ] OPTIONAL separate scope decision: add user follow-up Q&A later only if
    mentor/program leads explicitly prioritize it over stabilization.
"""
import time as _w8_time

_w8_missing = [name for name in ("st", "week7_graph", "question") if name not in globals()]

# TODO 1 — Implement a conservative quality gate.
# Requirements: result is dict; review.overall_status == PASS;
# final_report_markdown nonempty; >=1 evidence record; >=3 specialist outputs.
# FAIL CLOSED while TODO is unfinished (never mark a draft approved).
def _w8_t2_ready(result):
    if not isinstance(result, dict):
        return False
    review = result.get("review")
    if not isinstance(review, dict) or review.get("overall_status") != "PASS":
        return False
    report_text = result.get("final_report_markdown")
    if not isinstance(report_text, str) or not report_text.strip():
        return False
    if len(result.get("evidence_records") or []) < 1:
        return False
    if len(result.get("specialist_results") or []) < 3:
        return False
    return True


st.divider()
st.header("Week 8 — Final Research Workspace | Team 2")
st.caption("Supervisor · parallel specialists · shared evidence · gap check · synthesis · Reviewer")
if _w8_missing:
    st.error("Missing previous definitions: " + ", ".join(_w8_missing))
else:
    st.info("Week 8 integrates and stabilizes the existing work. Do not add new agents or unbounded research loops.")
    if st.button("Run final Team 2 research", key="w8_t2_run", type="primary"):
        if not question.strip():
            st.warning("Enter a research question in the original input first.")
        else:
            _w8_initial = {"question": question.strip(), "assignments": [],
                           "specialist_results": [], "evidence_records": [],
                           "report": {}, "report_markdown": "", "report_id": 0,
                           "review": {}, "initial_review": {}, "final_report_markdown": "",
                           "revision_performed": False, "gap_check": {},
                           "additional_research_performed": False, "new_evidence_count": 0}
            _w8_start = _w8_time.perf_counter()
            try:
                with st.spinner("Coordinating specialists, gathering evidence and reviewing report..."):
                    st.session_state["w8_t2_result"] = week7_graph.invoke(_w8_initial)
                st.session_state["w8_t2_seconds"] = round(_w8_time.perf_counter() - _w8_start, 1)
            except Exception as exc:
                st.session_state.pop("w8_t2_result", None)
                st.error(f"Research failed; no approved report was produced: {exc}")
    if "w8_t2_result" in st.session_state:
        _w8_result = st.session_state["w8_t2_result"]
        _w8_ok = _w8_t2_ready(_w8_result)
        _w8_evidence = _w8_result.get("evidence_records", []) or []
        _w8_specialists = _w8_result.get("specialist_results", []) or []
        _w8_a, _w8_b, _w8_c = st.columns(3)
        _w8_a.metric("Duration (seconds)", st.session_state.get("w8_t2_seconds", "—"))
        _w8_b.metric("Evidence records", len(_w8_evidence))
        _w8_c.metric("Specialist outputs", len(_w8_specialists))
        if _w8_ok:
            st.success("Reviewer PASS — approved for prototype demo (not independently fact-checked).")
            _w8_text = _w8_result.get("final_report_markdown", "")
            st.markdown(_w8_text)
            st.download_button("Download reviewed report (.md)", _w8_text,
                               file_name="elp_team2_report.md", mime="text/markdown", key="w8_t2_download")
        else:
            st.error("Reviewer did not approve this report; no approved download is offered.")
            with st.expander("Unapproved draft — internal debugging only"):
                st.markdown(_w8_result.get("final_report_markdown", "") or _w8_result.get("report_markdown", ""))
        with st.expander("Supervisor assignments / specialist contributions"):
            st.json(_w8_result.get("assignments", []))
            st.json(_w8_specialists)
        with st.expander("Gap/conflict check + bounded extra research"):
            st.json(_w8_result.get("gap_check", {}))
            st.write("Extra pass occurred:", bool(_w8_result.get("additional_research_performed")))
            st.write("New records:", _w8_result.get("new_evidence_count", 0))
        with st.expander("Reviewer scorecard and evidence"):
            st.json(_w8_result.get("review", {}))
            st.dataframe(_w8_evidence, use_container_width=True)
        # TODO 2 — Inspect each of the three specialist roles and verify
        # the output is distinct. Check that the gap pass occurs at most once.
        # TODO 3 — Manually open 3 cited source links and note problems.
