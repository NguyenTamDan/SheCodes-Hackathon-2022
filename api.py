"""Simple Developer Productivity API (one file).

Run:   pip install fastapi uvicorn pandas scikit-learn
       uvicorn main:app --reload
Docs:  http://127.0.0.1:8000/docs   (you can try every endpoint there)

Flow:  platform --POST data--> this API --> database (SQLite file)
       platform <--GET answers-- this API <-- database
"""
import os
import sqlite3
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_FILE = os.getenv("DB_FILE", "productivity.db")   # the database is just this file
SPRINT_DAYS = 10                                    # a ticket longer than this missed the sprint

app = FastAPI(title="Developer Productivity API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # lets a web page call us


# ---------------------------------------------------------------- database
def db():
    return sqlite3.connect(DB_FILE)


def create_tables():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY, sprint INTEGER, team TEXT, assignee TEXT, issue_type TEXT,
            points INTEGER, has_dependency INTEGER, assignee_tenure REAL,
            blocked_days REAL, qa_wait_days REAL, dev_days REAL, cycle_days REAL, missed_sprint INTEGER)""")
        con.execute("""CREATE TABLE IF NOT EXISTS people_sprint (
            sprint INTEGER, person TEXT, team TEXT, role TEXT, tenure_months REAL,
            core_work REAL, design_meetings REAL, cross_team_dependencies REAL, review_mentoring REAL, other REAL,
            points_done REAL, PRIMARY KEY (sprint, person))""")


create_tables()


def table(name: str) -> pd.DataFrame:
    with db() as con:
        return pd.read_sql_query(f"SELECT * FROM {name}", con)


# ---------------------------------------------------------------- what the platform sends us
class Ticket(BaseModel):
    ticket_id: str
    sprint: int = Field(ge=1)
    team: str
    assignee: str
    issue_type: Literal["Story", "Bug", "Task"]
    points: int = Field(ge=1, le=21)
    has_dependency: bool
    assignee_tenure: float = Field(ge=0)                 # months of experience
    # fill these in when the ticket is finished (leave empty while it is open)
    blocked_days: float | None = None
    qa_wait_days: float | None = None
    dev_days: float | None = None


class PersonSprint(BaseModel):
    sprint: int = Field(ge=1)
    person: str
    team: str
    role: Literal["Developer", "Senior Developer", "QA"]
    tenure_months: float = Field(ge=0)
    core_work: float = Field(ge=0)                       # hours
    design_meetings: float = Field(ge=0)
    cross_team_dependencies: float = Field(ge=0)
    review_mentoring: float = Field(ge=0)
    other: float = Field(ge=0)
    points_done: float | None = None                     # leave empty for QA


class NewTicket(BaseModel):
    team: str
    issue_type: Literal["Story", "Bug", "Task"]
    points: int = Field(ge=1, le=21)
    has_dependency: bool
    assignee_tenure: float = Field(ge=0)


# ---------------------------------------------------------------- platform -> database
@app.post("/tickets")
def save_tickets(tickets: list[Ticket]):
    """Save tickets. Sending the same ticket_id again updates it."""
    with db() as con:
        for t in tickets:
            parts = [t.blocked_days, t.qa_wait_days, t.dev_days]
            cycle = sum(parts) if None not in parts else None          # total days, only when finished
            missed = None if cycle is None else int(cycle > SPRINT_DAYS)
            con.execute("INSERT OR REPLACE INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (t.ticket_id, t.sprint, t.team, t.assignee, t.issue_type, t.points, int(t.has_dependency),
                         t.assignee_tenure, t.blocked_days, t.qa_wait_days, t.dev_days, cycle, missed))
    return {"saved": len(tickets)}


@app.post("/people-sprint")
def save_people_sprint(rows: list[PersonSprint]):
    """Save hours and points per person per sprint. Same sprint + person updates the row."""
    with db() as con:
        for r in rows:
            con.execute("INSERT OR REPLACE INTO people_sprint VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (r.sprint, r.person, r.team, r.role, r.tenure_months, r.core_work, r.design_meetings,
                         r.cross_team_dependencies, r.review_mentoring, r.other, r.points_done))
    return {"saved": len(rows)}


# ---------------------------------------------------------------- database -> platform
@app.get("/health")
def health():
    return {"status": "ok", "tickets": len(table("tickets")), "people_sprint_rows": len(table("people_sprint")),
            "model_trained": model is not None}


@app.get("/team-output")
def team_output():
    """What is happening: story points delivered per sprint, per team."""
    ps = table("people_sprint")
    ps = ps[ps.role != "QA"]
    return ps.groupby(["sprint", "team"], as_index=False).points_done.sum().round(1).to_dict("records")


@app.get("/ticket-delays")
def ticket_delays():
    """Why is it slow: average days per ticket spent building, blocked, and waiting for QA, per team."""
    t = table("tickets")
    t = t[t.cycle_days.notna()]
    out = t.groupby("team")[["dev_days", "blocked_days", "qa_wait_days"]].mean().round(1)
    out["share_missing_sprint"] = t.groupby("team").missed_sprint.mean().round(2)
    return out.reset_index().to_dict("records")


@app.get("/actions")
def actions():
    """What to do: simple rules a team leader can act on."""
    result = []
    ps = table("people_sprint")
    if ps.empty:
        return result
    ps["time_outside_core_work_pct"] = (1 - ps.core_work / 80) * 100       # 80 hours = one sprint

    seniors = ps[ps.role == "Senior Developer"].groupby("team").time_outside_core_work_pct.mean()
    for team, pct in seniors.items():
        if pct > seniors.median() + 10:
            result.append(f"Team {team}: senior developer spends {pct:.0f}% of time outside core work. Clarify roles.")

    latest = ps[(ps.sprint == ps.sprint.max()) & (ps.role != "QA") & (ps.tenure_months < 6)]
    medians = ps[(ps.sprint == ps.sprint.max()) & (ps.role != "QA")].groupby("team").points_done.median()
    for _, p in latest.iterrows():
        if p.points_done < 0.7 * medians[p.team]:
            result.append(f"{p.person} is new ({p.tenure_months:.0f} months) and delivering well below the team median. Check onboarding.")

    d = ticket_delays()
    for row in d:
        if row["qa_wait_days"] > 3:
            result.append(f"Team {row['team']}: tickets wait {row['qa_wait_days']} days for QA. Add test capacity.")
        if row["blocked_days"] > 2:
            result.append(f"Team {row['team']}: tickets are blocked {row['blocked_days']} days by other teams. Plan dependencies early.")
    return result


# ---------------------------------------------------------------- prediction: will a ticket miss the sprint?
model = None   # kept in memory; call /train again after a restart


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_bug"] = (df.issue_type == "Bug").astype(int)
    return df[["points", "has_dependency", "is_bug", "assignee_tenure", "team"]]


@app.post("/train")
def train():
    """Learn from finished tickets in the database."""
    global model
    t = table("tickets")
    t = t[t.missed_sprint.notna()]
    if len(t) < 60 or t.missed_sprint.nunique() < 2:
        raise HTTPException(422, f"Need at least 60 finished tickets, with both late and on-time ones. Found {len(t)}.")
    prep = ColumnTransformer([("num", StandardScaler(), ["points", "has_dependency", "is_bug", "assignee_tenure"]),
                              ("team", OneHotEncoder(handle_unknown="ignore"), ["team"])])
    model = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])
    model.fit(features(t), t.missed_sprint.astype(int))
    return {"trained_on_tickets": len(t), "late_share": round(float(t.missed_sprint.mean()), 2)}


@app.post("/risk")
def risk(ticket: NewTicket):
    """Chance that a new ticket misses the sprint."""
    if model is None:
        raise HTTPException(409, "Call POST /train first.")
    row = pd.DataFrame([ticket.model_dump()])
    row["has_dependency"] = row.has_dependency.astype(int)
    p = float(model.predict_proba(features(row))[0, 1])
    return {"risk_of_missing_sprint": round(p, 2), "level": "high" if p >= 0.5 else "medium" if p >= 0.25 else "low"}
