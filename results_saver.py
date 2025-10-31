# results_saver.py
from __future__ import annotations
import json
from datetime import datetime
import streamlit as st
from gsheets import open_sheet, open_or_create_worksheet

# Option A: one row per session, JSON blob of results
SHEET_RESULTS_JSON = "Matched Results (JSON)"
HEADER_JSON = ["timestamp", "user_id", "session_id", "results_json"]

def save_results_json(programmes: list[dict]) -> None:
    """
    Saves all matched programmes into ONE row (JSON column).
    """
    sh = open_sheet()
    ws = open_or_create_worksheet(sh, SHEET_RESULTS_JSON, HEADER_JSON)

    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [
        ts,
        st.session_state.get("user_id", ""),
        st.session_state.get("session_id", ""),
        json.dumps(programmes, ensure_ascii=False),
    ]
    ws.append_row(row, value_input_option="RAW")


# Option B: normalized, one row per programme
SHEET_RESULTS_TABLE = "Matched Results (Table)"
HEADER_TABLE = [
    "timestamp", "user_id", "session_id",
    "title", "description", "funding_area",
    "förderart", "höhe_der_förderung", "score"
]

def save_results_table(programmes: list[dict]) -> None:
    """
    Saves each programme as its own row (normalized table).
    """
    if not programmes:
        return

    sh = open_sheet()
    ws = open_or_create_worksheet(sh, SHEET_RESULTS_TABLE, HEADER_TABLE)

    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    rows = []
    for p in programmes:
        rows.append([
            ts,
            st.session_state.get("user_id", ""),
            st.session_state.get("session_id", ""),
            p.get("title", ""),
            p.get("description", ""),
            p.get("funding_area", ""),
            ", ".join(p.get("förderart", [])) if isinstance(p.get("förderart"), list) else (p.get("förderart") or ""),
            p.get("höhe_der_förderung", "") or "",
            f'{p.get("score", 0):.3f}',
        ])

    ws.append_rows(rows, value_input_option="RAW")
