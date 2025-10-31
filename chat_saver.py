# chat_saver.py
from __future__ import annotations
import json
from datetime import datetime
import streamlit as st
from gsheets import open_sheet, open_or_create_worksheet

SHEET_SESSIONS = "Chat Sessions"
HEADER = ["timestamp", "user_id", "session_id", "project_description", "answers_json", "messages_json"]

def save_chat_session_row() -> None:
    """
    Appends ONE row per conversation with:
      - timestamp
      - user_id
      - session_id
      - project_description (string)
      - answers_json (dict as JSON)
      - messages_json (list[(role, msg)] as JSON)
    """
    sh = open_sheet()
    ws = open_or_create_worksheet(sh, SHEET_SESSIONS, HEADER)

    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [
        ts,
        st.session_state.get("user_id", ""),
        st.session_state.get("session_id", ""),
        st.session_state.get("project_description", ""),
        json.dumps(st.session_state.get("answers", {}), ensure_ascii=False),
        json.dumps(st.session_state.get("chat", []), ensure_ascii=False),
    ]
    ws.append_row(row, value_input_option="RAW")
