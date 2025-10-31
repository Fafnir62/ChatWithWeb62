# lead_saver.py
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import streamlit as st

SHEET_LEADS = "Leads"

def _gs_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    return gspread.authorize(creds)

def _open_or_create_ws(sh):
    try:
        return sh.worksheet(SHEET_LEADS)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_LEADS, rows="2000", cols="20")
        ws.append_row([
            "timestamp", "user_id", "session_id",
            "company", "name", "email", "phone",
            "programme_title", "newsletter_optin", "datenschutz_optin"
        ])
        return ws

def save_lead(company, name, email, phone, programme_title, newsletter, datenschutz):
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    client = _gs_client()
    sh = client.open_by_key(st.secrets["sheets"]["answers_sheet_id"])
    ws = _open_or_create_ws(sh)

    ws.append_row([
        ts,
        st.session_state.get("user_id"),
        st.session_state.get("session_id"),
        company,
        name,
        email,
        phone,
        programme_title,
        "Yes" if newsletter else "No",
        "Yes" if datenschutz else "No",
    ])
