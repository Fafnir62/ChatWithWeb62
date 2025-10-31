# gsheets.py
from __future__ import annotations
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from typing import List

SCOPES: List[str] = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_client() -> gspread.Client:
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    return gspread.authorize(creds)

def open_sheet():
    client = get_client()
    return client.open_by_key(st.secrets["sheets"]["answers_sheet_id"])

def open_or_create_worksheet(sh, title: str, header: list[str]):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows="2000", cols="26")
        if header:
            ws.append_row(header, value_input_option="RAW")
    return ws
