# app.py
import os, sys
import streamlit as st
from dotenv import load_dotenv

# Ensure local package path (helps when launching from other dirs)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.check_all import check_all
from core.merge import merge_answers, pretty_status
from core.rules import next_missing

# Checkers (we pass context where relevant)
from core.questions.bundesland import check_bundesland
from core.questions.projektkosten import check_projektkosten
from core.questions.eigenanteil import check_eigenanteil
from core.questions.gruendungsjahr import check_gruendungsjahr
from core.questions.kategorie import check_kategorie
from core.questions.branche import check_branche

# Matching service
from matching import match_programs, render_matches

# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
st.set_page_config(page_title="Fördermittel-Chat", page_icon="🤖", layout="wide")

# Minimal styles (optional)
st.markdown("""
<style>
.small { opacity:.8; font-size:.9em }
</style>
""", unsafe_allow_html=True)

# ---------- INIT SESSION STATE SAFELY ----------
if "chat" not in st.session_state:
    st.session_state["chat"] = []
if "answers" not in st.session_state:
    st.session_state["answers"] = {
        "kategorie": "not found",        # Innovation / Investition / Finanzierung
        "branche": "not found",
        "bundesland": "not found",
        "gruendungsjahr": "not found",   # only required if kategorie == Innovation
        "projektkosten_eur": "not found",
        "eigenanteil_eur": "not found",
    }
if "pending_key" not in st.session_state:
    st.session_state["pending_key"] = None
if "pending_label" not in st.session_state:
    st.session_state["pending_label"] = None
if "project_description" not in st.session_state:
    st.session_state["project_description"] = ""

# Contact form visibility/state controlled by matching.render_matches
st.session_state.setdefault("show_contact_form", False)
st.session_state.setdefault("contact_selection", None)

# Cache for results across reruns (so they remain visible on click)
st.session_state.setdefault("latest_matches", None)
st.session_state.setdefault("matches_key", None)

# ---------- INTRO ONCE ----------
if not st.session_state["chat"]:
    st.session_state["chat"].append((
        "ai",
        "👋 Erklär dein Projekt bitte in ein paar Sätzen. "
        "Ich prüfe, welche der 6 Fragen damit bereits beantwortet sind."
    ))

# ---------- RENDER HISTORY ----------
for role, msg in st.session_state["chat"]:
    with st.chat_message("ai" if role == "ai" else "human"):
        st.markdown(msg)

# ---------- INPUT ----------
user_msg = st.chat_input("Beschreibe dein Projekt …")
if user_msg:
    # 1) Show user's message immediately (before any analysis)
    with st.chat_message("human"):
        st.markdown(user_msg)

    # Keep/refresh the project description for matching (prefer the longest message)
    if len(user_msg) > len(st.session_state["project_description"]):
        st.session_state["project_description"] = user_msg

    # 2) Create an AI message bubble with a spinner while thinking
    with st.chat_message("ai"):
        with st.spinner("🤖 Der Agent denkt …"):
            # ANALYZE (context-aware routing first, then fallback extractors)
            pending_key = st.session_state.get("pending_key")
            pending_label = st.session_state.get("pending_label")
            partial = {}
            used_direct = False

            if pending_key:
                if pending_key == "bundesland":
                    out = check_bundesland(user_msg, context=pending_label)
                    if out.get("answered"):
                        partial = {"bundesland": out["value"]}
                        used_direct = True
                elif pending_key == "projektkosten_eur":
                    out = check_projektkosten(user_msg, context=pending_label)
                    if out.get("answered"):
                        partial = {"projektkosten_eur": out["value"]}
                        used_direct = True
                elif pending_key == "eigenanteil_eur":
                    out = check_eigenanteil(user_msg, context=pending_label)
                    if out.get("answered"):
                        partial = {"eigenanteil_eur": out["value"]}
                        used_direct = True
                elif pending_key == "gruendungsjahr":
                    out = check_gruendungsjahr(user_msg)
                    if out.get("answered"):
                        partial = {"gruendungsjahr": out["value"]}
                        used_direct = True
                elif pending_key == "kategorie":
                    out = check_kategorie(user_msg)
                    if out.get("answered"):
                        partial = {"kategorie": out["value"]}
                        used_direct = True
                elif pending_key == "branche":
                    out = check_branche(user_msg)
                    if out.get("answered"):
                        partial = {"branche": out["value"]}
                        used_direct = True

            if not used_direct:
                partial = check_all(user_msg)

            # Merge recognized fields
            st.session_state["answers"] = merge_answers(st.session_state["answers"], partial)

            # Build summary + figure out next question (order + Innovation rule)
            status = pretty_status(st.session_state["answers"])
            next_key, next_label = next_missing(st.session_state["answers"])

            if next_label:
                st.session_state["pending_key"] = next_key
                st.session_state["pending_label"] = next_label
                bot = "Danke! Bisher habe ich erkannt:\n\n" + status + f"\n\n**{next_label}**"
            else:
                st.session_state["pending_key"] = None
                st.session_state["pending_label"] = None
                bot = (
                    "Top, alle benötigten Angaben sind da:\n\n" + status +
                    "\n\nDu kannst jederzeit korrigieren (z. B. „Bundesland ist Bayern“)."
                )

        # After spinner closes, show the computed bot message
        st.markdown(bot)

    # 3) Persist both messages into history for future reruns
    st.session_state["chat"].append(("human", user_msg))
    st.session_state["chat"].append(("ai", bot))

# ---------- RESULTS SECTION (always render when answers complete) ----------
def _answers_key():
    # Stable key to detect changes and avoid recomputation when not needed
    a = st.session_state["answers"]
    # tuple of sorted items so key order doesn't matter
    items = tuple(sorted((k, str(v)) for k, v in a.items()))
    return (items, st.session_state.get("project_description", ""))

# Only render results when all answers are present
cur_next_key, cur_next_label = next_missing(st.session_state["answers"])
if cur_next_label is None:
    st.markdown("## 🎯 Relevante Fördermittel passend zu deinem Projekt")

    key = _answers_key()
    # Recompute matches only if answers or description changed
    if st.session_state["matches_key"] != key or st.session_state.get("latest_matches") is None:
        json_path = os.path.join(os.path.dirname(__file__), "foerdermittel_normalized.json")

        # IMPORTANT: don't reset contact flags here when only re-rendering due to button click.
        # Reset selection ONLY when the underlying result set changes.
        st.session_state["show_contact_form"] = False
        st.session_state["contact_selection"] = None

        matches = match_programs(
            st.session_state["answers"],
            st.session_state.get("project_description", ""),
            json_path=json_path,
            min_score=0.7,      # stricter threshold
            max_results=None    # show all ≥ threshold
        )
        st.session_state["latest_matches"] = matches
        st.session_state["matches_key"] = key

    # Always render the last computed results so they remain visible on reruns
    render_matches(st.session_state["latest_matches"])
