import os, sys, uuid, html
import streamlit as st
from dotenv import load_dotenv

# Add project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ---- CORE LOGIC ----
from core.check_all import check_all
from core.merge import merge_answers, pretty_status
from core.rules import next_missing

# Question extractors
from core.questions.bundesland import check_bundesland
from core.questions.projektkosten import check_projektkosten
from core.questions.eigenanteil import check_eigenanteil
from core.questions.gruendungsjahr import check_gruendungsjahr
from core.questions.kategorie import check_kategorie
from core.questions.branche import check_branche

# Matcher
from matching import match_programs, render_matches

# ---- SAVERS ----
from chat_saver import save_chat_session_row
from results_saver import save_results_json, save_results_table
from lead_saver import save_lead


# -------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Fördermittel-Chat", page_icon="💬", layout="wide")

# -------- Global font (DM Sans) + layout --------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

html, body, div, p,
h1, h2, h3, h4, h5, h6,
input, button, textarea { font-family: 'DM Sans', sans-serif; }

/* Basic spacing only */
.block-container { padding-top: 1rem; }

/* ---------------- Chat bubbles ---------------- */
.chat-wrap { display:flex; flex-direction:column; gap:12px; margin-top:6px; }
.turn-gap  { height:40px; }
.msg { display:flex; align-items:center; width:100%; }
.msg.ai   { justify-content:flex-start; }
.msg.user { justify-content:flex-end; }

.msg .avatar { width:32px; height:32px; margin-right:12px; display:flex; align-items:center; justify-content:center; flex:0 0 32px; }
.msg .avatar svg { width:32px; height:32px; display:block; }
.msg.user .avatar { display:none; }

.msg .bubble { max-width:78%; line-height:1.55; font-size:0.98rem; }
.msg.ai .bubble { background:transparent; border:none; padding:0; margin-left:4px; }
.msg.user .bubble { background:#f2f2f2; border:none; color:#111; padding:12px 16px; border-radius:12px; margin-right:6px; }
.msg .bubble .t { white-space:pre-wrap; }

/* ---------------- First screen layout only ---------------- */
.first-screen-title {
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  top: 42vh;
  width: min(920px, 92vw);
  text-align: center;
  z-index: 1000;
}
.first-screen-title h1 {
  font-size: clamp(18px, 2.2vw, 26px);
  font-weight: 600;
  line-height: 1.5;
  margin: 0;
}

/* Center the native chat input on the first screen (no structure changes) */
.first-screen-input-center [data-testid="stChatInput"] {
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  top: 54vh;                 /* adjust to sit below the title */
  bottom: auto;
  width: min(920px, 92vw);
  z-index: 1000;
}
.first-screen-input-center .block-container { padding-bottom: 0 !important; }

/* ====== Always show the thin pink border (same as focus) ====== */
:root { --bm-primary: var(--primary-color, #ff006e); }

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input[type="text"] {
  border-color: var(--bm-primary) !important;   /* always pink border */
  outline: none !important;
  /* --- Slightly darker input background while keeping text readable --- */
  background-color: #f6f7f9 !important;        /* subtle grey */
  color: #111 !important;                       /* high contrast text */
}

/* Keep identical on focus (no visual jump) */
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] input[type="text"]:focus {
  border-color: var(--bm-primary) !important;
  outline: none !important;
  background-color: #f6f7f9 !important;
  color: #111 !important;
}

/* Darker placeholder text */
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
  color: rgba(0,0,0,0.68) !important;
  opacity: 1 !important;
}
[data-testid="stChatInput"] textarea::-webkit-input-placeholder,
[data-testid="stChatInput"] input::-webkit-input-placeholder {
  color: rgba(0,0,0,0.68) !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# ---- SESSION STATE ----
# -------------------------------------------------------
st.session_state.setdefault("chat", [])  # list of tuples: ("ai"|"user", text)
st.session_state.setdefault("answers", {
    "kategorie": "not found",
    "branche": "not found",
    "bundesland": "not found",
    "gruendungsjahr": "not found",
    "projektkosten_eur": "not found",
    "eigenanteil_eur": "not found",
})
st.session_state.setdefault("pending_key", None)
st.session_state.setdefault("pending_label", None)
st.session_state.setdefault("project_description", "")
st.session_state.setdefault("latest_matches", None)
st.session_state.setdefault("matches_key", None)
st.session_state.setdefault("user_id", str(uuid.uuid4()))
st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("session_saved", False)
st.session_state.setdefault("results_saved", False)
st.session_state.setdefault("pending_user", None)  # Optimistic UI flag


# ---------- helpers ----------
def _escape(md_text: str) -> str:
    return html.escape(md_text)

AI_ICON_SVG = """
<svg viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="2" y="2" width="28" height="28" rx="8" fill="#ff006e"/>
  <rect x="9" y="10" width="14" height="12" rx="3" fill="#ffffff"/>
  <circle cx="13" cy="16" r="1.6" fill="#ff006e"/>
  <circle cx="19" cy="16" r="1.6" fill="#ff006e"/>
  <rect x="13" y="19.5" width="6" height="2" rx="1" fill="#ff006e"/>
  <rect x="15" y="6" width="2" height="3" rx="1" fill="#ffffff"/>
</svg>
"""

def render_chat(history):
    st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
    prev_role = None
    for role, msg in history:
        if prev_role is not None and role != prev_role:
            st.markdown("<div class='turn-gap'></div>", unsafe_allow_html=True)

        if role == "ai":
            st.markdown(
                f"""
                <div class="msg ai">
                  <div class="avatar">{AI_ICON_SVG}</div>
                  <div class="bubble"><div class="t">{_escape(msg)}</div></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="msg user">
                  <div class="avatar"></div>
                  <div class="bubble"><div class="t">{_escape(msg)}</div></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        prev_role = role
    st.markdown("</div>", unsafe_allow_html=True)


# ---- FIRST SCREEN (no messages yet) ----
first_screen = len(st.session_state["chat"]) == 0

if first_screen:
    # Title (not a chat message)
    st.markdown(
        """
        <div class="first-screen-title">
          <h1>
            Lass uns starten. Beschreibe zunächst dein Projekt so genau und ausführlich wie möglich.
            Je genauer du bist, desto besser wird am Ende das Matching mit relevanten Fördermitteln für euch.
          </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Wrapper class just to center the input
    st.markdown('<div class="first-screen-input-center">', unsafe_allow_html=True)

# ---- USER INPUT (native Streamlit behavior) ----
placeholder_text = (
    "Diese Frage musst du bitte mindestens beantworten: "
    "Was habt ihr bzw. was werdet ihr entwickeln, erforschen, sanieren oder finanzieren?"
)
user_msg = st.chat_input(placeholder_text)

if first_screen:
    st.markdown('</div>', unsafe_allow_html=True)

# When the user sends the first message, it becomes the first chat entry (from the user).
if user_msg:
    st.session_state["chat"].append(("user", user_msg))
    st.session_state["pending_user"] = user_msg
    st.rerun()

# ---- DISPLAY CHAT after first message ----
if not first_screen:
    render_chat(st.session_state["chat"])

# ---- PROCESS ANY PENDING USER MESSAGE (optimistic UX) ----
if st.session_state.get("pending_user"):
    with st.spinner("Einen Moment …"):
        user_msg_to_process = st.session_state["pending_user"]

        if len(user_msg_to_process) > len(st.session_state["project_description"]):
            st.session_state["project_description"] = user_msg_to_process

        pending_key  = st.session_state["pending_key"]
        pending_label= st.session_state["pending_label"]
        partial, used = {}, False

        if pending_key:
            if   pending_key == "bundesland":        out = check_bundesland(user_msg_to_process, context=pending_label); used=True
            elif pending_key == "projektkosten_eur":  out = check_projektkosten(user_msg_to_process, context=pending_label); used=True
            elif pending_key == "eigenanteil_eur":    out = check_eigenanteil(user_msg_to_process, context=pending_label); used=True
            elif pending_key == "gruendungsjahr":     out = check_gruendungsjahr(user_msg_to_process); used=True
            elif pending_key == "kategorie":          out = check_kategorie(user_msg_to_process); used=True
            elif pending_key == "branche":            out = check_branche(user_msg_to_process); used=True
            else: used = False

            if used and out.get("answered"):
                partial = { pending_key: out["value"] }

        if not partial:
            partial = check_all(user_msg_to_process)

        st.session_state["answers"] = merge_answers(st.session_state["answers"], partial)
        status = pretty_status(st.session_state["answers"])
        nk, nl = next_missing(st.session_state["answers"])

        if nl:
            st.session_state["pending_key"] = nk
            st.session_state["pending_label"] = nl
            bot = f"Danke! Bisher erkannt:\n\n{status}\n\n**{nl}**"
        else:
            st.session_state["pending_key"] = None
            st.session_state["pending_label"] = None
            bot = f"✅ Alles erfasst!\n\n{status}\n\nDu kannst Werte korrigieren."

        st.session_state["chat"].append(("ai", bot))
        st.session_state["pending_user"] = None
        st.rerun()


# ---- RESULTS ----
def _answers_key():
    a = st.session_state["answers"]
    return tuple(sorted((k, str(v)) for k, v in a.items())), st.session_state["project_description"]

nk, nl = next_missing(st.session_state["answers"])
if nl is None:
    st.markdown("## 🎯 Passende Förderprogramme")

    key = _answers_key()
    if st.session_state["matches_key"] != key or st.session_state["latest_matches"] is None:
        json_path = os.path.join(os.path.dirname(__file__), "foerdermittel_normalized.json")
        matches = match_programs(
            st.session_state["answers"],
            st.session_state["project_description"],
            json_path=json_path,
            min_score=0.7
        )
        st.session_state["latest_matches"] = matches
        st.session_state["matches_key"] = key

    if not st.session_state["session_saved"]:
        save_chat_session_row()
        st.session_state["session_saved"] = True

    if not st.session_state["results_saved"]:
        save_results_json(st.session_state["latest_matches"])
        save_results_table(st.session_state["latest_matches"])
        st.session_state["results_saved"] = True

    render_matches(st.session_state["latest_matches"])

    st.markdown("### 📩 Interesse? Wir melden uns")
    with st.form("lead_form", clear_on_submit=True):
        company = st.text_input("Unternehmen *")
        name = st.text_input("Name *")
        email = st.text_input("E-Mail *")
        phone = st.text_input("Telefon (optional)")
        newsletter = st.checkbox("Newsletter erhalten?")
        datenschutz = st.checkbox("Ich stimme dem Datenschutz zu *")

        submit = st.form_submit_button("Absenden ✅")
        if submit:
            if not company or not name or not email or not datenschutz:
                st.error("Bitte Pflichtfelder ausfüllen.")
            else:
                save_lead(company, name, email, phone,
                          "Chat Match Result", newsletter, datenschutz)
                st.success("✅ Danke! Wir melden uns.")
