# app.py  –  Funding-Assistant with score-based programme matching
# ---------------------------------------------------------------
import os, json, uuid, streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import html


# NEW → helper for funding-programme matching
from matcher_base     import get_index           # nur um Index einmal zu bauen
from matcher_location  import adjusted_matches    # Location-aware Matching

# ─── ENV & CONFIG ───────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Fördermittel-Chat",
                   page_icon="🤖",
                   layout="wide")

with open("styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# ─── SESSION STATE ──────────────────────────────────────────────
st.session_state.setdefault("tree_node", "start")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("tree_complete", False)
st.session_state.setdefault("tree_answers", {})
st.session_state.setdefault("user_id", str(uuid.uuid4()))
st.session_state.setdefault("last_tree_msg", None)
st.session_state.setdefault("matches_shown", False)

# ─── LOAD TREE & LINKS ──────────────────────────────────────────
with open("tree.json", encoding="utf-8") as f:
    TREE = json.load(f)
with open("links.json", encoding="utf-8") as f:
    LINKS = json.load(f)

# ─── FIRST TREE MESSAGE (only once) ─────────────────────────────
def push_first_tree_msg():
    first = TREE[st.session_state.tree_node].get("frage") \
            or TREE[st.session_state.tree_node].get("antwort")
    if first and first != st.session_state.last_tree_msg:
        st.session_state.chat_history.append(AIMessage(content=first))
        st.session_state.last_tree_msg = first

if not st.session_state.chat_history:
    push_first_tree_msg()

# ─── INIT VECTOR STORE FOR WEB SOURCES ──────────────────────────
def init_faiss(urls, persist_dir="faiss_index"):
    emb = OpenAIEmbeddings()
    if os.path.isdir(persist_dir):
        return FAISS.load_local(persist_dir, emb,
                                allow_dangerous_deserialization=True)
    docs = []
    for u in urls:
        docs.extend(WebBaseLoader(u).load())
    chunks = RecursiveCharacterTextSplitter().split_documents(docs)
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(persist_dir)
    return vs

if "vector_store" not in st.session_state:
    with st.spinner("🔄 Wissensbasis wird aufgebaut…"):
        st.session_state.vector_store = init_faiss(LINKS)

# ─── INIT LLM CHAT / RAG ────────────────────────────────────────
if "conversation_chain" not in st.session_state:
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )
    retriever = st.session_state.vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Du bist ein Experte für Fördermittel in Deutschland und Europa. "
         "Nutze den folgenden Kontext. Wenn die Antwort im Kontext steht, "
         "antworte nur damit. Sonst nutze dein Wissen. Wenn es keine "
         "Förderfrage ist, antworte:\n"
         "\"Die Frage kann und möchte ich nicht beantworten…\"\n\nKontext:\n{context}"),
        ("user", "{question}")
    ])
    st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# ─── FUNDING-PROGRAMME INDEX (built once) ───────────────────────
if "programme_index" not in st.session_state:
    st.session_state.programme_index = get_index()

# ─── SAVE USER ANSWERS ──────────────────────────────────────────
# ─── SAVE USER ANSWERS (Google Sheets) ──────────────────────────
def save_user_answers():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scope
    )
    client = gspread.authorize(creds)

    sh = client.open_by_key(st.secrets["sheets"]["answers_sheet_id"])
    ws_name = st.secrets["sheets"].get("worksheet_name")
    ws = sh.worksheet(ws_name) if ws_name else sh.sheet1  # 1ʳᵉ feuille sinon

    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user_id":   st.session_state.user_id,
        **st.session_state.tree_answers,
    }

    # ligne d’en-tête si la feuille est vide
    if ws.row_count == 0 or ws.acell("A1").value == "":
        ws.append_row(list(row.keys()))

    ws.append_row(list(row.values()))

# ─── ADVANCE TREE ───────────────────────────────────────────────
def advance_tree(next_node: str, user_reply: str):
    cur = st.session_state.tree_node
    st.session_state.pop(f"input_{cur}", None)

    cur_q = TREE[cur].get("frage") or TREE[cur].get("antwort") or ""
    kept = []
    for m in st.session_state.chat_history:
        kept.append(m)
        if isinstance(m, AIMessage) and m.content.strip() == cur_q.strip():
            break
    st.session_state.chat_history = kept

    st.session_state.chat_history.append(HumanMessage(content=user_reply))
    st.session_state.tree_answers[cur] = user_reply
    st.session_state.tree_node = next_node

    nxt_q = TREE[next_node].get("frage") or TREE[next_node].get("antwort")
    if nxt_q and nxt_q != st.session_state.last_tree_msg:
        st.session_state.chat_history.append(AIMessage(content=nxt_q))
        st.session_state.last_tree_msg = nxt_q

    st.session_state.tree_complete = (next_node == "chat")
    if st.session_state.tree_complete:
        save_user_answers()
# ─── HANDLE FREE CHAT ───────────────────────────────────────────
def handle_free_chat(txt: str):
    st.session_state.chat_history.append(HumanMessage(content=txt))
    ans = st.session_state.conversation_chain({"question": txt})["answer"]
    st.session_state.chat_history.append(AIMessage(content=ans))


import html
# ---------- Lead-Speicher --------------------------------------------------
def _save_lead(
    unternehmen: str,
    name: str,
    phone: str,
    mail: str,
    programme: str,
    newsletter_optin: bool,
    datenschutz_optin: bool
) -> None:
    # 1) Google-Auth
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    client = gspread.authorize(creds)

    # 2) Spreadsheet öffnen
    sh = client.open_by_key(st.secrets["sheets"]["answers_sheet_id"])

    # 3) Arbeitsblatt »Réponses 2« holen – falls es nicht existiert, anlegen
    try:
        ws = sh.worksheet("Réponses 2")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Réponses 2", rows="1000", cols="20")

    # 4) Header in der ersten Zeile nur einmal schreiben
    if ws.acell("A1").value == "":
        ws.append_row(
            [
                "timestamp", "user_id", "programme", 
                "unternehmen", "name", "phone", "mail", 
                "newsletter_optin", "datenschutz_optin"
            ],
            value_input_option="RAW"
        )

    # 5) Lead-Zeile anhängen
    ws.append_row(
        [
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            st.session_state.user_id,
            programme,
            unternehmen,
            name,
            phone,
            mail,
            "Ja" if newsletter_optin else "Nein",
            "Ja" if datenschutz_optin else "Nein",
        ],
        value_input_option="RAW"
    )

# ---------- Hauptfunktion --------------------------------------------------
def show_funding_matches(min_score: float = 0.30, base_k: int = 20) -> None:
    if "matched_programmes" not in st.session_state:
        profile  = "\n".join(f"{k}: {v}" for k, v in st.session_state.tree_answers.items())
        user_loc = st.session_state.tree_answers.get("location", "")
        st.session_state.matched_programmes = adjusted_matches(
            profile, user_location=user_loc, base_k=base_k, max_score=min_score
        )

    programmes = st.session_state.matched_programmes
    if not programmes:
        st.chat_message("ai").markdown("❌ Leider passt kein Förderprogramm ausreichend zu Ihrem Profil.")
        return

    st.markdown(
            f"""
            <p style='text-align:center; margin:1.5rem 0; font-size:1.1rem; line-height:1.4;'>
            🚀 Herzlichen Glückwunsch! Aufgrund Ihrer Angaben scheint es Fördermöglichkeiten für Ihr Projekt zu geben. Diese Förderprogramme hat der KI-Agent als besonders passend bewertet.
            </p>
            """,
            unsafe_allow_html=True)

    for idx, p in enumerate(programmes):
        with st.container(border=True):
            if idx < 2:
                # Blurred + Upgrade-Teaser mit Anfrage-Link
                st.markdown(
                    f"""
                    <div style='
                        filter: blur(0px);
                        pointer-events: none;
                        user-select: none;
                        color: gray;
                        border: 1px solid #ddd;
                        padding: 1rem;
                        border-radius: 8px;
                        background-color: #f9f9f9;
                    '>
                        <h4>{p['title']}</h4>
                        <p>{p['description']}</p>
                        <p>
                            📍 <strong>Gebiet:</strong> {p['funding_area']} &nbsp;&nbsp;
                            💶 <strong>Art:</strong> {', '.join(p['förderart'])} &nbsp;&nbsp;
                            💰 <strong>Höhe:</strong> {p['höhe_der_förderung'] or '–'} &nbsp;&nbsp;
                            📊 <strong>Score:</strong> {p['score']:.3f}
                        </p>
                    </div>
                    <div style='
                        margin-top: 0.5rem;
                        background: #000000;
                        padding: 1rem;
                        color: white;
                        font-weight: normal;
                        text-align: center;
                        border-radius: 8px;
                        border: 1px solid #000000;
                        font-family: 'DM Sans', sans-serif;
                    '>
                        <div style='margin-bottom:0.5rem;'>
                            🔓 Gerne senden wir Ihnen die geeignetsten Fördermittel per E-Mail.
                            <br>
                            Wir melden uns innerhalb von 24 Stunden mit konkreten Fördermöglichkeiten.
                        </div>
                        <a href="#kontaktformular" style='
                            display:inline-block;
                            background-color: #ff006e;
                            color: white;
                            text-decoration: none;
                            padding: 0.5rem 1rem;
                            border-radius: 5px;
                            font-weight: bold;
                            font-family: 'DM Sans', sans-serif;
                        '>Jetzt anfragen</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Normale Anzeige
                st.markdown(f"### {p['title']}", unsafe_allow_html=True)
                st.write(p["description"])
                meta = (
                    f"📍 **Gebiet:** {p['funding_area']} &nbsp;&nbsp; "
                    f"💶 **Art:** {', '.join(p['förderart'])} &nbsp;&nbsp; "
                    f"💰 **Höhe:** {p['höhe_der_förderung'] or '–'} &nbsp;&nbsp; "
                    f"📊 **Score:** {p['score']:.3f}"
                )
                st.markdown(meta)

                # Statt Button – Link mit Anker
                st.markdown(
                    f"""
                    <a href="#kontaktformular" style='
                        display:inline-block;
                        margin-top:0.5rem;
                        background-color: #ff006e;
                        color: white;
                        text-decoration: none;
                        padding: 0.5rem 1rem;
                        border-radius: 5px;
                        font-weight: bold;
                    '>Interesse / Rückruf</a>
                    """,
                    unsafe_allow_html=True
                )  

# ---------- Kontaktformular immer sichtbar --------------------------------
    # Anker für Sprung-Links
    st.markdown("<a name='kontaktformular'></a>", unsafe_allow_html=True)

    with st.form("lead_form", clear_on_submit=True):
        st.markdown("### Kontaktformular")

        st.markdown(
            "Bitte füllen Sie das folgende Formular aus – wir senden Ihnen die passenden Fördermittelvorschläge per E-Mail."
        )

        unternehmen = st.text_input(label="", placeholder="Unternehmensname *")
        name = st.text_input(label="", placeholder="Vorname, Nachname *")
        email = st.text_input(label="", placeholder="E-Mail-Adresse *")
        phone = st.text_input(label="", placeholder="Telefonnummer (optional)")

        st.markdown(
            """
            Die Welt der Fördermittel ist ständig im Wandel – gerne halten wir Sie in regelmäßigen Abständen auf dem Laufenden.
            Sie können diese Benachrichtigungen jederzeit abbestellen.
            """
        )
        newsletter_optin = st.checkbox(
            "Ich stimme zu, andere Benachrichtigungen von Fördermittel-Vergleich.de zu erhalten."
        )

        st.markdown(
            """
            Um Ihnen das Ergebnis Ihres Förderchecks mitzuteilen, müssen wir Ihre personenbezogenen Daten speichern und verarbeiten.
            """
        )
        datenschutz_optin = st.checkbox(
            "Ich stimme zu, dass meine Angaben zur Kontaktaufnahme und zur Bearbeitung meines Anliegens (z. B. zur Terminvereinbarung) gemäß der [Datenschutzerklärung](https://www.xn--frdermittel-vergleich-hec.de/datenschutz/) verarbeitet werden.*",
            help="Pflichtfeld"
        )

        st.markdown(
            """
            <small>
            Diese Einwilligung kann jederzeit (auch direkt im Anschluss) widerrufen werden. Informationen zum Abbestellen sowie unsere Datenschutzpraktiken und unsere Verpflichtung zum Schutz der Privatsphäre finden Sie in unseren Datenschutzbestimmungen.
            </small>
            """,
            unsafe_allow_html=True
        )

        submitted = st.form_submit_button("Absenden")
        if submitted:
            errors = []

            # Pflichtfelder prüfen
            if not unternehmen.strip():
                errors.append("Bitte geben Sie den Unternehmensnamen an.")
            if not name.strip():
                errors.append("Bitte geben Sie Ihren Namen an.")
            if not email.strip():
                errors.append("Bitte geben Sie Ihre E-Mail-Adresse an.")
            if not datenschutz_optin:
                errors.append("Sie müssen der Datenschutzerklärung zustimmen.")

            # E-Mail-Format prüfen
            import re
            email_regex = r"[^@]+@[^@]+\.[^@]+"
            if email and not re.match(email_regex, email):
                errors.append("Bitte geben Sie eine gültige E-Mail-Adresse ein.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                _save_lead(
                    unternehmen,
                    name,
                    phone,
                    email,
                    st.session_state.get("lead_programme", {}).get("title", "Lead aus Formular"),
                    newsletter_optin,
                    datenschutz_optin
                )
                st.success("Vielen Dank – wir melden uns innerhalb von 24 Stunden mit passenden Fördermöglichkeiten!")

                    
# ─── RENDER CHAT & INPUTS ───────────────────────────────────────
st.markdown("""
<div class="intro-box">
👋 Hallo! Ich bin <strong>Ihr kostenfreier KI-Fördermittelberater</strong>.
Ich helfe Ihnen in Rekordzeit das passende Fördermittel zu finden und die Förderfähigkeit zu überprüfen. Bitte beantworte folgende 10 Fragen möglichst ausführlich, um das beste Ergebnis zu erzielen.
</div>
""", unsafe_allow_html=True)

current = TREE[st.session_state.tree_node]
need_input = not st.session_state.tree_complete

for m in st.session_state.chat_history:
    role = "human" if isinstance(m, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(m.content)

    if (
        need_input
        and isinstance(m, AIMessage)
        and m.content.strip() == (current.get("frage") or current.get("antwort", "")).strip()
    ):
        need_input = False

        # --- 1️⃣ Select Dropdown Node ----------------------
        if "select" in current:
            sel_key = f"input_{st.session_state.tree_node}"
            selection = st.selectbox(
                "Auswahl treffen",
                current["select"]["choices"],
                key=sel_key
            )
            if st.button("Absenden", key=f"btn_{st.session_state.tree_node}_select"):
                if selection.strip():
                    advance_tree(current["select"]["next"], selection.strip())
                    st.rerun()
                else:
                    st.warning("Bitte eine Auswahl treffen…")

        # --- 2️⃣ Text Input Node with Enter or Button ------
        elif "optionen" in current:
            keys = list(current["optionen"].keys())

            if keys in (["Weiter"], ["Absenden"]):
                inp_key = f"input_{st.session_state.tree_node}"
                prev_key = f"{inp_key}_prev"

                # Hole aktuellen Wert
                current_val = st.session_state.get(inp_key, "")
                prev_val = st.session_state.get(prev_key, "")

                # Textfeld ohne Label
                reply = st.text_input(
                    "",
                    placeholder="Antwort hier eingeben …",
                    key=inp_key
                )

                # ENTER-Simulation → wenn sich Wert geändert hat
                if reply.strip() and reply != prev_val:
                    st.session_state[prev_key] = reply
                    advance_tree(current["optionen"][keys[0]], reply.strip())
                    st.rerun()

                # Fallback-Button
                if st.button("Absenden", key=f"btn_{st.session_state.tree_node}_text"):
                    if reply.strip():
                        st.session_state[prev_key] = reply
                        advance_tree(current["optionen"][keys[0]], reply.strip())
                        st.rerun()
                    else:
                        st.warning("Bitte etwas eingeben …")

            # --- 3️⃣ Multiple Choice Buttons ---------------
            else:
                st.markdown('<div class="button-wrap">', unsafe_allow_html=True)
                for lbl, nxt in current["optionen"].items():
                    with st.container():
                        if st.button(lbl, use_container_width=True, key=f"btn_{st.session_state.tree_node}_{lbl}"):
                            advance_tree(nxt, lbl)
                            st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # --- 4️⃣ External Link Leaf -------------------------
        elif "button_label" in current:
            st.link_button(current["button_label"], current["button_link"])



# ─── ALWAYS-ON FREE CHAT ────────────────────────────────────────
#if txt := st.chat_input("💬 Assistent fragen…"):
  #  handle_free_chat(txt)
  #  st.rerun()

# ─── SHOW PROGRAMME MATCHES (once) ──────────────────────────────
if st.session_state.tree_complete:
    show_funding_matches(min_score=0.35, base_k=20)
