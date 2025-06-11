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
def _save_lead(name: str, phone: str, mail: str, programme: str) -> None:
    # 1) Google-Auth
    creds  = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=["https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive"])
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
            ["timestamp", "user_id", "programme", "name", "phone", "mail"],
            value_input_option="RAW")

    # 5) Lead-Zeile anhängen
    ws.append_row(
        [
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            st.session_state.user_id,
            programme,
            name,
            phone,
            mail,
        ],
        value_input_option="RAW",
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

    with st.chat_message("ai"):
        st.markdown(
            f"<h3 style='text-align:center;margin:2rem 0 1rem;'>"
            f"Gefundene Förderprogramme&nbsp;(Score ≤ {min_score:.2f})</h3>",
            unsafe_allow_html=True)

        for idx, p in enumerate(programmes):
            with st.container(border=True):
                # --------- Textbereich --------------------------------------------------
                st.markdown(f"### {p['title']}", unsafe_allow_html=True)
                st.write(p["description"])
                meta = (
                    f"📍 **Gebiet:** {p['funding_area']} &nbsp;&nbsp; "
                    f"💶 **Art:** {', '.join(p['förderart'])} &nbsp;&nbsp; "
                    f"💰 **Höhe:** {p['höhe_der_förderung'] or '–'} &nbsp;&nbsp; "
                    f"📊 **Score:** {p['score']:.3f}"
                )
                st.markdown(meta)

                # --------- Button  ------------------------------------------------------
                btn_key = f"lead_btn_{idx}"
                if st.button("Interesse / Rückruf", key=btn_key):
                    st.session_state.lead_programme = p
                    st.session_state.show_lead = True

    # ---------- Pop-up / Formular ------------------------------------------
    if st.session_state.get("show_lead", False):
        prog = st.session_state["lead_programme"]
        container = st.modal(f"Kontakt für: {prog['title']}") if hasattr(st, "modal") else st.container()

        with container:
            with st.form("lead_form", clear_on_submit=True):
                name  = st.text_input("Name")
                phone = st.text_input("Telefon")
                mail  = st.text_input("E-Mail")
                if st.form_submit_button("Absenden"):
                    _save_lead(name, phone, mail, prog["title"])
                    st.success("Vielen Dank – wir melden uns!")
                    st.session_state.show_lead = False
                    
# ─── RENDER CHAT & INPUTS ───────────────────────────────────────
current = TREE[st.session_state.tree_node]
need_input = not st.session_state.tree_complete

for m in st.session_state.chat_history:
    role = "human" if isinstance(m, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(m.content)

    if (need_input and isinstance(m, AIMessage)
        and m.content.strip() == (current.get("frage")
                                  or current.get("antwort", "")).strip()):

        need_input = False
        if "optionen" in current:
            keys = list(current["optionen"].keys())

            # text-input node
            if keys in (["Weiter"], ["Absenden"]):
                reply = st.text_input(
                    "✏️ Antwort eingeben",
                    value="",
                    key=f"input_{st.session_state.tree_node}"
                )
                if st.button("✅ Absenden"):
                    if reply.strip():
                        advance_tree(current["optionen"][keys[0]], reply.strip())
                        st.rerun()
                    else:
                        st.warning("Bitte etwas eingeben…")

            # multiple-choice node
            else:
                cols = st.columns(len(current["optionen"]))
                for i, (lbl, nxt) in enumerate(current["optionen"].items()):
                    with cols[i]:
                        if st.button(lbl):
                            advance_tree(nxt, lbl)
                            st.rerun()

        elif "button_label" in current:   # leaf with external link
            st.link_button(current["button_label"],
                           current["button_link"])

# ─── ALWAYS-ON FREE CHAT ────────────────────────────────────────
#if txt := st.chat_input("💬 Assistent fragen…"):
  #  handle_free_chat(txt)
  #  st.rerun()

# ─── SHOW PROGRAMME MATCHES (once) ──────────────────────────────
if st.session_state.tree_complete:
    show_funding_matches(min_score=0.35, base_k=20)
