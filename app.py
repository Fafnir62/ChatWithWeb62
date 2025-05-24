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

# NEW → helper for funding-programme matching
from funding_matcher import build_or_load_index, matches_above_threshold


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
    st.session_state.programme_index = build_or_load_index()

# ─── SAVE USER ANSWERS ──────────────────────────────────────────
def save_user_answers():
    os.makedirs("user_data", exist_ok=True)
    with open(f"user_data/user_{st.session_state.user_id}.json", "w",
              encoding="utf-8") as f:
        json.dump({"user_id": st.session_state.user_id,
                   "answers":  st.session_state.tree_answers},
                  f, indent=2, ensure_ascii=False)

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
    save_user_answers()

# ─── HANDLE FREE CHAT ───────────────────────────────────────────
def handle_free_chat(txt: str):
    st.session_state.chat_history.append(HumanMessage(content=txt))
    ans = st.session_state.conversation_chain({"question": txt})["answer"]
    st.session_state.chat_history.append(AIMessage(content=ans))

# ─── SHOW MATCHES ONCE WHEN TREE DONE ───────────────────────────
def show_funding_matches(min_score: float = 0.30, k: int = 20):
    """
    Display all programmes with similarity score ≤ min_score.
    """
    if st.session_state.matches_shown or st.session_state.programme_index is None:
        return

    profile = "\n".join(f"{k}: {v}"
                        for k, v in st.session_state.tree_answers.items())

    programmes = matches_above_threshold(profile,
                                         min_score=min_score,
                                         k=k)

    with st.chat_message("ai"):
        if not programmes:
            st.markdown("❌ Leider passt kein Förderprogramm ausreichend zu Ihrem Profil.")
        else:
            st.markdown(f"🔍 **Programme mit Score ≤ {min_score:.2f}:**")

            # 💡 Erklärung zum Score hinzufügen
            st.markdown("""
            <div style="padding: 1rem; background-color: #f0f2f6; border-left: 5px solid #4a90e2; border-radius: 8px;">
            <h4 style="margin-top: 0;">🧠 Was bedeutet der Score?</h4>
            <p style="margin-bottom: 0.5rem;">
                Der <strong>Score-Wert</strong> zeigt, wie gut ein Förderprogramm zu Ihrem Projekt passt.
            </p>
            <ul style="margin-top: 0;">
                <li>🔵 <strong>Score &lt; 0.25</strong>: Sehr gute Übereinstimmung</li>
                <li>🟢 <strong>Score 0.30 – 0.30</strong>: Gute Relevanz</li>
                <li>⚪ <strong>Score &gt; 0.35</strong>: Geringe Passgenauigkeit</li>
            </ul>
            <p style="margin-top: 0.5rem;">
                <em>Je niedriger der Score, desto besser passt das Förderprogramm zu Ihrem Vorhaben.</em>
            </p>
            </div>
            """, unsafe_allow_html=True)
            for p in programmes:
                st.markdown(f"""
            **{p['title']}**

            {p['description']}

            - 🗂️ **Kategorie**: {p.get('category', '–')}
            - 🌍 **Fördergebiet**: {p.get('funding_area', '–')}
            - 🆔 **Call-ID**: {p.get('call_id', '–')}
            - ⏳ **Frist**: {p.get('submission_deadline', '–')}
            - 💶 **Förderart**: {', '.join(p.get('förderart', [])) if p.get('förderart') else '–'}
            - 📊 **Förderhöhe**: {p.get('höhe_der_förderung', '–')}
            - 🧮 **Score**: {p['score']:.3f}
            """)

    st.session_state.matches_shown = True

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
            if keys in (["Continue"], ["Submit"]):
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
if txt := st.chat_input("💬 Assistent fragen…"):
    handle_free_chat(txt)
    st.rerun()

# ─── SHOW PROGRAMME MATCHES (once) ──────────────────────────────
if st.session_state.tree_complete:
    show_funding_matches(min_score=0.35, k=20)
