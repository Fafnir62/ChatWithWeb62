# ── sqlite3 monkey‐patch for Chromadb compatibility (just in case) ──
try:
    import pysqlite3 as sqlite3
    import sys
    sys.modules["sqlite3"] = sqlite3
except ImportError:
    pass

import os
import json
import streamlit as st
from dotenv import load_dotenv

# LangChain core messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# FAISS & embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Conversational chain + memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

# ———————————————————————————————————————————————————————
# CONFIG & STYLE
# ———————————————————————————————————————————————————————
load_dotenv()  # loads OPENAI_API_KEY

st.set_page_config(page_title="Fördermittel-Chat", page_icon="🤝", layout="wide")
PINK = "#D9005A"
st.markdown(f"""
<style>
.reportview-container .main {{ background: {PINK}; color: white; }}
.sidebar .sidebar-content {{ background: {PINK}; color: white; }}
.stButton>button {{
  background: white;
  color: {PINK};
  font-weight: bold;
  width: 100%;
  margin-bottom: 8px;
}}
</style>
""", unsafe_allow_html=True)

st.image("Foerdermittel Vergleich.png", width=200)

# ———————————————————————————————————————————————————————
# LOAD DATA
# ———————————————————————————————————————————————————————
with open("links.json", "r", encoding="utf-8") as f:
    LINKS = json.load(f)
with open("tree.json", "r", encoding="utf-8") as f:
    TREE = json.load(f)

# ———————————————————————————————————————————————————————
# INIT OR LOAD FAISS
# ———————————————————————————————————————————————————————
def init_faiss(urls, persist_dir="faiss_index"):
    emb = OpenAIEmbeddings()
    if os.path.isdir(persist_dir):
        return FAISS.load_local(persist_dir, emb, allow_dangerous_deserialization=True)
    docs = []
    for url in urls:
        docs.extend(WebBaseLoader(url).load())
    chunks = RecursiveCharacterTextSplitter().split_documents(docs)
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(persist_dir)
    return vs

if "vector_store" not in st.session_state:
    with st.spinner("Wissensbasis (FAISS) wird aufgebaut…"):
        st.session_state.vector_store = init_faiss(LINKS)

# ———————————————————————————————————————————————————————
# SESSION STATE & CHAIN INIT
# ———————————————————————————————————————————————————————
st.session_state.setdefault("tree_node", "start")
st.session_state.setdefault("chat_history", [])

if "conversation_chain" not in st.session_state:
    # 1) Our LLM
    llm = ChatOpenAI(temperature=0)

    # 2) Explicitly record question→answer so follow-ups get rewritten
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    # 3) Base retriever from FAISS
    retriever = st.session_state.vector_store.as_retriever()

    # 4) Prompt: use context *if* it contains the answer; otherwise fall back
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Du bist ein Experte für Fördermittel in Deutschland und Europa. "
            "Nutze den folgenden Kontext. "
            "Wenn die Antwort im Kontext steht, antworte ausschließlich darauf. "
            "Ist sie nicht im Kontext enthalten, beantworte basierend auf deinem allgemeinen Wissen zu Fördermitteln. "
            "Wenn die Frage nicht zum Bereich Fördermittel gehört, antworte genau:\n\n"
            "\"Die Frage kann und möchte ich nicht beantworten, denn meine Expertise liegt bei Fördermitteln. "
            "Stell mir bitte eine Frage rund um das Thema Fördermittel in Deutschland, Europa oder in einem bestimmten Bundesland.\"\n\n"
            "Kontext:\n{context}"
        ),
        ("user", "{question}")
    ])

    # 5) Build the conversational-retrieval chain
    st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": answer_prompt}
    )

# ———————————————————————————————————————————————————————
# TREE CALLBACK
# ———————————————————————————————————————————————————————
def advance_tree(next_node: str):
    st.session_state.tree_node = next_node

# ———————————————————————————————————————————————————————
# CHAT CALLBACK
# ———————————————————————————————————————————————————————
def handle_send():
    q = st.session_state.user_text.strip()
    if not q:
        return

    # 1) Add to UI history
    st.session_state.chat_history.append(HumanMessage(content=q))

    # 2) Run the chain with just the new question — it will:
    #    • rewrite the question using full memory
    #    • retrieve from FAISS
    #    • combine via our prompt (which now falls back to general knowledge)
    res = st.session_state.conversation_chain({"question": q})
    a = res["answer"]

    # 3) Record and display
    st.session_state.chat_history.append(AIMessage(content=a))
    st.session_state.user_text = ""

# ———————————————————————————————————————————————————————
# RENDER DECISION TREE (top)
# ———————————————————————————————————————————————————————
node = TREE[st.session_state.tree_node]
if "frage" in node:
    st.markdown(f"### {node['frage']}")
    for label, nxt in node["optionen"].items():
        st.button(
            label,
            key=f"tree_{st.session_state.tree_node}_{label}",
            on_click=advance_tree,
            args=(nxt,)
        )
else:
    st.markdown(f"**{node['antwort']}**")
    st.markdown(f"""
      <a href="{node['button_link']}" target="_blank">
        <button style="
          background-color: white;
          color: {PINK};
          padding: 8px 16px;
          border: none;
          font-weight: bold;
          width: 100%;
          margin-top: 8px;
        ">
          {node['button_label']}
        </button>
      </a>
    """, unsafe_allow_html=True)

# ———————————————————————————————————————————————————————
# RENDER CHAT (40 px below tree)
# ———————————————————————————————————————————————————————
st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    role = "human" if isinstance(msg, HumanMessage) else "ai"
    with st.chat_message(role):
        st.write(msg.content)

# ———————————————————————————————————————————————————————
# CHAT INPUT (single-step)
# ———————————————————————————————————————————————————————
st.text_input(
    label="Ihre Frage zu Fördermitteln…",
    key="user_text",
    on_change=handle_send,
    placeholder="Tippen und Enter drücken…"
)
