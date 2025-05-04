import streamlit as st
import json
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load your OPENAI_API_KEY from .env
load_dotenv()

def load_links(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_vectorstore_from_urls(urls: list[str]) -> Chroma:
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(chunks, embeddings)

def get_context_retriever_chain(vector_store: Chroma):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    # Strict “only-from-context” system prompt:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI assistant. Answer the user's question ONLY using the context below. "
         "If the answer is not present in the context, reply exactly \"I don't know.\"\n\n"
         "Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, docs_chain)

def get_response(user_input: str) -> str:
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)
    output = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return output["answer"]

# ——— Streamlit App ———

st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Websites")

# 1) Load links.json once
if "links" not in st.session_state:
    st.session_state.links = load_links("links.json")

# 2) Build vector store once
if "vector_store" not in st.session_state:
    with st.spinner("🔎 Ingesting your URLs... this may take a minute"):
        st.session_state.vector_store = get_vectorstore_from_urls(st.session_state.links)

# 3) Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I have processed all your provided websites. How can I help you today?")
    ]

# 4) Handle user input
user_query = st.chat_input("Type your message here…")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    answer = get_response(user_query)
    st.session_state.chat_history.append(AIMessage(content=answer))

# 5) Render the chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("human"):
            st.write(msg.content)
    else:
        with st.chat_message("ai"):
            st.write(msg.content)
