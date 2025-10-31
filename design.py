# design.py
import streamlit as st

def init_chat_css():
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }
    .chat-bubble {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        max-width: 80%;
        word-break: break-word;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .chat-ai {
        background: #F4F4F5;
        color: #000;
        align-self: flex-start;
        border: 1px solid #DDD;
    }
    .chat-user {
        background: #DCF7C5;
        color: #000;
        align-self: flex-end;
        border: 1px solid #C8E6A0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_chat(chat_history):
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for role, msg in chat_history:
        css = "chat-ai" if role == "ai" else "chat-user"
        st.markdown(
            f'<div class="chat-bubble {css}">{msg}</div>',
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
