import json
from langchain_openai import ChatOpenAI

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(temperature=0)
    return _llm

def call_llm(system: str, user: str) -> dict:
    llm = get_llm()
    res = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    try:
        return json.loads(res.content)
    except Exception:
        return {}
