SYSTEM_PROMPT = '''You are a concise AI assistant. Answer using the supplied context. If unsure, say you don't know.'''

def build_prompt(query: str, contexts):
    ctx = "\n\n".join([f"[{c.doc_id}]\n{c.text}" for c in contexts])
    return f"""{SYSTEM_PROMPT}

Context:
{ctx}

User question: {query}

Answer:""""
