import streamlit as st
import requests
import asyncio
import websockets
import json

API = st.secrets.get('API_URL', 'http://localhost:8000')

st.set_page_config(page_title='GenAI & NLP Analytics', layout='wide')
st.title('🧠 GenAI & NLP Analytics — Demo')

tab1, tab2 = st.tabs(['🔎 RAG Chat', '📈 Financial Analytics'])

with tab1:
    st.subheader('Ask a question (RAG)')
    q = st.text_input('Your query', 'Explain RAG vs fine-tuning for finance.')
    if st.button('Submit'):
        r = requests.post(f'{API}/rag/query', json={'query': q, 'k': 3}, timeout=30)
        if r.ok:
            data = r.json()
            st.markdown('**Answer**')
            st.write(data['answer'])
            with st.expander('Context'):
                for c in data['context']:
                    st.markdown(f"**{c['doc_id']}**\n\n{c['text']}")
        else:
            st.error(r.text)

    st.divider()
    st.subheader('Live Generation (WebSocket)')
    prompt = st.text_input('Prompt', 'Tell me a joke about distributed systems.')
    if st.button('Stream it'):
        placeholder = st.empty()
        async def run_ws():
            try:
                async with websockets.connect(API.replace('http', 'ws') + '/ws/generate') as ws:
                    await ws.send(prompt)
                    buf = ''
                    while True:
                        msg = await ws.recv()
                        buf += msg
                        placeholder.write(buf)
                        if msg.endswith('[END]'):
                            break
            except Exception as e:
                st.error(str(e))
        asyncio.run(run_ws())

with tab2:
    st.subheader('Signals')
    symbol = st.text_input('Symbol', 'AAPL')
    n = st.slider('Bars', 100, 1000, 200, 50)
    if st.button('Compute signals'):
        r = requests.post(f'{API}/finance/signal', json={'symbol': symbol, 'n': n}, timeout=30)
        if r.ok:
            data = r.json()
            st.write('**Preview (last 10 bars)**')
            st.dataframe(data['preview'])
            st.json(data['features'])
        else:
            st.error(r.text)

st.caption('Prototype for education/demo. Replace the model and data sources for production use.')
