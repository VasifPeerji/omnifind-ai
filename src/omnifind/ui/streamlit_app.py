import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title='OmniFind AI Demo', layout='wide')
st.title('OmniFind AI — Demo')

with st.sidebar:
    st.header('Search')
    mode = st.radio('Mode', ['Text', 'Image'])

if mode == 'Text':
    q = st.text_input('Enter search query (e.g., "white cotton shirt")')
    if st.button('Search'):
        try:
            resp = requests.post('http://localhost:8000/search/text', params={'query': q}, timeout=5)
        except Exception as e:
            st.error(f'Error connecting to backend: {e}')
            resp = None
        if resp and resp.ok:
            data = resp.json()
            for p in data.get('results', []):
                st.write(f"**{p.get('title')}** — {p.get('brand')} — ₹{p.get('price')}")
                st.write(p.get('url'))
                st.markdown('---')
        else:
            st.error('Backend error or no results')
else:
    uploaded = st.file_uploader('Upload image', type=['jpg','png'])
    if uploaded is not None and st.button('Search image'):
        files = {'file': (uploaded.name, uploaded.getvalue())}
        try:
            resp = requests.post('http://localhost:8000/search/image', files=files, timeout=10)
        except Exception as e:
            st.error(f'Error connecting to backend: {e}')
            resp = None
        if resp and resp.ok:
            data = resp.json()
            for p in data.get('results', []):
                st.write(f"**{p.get('title')}** — {p.get('brand')} — ₹{p.get('price')}")
                st.write(p.get('url'))
                st.markdown('---')
        else:
            st.error('Backend error or no results')
