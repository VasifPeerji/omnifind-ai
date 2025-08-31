import streamlit as st
import requests

st.set_page_config(page_title='OmniFind AI Demo', layout='wide')
st.title('OmniFind AI — Demo')

API_BASE = "http://localhost:8000"

with st.sidebar:
    st.header('Search')
    mode = st.radio('Mode', ['Text', 'Image'])
    top_k = st.number_input('Top K', min_value=1, max_value=20, value=5, step=1)

    st.markdown("---")
    st.subheader("Filters (optional)")
    brand = st.text_input("Brand (exact, or comma-separated)")
    category = st.text_input("Category (exact, or comma-separated)")
    col_a, col_b = st.columns(2)
    with col_a:
        price_min = st.number_input("Min Price", min_value=0, value=0, step=1)
    with col_b:
        price_max = st.number_input("Max Price", min_value=0, value=0, step=1)


def parse_multi(text):
    text = text.strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parts


if mode == 'Text':
    q = st.text_input('Enter search query (e.g., "white cotton shirt")')
    if st.button('Search'):
        filters = {
            "brand": parse_multi(brand),
            "category": parse_multi(category),
            "price_min": price_min if price_min > 0 else None,
            "price_max": price_max if price_max > 0 else None
        }
        payload = {
            "query": q,
            "top_k": int(top_k),
            "filters": filters
        }
        try:
            resp = requests.post(f"{API_BASE}/search/text", json=payload, timeout=10)
            if resp.ok:
                data = resp.json()

                # ✅ Show spell correction if applied
                corrected_query = data.get("corrected_query")
                if corrected_query and corrected_query != q:
                    st.info(f"Showing results for: **{corrected_query}** (instead of '{q}')")

                for p in data.get('results', []):
                    st.write(f"**{p.get('title')}** — {p.get('brand')} — ₹{p.get('price')}")
                    st.write(p.get('url'))
                    st.caption(f"{p.get('category','')} | {p.get('color','')} {p.get('material','')}")
                    st.markdown('---')
            else:
                st.error(f"Backend error: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f'Error connecting to backend: {e}')
else:
    st.info("Image search coming next (CLIP). For now, use Text mode.")
