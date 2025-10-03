import streamlit as st
import requests

st.set_page_config(page_title='OmniFind AI Demo', layout='wide')
st.title('🛒 OmniFind AI — Demo')

API_BASE = "http://localhost:8000"

# ---------- Sidebar ----------
with st.sidebar:
    st.header('Search Options')
    mode = st.radio('Mode', ['Text', 'Image'])
    top_k = st.number_input('Top K', min_value=1, max_value=20, value=5, step=1)

    st.markdown("---")
    st.subheader("Filters (optional)")
    category = st.text_input("Category (exact or comma-separated)")

    col_a, col_b = st.columns(2)
    with col_a:
        price_min = st.number_input("Min Price ($)", min_value=0, value=0, step=1)
    with col_b:
        price_max = st.number_input("Max Price ($)", min_value=0, value=0, step=1)

    col_c, col_d = st.columns(2)
    with col_c:
        stars_min = st.number_input("Min Stars", min_value=0.0, max_value=5.0, step=0.1)
    with col_d:
        stars_max = st.number_input("Max Stars", min_value=0.0, max_value=5.0, step=0.1)

    is_best = st.checkbox("Only Best Sellers")


# ---------- Helper ----------
def parse_multi(text):
    text = text.strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts


# ---------- Main ----------
if mode == 'Text':
    q = st.text_input('Enter search query (e.g., "white cotton shirt")')
    if st.button('Search') and q.strip():
        filters = {
            "category_name": parse_multi(category),
            "price_min": price_min if price_min > 0 else None,
            "price_max": price_max if price_max > 0 else None,
            "stars_min": stars_min if stars_min > 0 else None,
            "stars_max": stars_max if stars_max > 0 else None,
            "isBestSeller": True if is_best else None,
        }
        payload = {
            "query": q,
            "top_k": int(top_k),
            "filters": filters
        }

        try:
            resp = requests.post(f"{API_BASE}/search/text", json=payload, timeout=15)
            if resp.ok:
                data = resp.json()

                # ✅ Spell-corrected query
                corrected_query = data.get("corrected_query")
                if corrected_query and corrected_query != q:
                    st.info(f"Showing results for: **{corrected_query}** (instead of '{q}')")

                # ✅ Corrected filters
                corrected_filters = data.get("corrected_filters", {})
                if corrected_filters:
                    parts = []
                    if "category_name" in corrected_filters and corrected_filters["category_name"]:
                        parts.append(f"Category → {corrected_filters['category_name']}")
                    if "price_min" in corrected_filters or "price_max" in corrected_filters:
                        pm = corrected_filters.get("price_min", "")
                        pM = corrected_filters.get("price_max", "")
                        if pm or pM:
                            parts.append(f"Price → {pm} to {pM}".strip())
                    if "stars_min" in corrected_filters or "stars_max" in corrected_filters:
                        sm = corrected_filters.get("stars_min", "")
                        sM = corrected_filters.get("stars_max", "")
                        if sm or sM:
                            parts.append(f"Stars → {sm} to {sM}".strip())
                    if "isBestSeller" in corrected_filters:
                        parts.append("Best Seller only")
                    if parts:
                        st.info("Filters applied (after correction): " + " | ".join(parts))

                # ✅ Show results in card style
                for p in data.get('results', []):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if p.get("imgUrl"):
                                st.image(p.get("imgUrl"), width=250)  # or any desired pixel width

                        with col2:
                            st.subheader(p.get("title", "Unknown Product"))
                            st.caption(f"Category: {p.get('category_name', 'N/A')}")
                            price_usd = p.get("price", 0)
                            price_inr = round(price_usd * 85, 2) if isinstance(price_usd, (int, float)) else "N/A"
                            st.markdown(f"💲 **{price_usd} USD**  |  ₹ **{price_inr} INR**")
                            if p.get("url"):
                                st.markdown(f"[🔗 View on Amazon]({p.get('url')})", unsafe_allow_html=True)
                        st.markdown("---")
            else:
                st.error(f"Backend error: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f'Error connecting to backend: {e}')
else:
    st.info("📷 Image search coming next (CLIP). For now, use Text mode.")
