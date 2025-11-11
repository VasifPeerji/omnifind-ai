# src/omnifind/ui/streamlit_app.py
import streamlit as st
import requests
import tempfile
import os
from pathlib import Path

st.set_page_config(page_title='OmniFind AI Demo', layout='wide')
st.title('🛒 OmniFind AI — Multi-Modal Search Demo')

API_BASE = "http://localhost:8000"

# ---------- Sidebar ----------
with st.sidebar:
    st.header('Search Options')
    mode = st.radio('Mode', ['Text Search', 'Visual Search'])
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
        stars_min = st.number_input("Min Stars", min_value=0.0, max_value=5.0, step=0.1, value=0.0)
    with col_d:
        stars_max = st.number_input("Max Stars", min_value=0.0, max_value=5.0, step=0.1, value=5.0)

    is_best = st.checkbox("Only Best Sellers")
    
    st.markdown("---")
    st.subheader("Advanced")
    alpha = st.slider("Semantic vs Keyword Balance", 
                     min_value=0.0, max_value=1.0, value=0.6, step=0.1,
                     help="0.0 = Keyword only, 1.0 = Semantic only")
    
    # Health check
    if st.button("Check API Status"):
        try:
            resp = requests.get(f"{API_BASE}/", timeout=5)
            if resp.ok:
                health = resp.json()
                st.success(f"✅ API Healthy - {health['num_products']:,} products")
                st.info(f"Retriever: {health['retriever_type']}")
                # ← REMOVED: visual_search_available (not in API response)
            else:
                st.error(f"❌ API Error: {resp.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {e}")

# ---------- Helper Functions ----------
def parse_multi(text):
    """Parse comma-separated values"""
    text = text.strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts

# def format_price(price):
#     """Format price with USD and INR conversion"""
#     if not price or price == 0:
#         return "Price not available"
    
#     try:
#         price_usd = float(price)
#         price_inr = round(price_usd * 85, 2)
#         return f"💲 **${price_usd}**  |  ₹ **{price_inr:,}**"
#     except:
#         return f"💲 **{price}**"

def display_product_card(product, index):
    """Display product in a nice card format"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Image handling - FIX: Use correct field names
            image_url = (product.get("imgUrl") or 
                        product.get("image") or 
                        product.get("main_image") or 
                        product.get("image_url"))
            
            if image_url:
                try:
                    st.image(image_url, width=200, use_column_width=True)
                except:
                    st.image("https://via.placeholder.com/200x200/CCCCCC/666666?text=No+Image", width=200)
            else:
                st.image("https://via.placeholder.com/200x200/CCCCCC/666666?text=No+Image", width=200)
        
        with col2:
            # Title
            title = product.get("title", "Unknown Product")
            st.subheader(f"{index}. {title[:80]}{'...' if len(title) > 80 else ''}")
            
            # Scores (if available)
            score_cols = st.columns(3)
            with score_cols[0]:
                if "_score" in product:
                    st.metric("Relevance", f"{product['_score']:.3f}")
                elif "_similarity" in product:  # ← ADD: For visual search
                    st.metric("Similarity", f"{product['_similarity']:.3f}")
            with score_cols[1]:
                if "_faiss_score" in product:
                    st.metric("Semantic", f"{product['_faiss_score']:.2f}")
            with score_cols[2]:
                if "_bm25_score" in product:
                    st.metric("Keyword", f"{product['_bm25_score']:.2f}")
            
            # Product details
            st.caption(f"**Category:** {product.get('category_name', 'N/A')}")
            
            # Brand extraction (might not be in data)
            brand = product.get('brand', 'N/A')
            st.caption(f"**Brand:** {brand}")
            
            # Rating
            stars = product.get('stars', 0)
            reviews = product.get('reviews', 0)
            if stars > 0:
                st.caption(f"**Rating:** ⭐ {stars} ({reviews:,} reviews)")
            
            # Price
            # st.markdown(format_price(product.get("price")))
            #Display price only in Inr
            st.markdown(f"₹ **{product.get('price', 'N/A')}**")
            
            # Actions
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                url = product.get("productURL") or product.get("url")  # ← FIX: Field name
                if url:
                    st.markdown(f"[🔗 View on Amazon]({url})", unsafe_allow_html=True)
            with col_b:
                if product.get("asin"):
                    st.code(f"ASIN: {product['asin']}")
            with col_c:
                if product.get("isBestSeller"):
                    st.success("🏆 Best Seller")
        
        st.markdown("---")

# ---------- Text Search ----------
if mode == 'Text Search':
    st.header("🔍 Text-Based Product Search")
    
    q = st.text_input('Enter search query (e.g., "nike running shoes under $100")', 
                     placeholder="Try: 'red dress for wedding' or 'B0979NG867'")
    
    if st.button('Search Products', type="primary") and q.strip():
        with st.spinner('Searching across products...'):
            # Build filters
            filters = {
                "category_name": parse_multi(category),
                "price_min": price_min if price_min > 0 else None,
                "price_max": price_max if price_max > 0 else None,
                "stars_min": stars_min if stars_min > 0 else None,
                "stars_max": stars_max if stars_max > 0 and stars_max < 5.0 else None,
                "isBestSeller": True if is_best else None,
            }
            
            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}
            
            payload = {
                "query": q,
                "top_k": int(top_k),
                # "alpha": alpha
            }
            
            # Only add filters if they exist
            if filters:
                payload["filters"] = filters

            try:
                resp = requests.post(f"{API_BASE}/search/text", json=payload, timeout=30)
                if resp.ok:
                    data = resp.json()
                    
                    # Display search info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Results Found", len(data.get('results', [])))
                    with col2:
                        st.metric("Search Latency", f"{data.get('latency_ms', 0):.0f}ms")
                    with col3:
                        st.metric("Search Type", data.get('retriever_type', 'Unknown'))
                    
                    # ✅ Spell-corrected query
                    corrected_query = data.get("corrected_query")
                    if corrected_query and corrected_query != q:
                        st.info(f"🔍 **Corrected query:** '{q}' → '{corrected_query}'")
                    
                    # ✅ Corrected filters
                    corrected_filters = data.get("corrected_filters", {})
                    if corrected_filters:
                        filter_parts = []
                        if corrected_filters.get("price_max"):
                            filter_parts.append(f"**Price:** Under ${corrected_filters['price_max']}")
                        if corrected_filters.get("category_name"):
                            filter_parts.append(f"**Category:** {corrected_filters['category_name']}")
                        if corrected_filters.get("stars_min"):
                            filter_parts.append(f"**Min Stars:** {corrected_filters['stars_min']}+")
                        if corrected_filters.get("isBestSeller"):
                            filter_parts.append("**Best Seller Only**")
                        
                        if filter_parts:
                            st.success("🎯 " + " | ".join(filter_parts))

                    # ✅ Display results
                    results = data.get('results', [])
                    if results:
                        st.subheader(f"🎯 Top {len(results)} Results")
                        for i, product in enumerate(results, 1):
                            display_product_card(product, i)
                    else:
                        st.warning("No products found. Try broadening your search criteria.")
                        
                else:
                    st.error(f"❌ Backend error: {resp.status_code} - {resp.text}")
                    
            except requests.exceptions.Timeout:
                st.error("⏰ Search timeout - try a simpler query or reduce Top K")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to search API. Make sure the server is running:\n`uvicorn src.omnifind.api.main:app --reload`")
            except Exception as e:
                st.error(f'❌ Search error: {e}')

# ---------- Visual Search ----------
else:  # Visual Search
    st.header("📷 Visual Product Search")
    st.info("Upload an image to find similar products (Google Lens-style)")
    
    uploaded_file = st.file_uploader("Choose a product image", 
                                   type=['jpg', 'jpeg', 'png', 'webp'],
                                   help="Upload clear images of products for best results")
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Search Settings")
            visual_top_k = st.slider("Number of results", 1, 20, top_k)
            
            # FIX: Separate visual filters
            st.subheader("Filters")
            visual_price_min = st.number_input("Min Price ($)", min_value=0, value=0, key="visual_price_min")
            visual_price_max = st.number_input("Max Price ($)", min_value=0, value=0, key="visual_price_max")
    
    if st.button('Search Similar Products', type="primary") and uploaded_file is not None:
        with st.spinner('Analyzing image and finding similar products...'):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Prepare multipart form data - FIX: Correct format
                files = {
                    'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                # Form data for query parameters
                data = {
                    'top_k': visual_top_k,
                }
                
                # Add optional filters
                if visual_price_min > 0:
                    data['price_min'] = visual_price_min
                if visual_price_max > 0:
                    data['price_max'] = visual_price_max
                
                resp = requests.post(
                    f"{API_BASE}/search/image", 
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if resp.ok:
                    result = resp.json()
                    
                    # Display visual search results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Products Found", result.get("count", 0))
                    with col2:
                        st.metric("Search Latency", f"{result.get('latency_ms', 0):.0f}ms")
                    with col3:
                        st.metric("Search Type", result.get("search_type", "image"))
                    
                    # Display results
                    results = result.get("results", [])
                    if results:
                        # Show best match score
                        best_score = results[0].get("_similarity", 0) if results else 0
                        if best_score > 0.7:
                            st.success(f"🎯 Excellent match! (Score: {best_score:.3f})")
                        elif best_score > 0.5:
                            st.info(f"🔍 Good match (Score: {best_score:.3f})")
                        else:
                            st.warning(f"⚠️ Approximate match (Score: {best_score:.3f})")
                        
                        st.subheader(f"👁️ Visually Similar Products")
                        for i, product in enumerate(results, 1):
                            display_product_card(product, i)
                    else:
                        st.warning("No visually similar products found.")
                        
                else:
                    error_detail = resp.json().get("detail", "Unknown error")
                    if "not found" in str(error_detail).lower():
                        st.error("❌ Visual search is not available. Image index not built.")
                        st.info("💡 Build image index first:\n```\npython -m omnifind.embeddings.image_embedder --max-products 60000\n```")
                    else:
                        st.error(f"❌ Visual search error: {error_detail}")
                        
            except requests.exceptions.Timeout:
                st.error("⏰ Visual search timeout - try a smaller image")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to visual search API. Make sure server is running.")
            except Exception as e:
                st.error(f'❌ Visual search error: {e}')
                import traceback
                st.code(traceback.format_exc())

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by <b>OmniFind AI</b> • Hybrid Semantic + Visual Search</p>
    <p><small>Text: BGE-large-en + BM25 | Visual: CLIP-ViT-B-32</small></p>
</div>
""", unsafe_allow_html=True)