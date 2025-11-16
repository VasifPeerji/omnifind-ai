"""
OmniFind AI - Modern Multi-Modal Search UI

Features:
- Text search with query understanding
- Image-to-image search (upload photo)
- Text-to-image search (visual semantic search)
- Hybrid image+text search
- Advanced filtering
- Real-time metrics
- Modern, attractive UI
"""
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title='OmniFind AI', layout='wide', page_icon='🛒')

# Modern CSS styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #FF9900 0%, #FF6B00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF9900 0%, #FF6B00 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 153, 0, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 153, 0, 0.3);
    }
    
    /* Card styling */
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        transform: translateY(-4px);
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* File uploader */
    .stFileUploader {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        border: 2px dashed #dee2e6;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #FF9900;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛒 OmniFind AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Next-Gen Multi-Modal Product Search Engine</p>', unsafe_allow_html=True)

API_BASE = "http://localhost:8000"

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("### ⚙️ Search Settings")
    
    # Mode selection
    mode = st.selectbox(
        '🔍 Search Mode',
        ['Text Search', 'Image Search', 'Visual Text Search', 'Hybrid Search'],
        help="Choose your search method"
    )
    
    top_k = st.slider('📊 Results Count', 1, 20, 5)
    
    # ===== Filters =====
    st.markdown("---")
    st.markdown("### 🎯 Filters")
    
    with st.expander("💰 Price Range", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            price_min = st.number_input("Min (₹)", min_value=0, value=0, step=100)
        with col_b:
            price_max = st.number_input("Max (₹)", min_value=0, value=0, step=100)
    
    with st.expander("⭐ Rating", expanded=False):
        col_c, col_d = st.columns(2)
        with col_c:
            stars_min = st.number_input("Min Stars", 0.0, 5.0, 0.0, 0.5)
        with col_d:
            stars_max = st.number_input("Max Stars", 0.0, 5.0, 5.0, 0.5)
    
    category = st.text_input("📁 Category", placeholder="Electronics, Fashion...")
    is_best = st.checkbox("🏆 Best Sellers Only")
    
    # ===== Advanced Settings =====
    st.markdown("---")
    st.markdown("### ⚡ Advanced")
    
    if mode == 'Hybrid Search':
        alpha = st.slider(
            "🎚️ Image Weight", 0.0, 1.0, 0.5, 0.1,
            help="Balance between image and text"
        )
    
    # ===== API Status =====
    st.markdown("---")
    if st.button("🔌 API Status", key="api_check"):
        try:
            resp = requests.get(f"{API_BASE}/", timeout=5)
            if resp.ok:
                health = resp.json()
                st.success("✅ Connected")
                st.metric("Products", f"{health['num_products']:,}")
                visual_icon = "✅" if health.get('visual_search_available') else "❌"
                st.caption(f"Visual Search: {visual_icon}")
            else:
                st.error(f"❌ Error: {resp.status_code}")
        except Exception as e:
            st.error("❌ Disconnected")

# ==================== Helper Functions ====================
def build_filters():
    """Build filters dict from sidebar inputs"""
    filters = {}
    if price_min > 0:
        filters["price_min"] = price_min
    if price_max > 0:
        filters["price_max"] = price_max
    if stars_min > 0:
        filters["stars_min"] = stars_min
    if stars_max < 5.0:
        filters["stars_max"] = stars_max
    if category.strip():
        filters["category_name"] = category.strip()
    if is_best:
        filters["isBestSeller"] = True
    return filters

def display_product_card(product, index):
    """Display product in modern card format"""
    with st.container():
        # Create card wrapper
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Image handling - FIX: Proper field extraction
            img_url = (
                product.get("imgUrl") or 
                product.get("image") or 
                product.get("main_image") or
                product.get("image_url")
            )
            
            if img_url and img_url.strip():
                try:
                    # Use width parameter instead of use_container_width
                    st.image(img_url, width=200)
                except Exception as e:
                    st.image("https://via.placeholder.com/200x200/EEEEEE/999999?text=No+Image", width=200)
            else:
                st.image("https://via.placeholder.com/200x200/EEEEEE/999999?text=No+Image", width=200)
        
        with col2:
            # Title with emoji
            title = product.get("title", "Unknown Product")
            st.markdown(f"### {index}. {title[:80]}{'...' if len(title) > 80 else ''}")
            
            # Scores in columns
            score_cols = st.columns(3)
            with score_cols[0]:
                if "_score" in product:
                    st.metric("🎯 Relevance", f"{product['_score']:.3f}")
                elif "_similarity" in product:
                    st.metric("🎯 Similarity", f"{product['_similarity']:.3f}")
            with score_cols[1]:
                if "_faiss_score" in product:
                    st.metric("🧠 Semantic", f"{product['_faiss_score']:.2f}")
            with score_cols[2]:
                if "_bm25_score" in product:
                    st.metric("🔤 Keyword", f"{product['_bm25_score']:.2f}")
            
            # Product details
            detail_cols = st.columns(3)
            
            with detail_cols[0]:
                cat = product.get('category_name', 'N/A')
                st.caption(f"📁 **Category**")
                st.caption(cat[:25] + "..." if len(cat) > 25 else cat)
            
            with detail_cols[1]:
                stars = product.get('stars', 0)
                reviews = product.get('reviews', 0)
                if stars > 0:
                    st.caption(f"⭐ **Rating**")
                    st.caption(f"{stars} ({reviews:,} reviews)")
            
            with detail_cols[2]:
                price = product.get('price', 'N/A')
                st.caption(f"💰 **Price**")
                st.caption(f"₹{price}")
            
            # Actions
            col_act = st.columns([2, 1, 1])
            with col_act[0]:
                url = product.get("productURL") or product.get("url")
                if url:
                    st.markdown(f"[🛒 **View on Amazon**]({url})")
            with col_act[1]:
                asin = product.get("asin")
                if asin:
                    st.code(asin, language=None)
            with col_act[2]:
                if product.get("isBestSeller"):
                    st.success("🏆")
        
        st.markdown("---")

def display_search_metrics(data):
    """Display search metrics in modern cards"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Results", data.get('count', 0))
    with col2:
        st.metric("⚡ Latency", f"{data.get('latency_ms', 0):.0f}ms")
    with col3:
        engine = data.get('retriever_type', 'Unknown')
        st.metric("🔧 Engine", engine.replace('_', ' ').title()[:15])

# ==================== TEXT SEARCH ====================
if mode == 'Text Search':
    st.markdown("## 🔍 Text Search")
    st.caption("Natural language queries with AI-powered understanding")
    
    query = st.text_input(
        'Search Query',
        placeholder='Try: "nike running shoes", "red dress under $50", or "B0979NG867"',
        help="Type anything - we understand natural language!",
        label_visibility="collapsed"
    )
    
    search_btn = st.button('🚀 Search Products', type='primary', use_container_width=True)
    
    if search_btn and query.strip():
        with st.spinner('🔄 Searching...'):
            filters = build_filters()
            
            payload = {
                "query": query,
                "top_k": top_k,
            }
            if filters:
                payload["filters"] = filters
            
            try:
                resp = requests.post(f"{API_BASE}/search/text", json=payload, timeout=30)
                
                if resp.ok:
                    data = resp.json()
                    
                    # Metrics
                    display_search_metrics(data)
                    
                    # Query corrections
                    corrected = data.get("corrected_query")
                    if corrected and corrected != query:
                        st.info(f"🔍 **Did you mean:** {corrected}")
                    
                    # Filter suggestions
                    corr_filters = data.get("corrected_filters", {})
                    if corr_filters:
                        filter_parts = []
                        if corr_filters.get("price_max"):
                            filter_parts.append(f"💰 Under ₹{corr_filters['price_max']}")
                        if corr_filters.get("category_name"):
                            filter_parts.append(f"📁 {corr_filters['category_name']}")
                        if corr_filters.get("stars_min"):
                            filter_parts.append(f"⭐ {corr_filters['stars_min']}+")
                        if filter_parts:
                            st.success("🎯 **Auto-detected:** " + " | ".join(filter_parts))
                    
                    # Results
                    results = data.get('results', [])
                    if results:
                        st.markdown(f"### 📦 Top {len(results)} Results")
                        for i, prod in enumerate(results, 1):
                            display_product_card(prod, i)
                    else:
                        st.warning("😕 No products found. Try different keywords.")
                else:
                    st.error(f"❌ Error: {resp.status_code}")
            
            except requests.exceptions.Timeout:
                st.error("⏰ Request timeout")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==================== IMAGE SEARCH ====================
elif mode == 'Image Search':
    st.markdown("## 📷 Image Search")
    st.caption("Upload a product photo to find similar items (Google Lens style)")
    
    uploaded_file = st.file_uploader(
        "Upload Product Image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Best results with clear product images",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        col_img, col_settings = st.columns([1, 1])
        
        with col_img:
            image = Image.open(uploaded_file)
            # FIX: Use width instead of use_container_width
            st.image(image, caption="Query Image", width=400)
        
        with col_settings:
            st.markdown("### ⚙️ Search Settings")
            img_top_k = st.slider("Results", 1, 20, top_k, key="img_k")
            
            filters = build_filters()
            if filters:
                st.success(f"✅ {len(filters)} filters active")
        
        if st.button('🔍 Find Similar Products', type='primary', use_container_width=True):
            with st.spinner('🔄 Analyzing image...'):
                try:
                    # Prepare file upload
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='JPEG')
                    img_bytes.seek(0)
                    
                    files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
                    data = {'top_k': img_top_k}
                    
                    # Add filters
                    if price_min > 0:
                        data['price_min'] = price_min
                    if price_max > 0:
                        data['price_max'] = price_max
                    if category.strip():
                        data['category_name'] = category.strip()
                    if stars_min > 0:
                        data['stars_min'] = stars_min
                    
                    resp = requests.post(
                        f"{API_BASE}/search/image",
                        files=files,
                        data=data,
                        timeout=60
                    )
                    
                    if resp.ok:
                        result = resp.json()
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📦 Found", result.get('count', 0))
                        with col2:
                            st.metric("⚡ Time", f"{result.get('latency_ms', 0):.0f}ms")
                        with col3:
                            tta = result.get('tta_enabled', False)
                            st.metric("🎯 TTA", "✅" if tta else "❌")
                        
                        # Match quality
                        results = result.get('results', [])
                        if results:
                            best_score = results[0].get('_similarity', 0)
                            if best_score > 0.7:
                                st.success(f"🎯 **Excellent match!** (Score: {best_score:.3f})")
                            elif best_score > 0.5:
                                st.info(f"👍 **Good match** (Score: {best_score:.3f})")
                            else:
                                st.warning(f"⚠️ **Approximate match** (Score: {best_score:.3f})")
                            
                            st.markdown("### 👁️ Visually Similar Products")
                            for i, prod in enumerate(results, 1):
                                display_product_card(prod, i)
                        else:
                            st.warning("😕 No similar products found")
                    else:
                        error = resp.json().get('detail', 'Unknown error')
                        if "not available" in error.lower():
                            st.error("❌ Visual search not available")
                            st.code("python -m omnifind.embeddings.image_embedder")
                        else:
                            st.error(f"❌ {error}")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ==================== VISUAL TEXT SEARCH ====================
elif mode == 'Visual Text Search':
    st.markdown("## 🎨 Visual Text Search")
    st.caption("Describe what you want to see - find products that LOOK like your description")
    
    st.info("💡 **CLIP-powered:** We find products that visually match your text, not just keywords!")
    
    visual_query = st.text_input(
        'Visual Description',
        placeholder='e.g., "red evening dress with sequins", "blue denim jacket", "leather boots"',
        help="Describe the visual appearance",
        label_visibility="collapsed"
    )
    
    if st.button('🔍 Search Visually', type='primary', use_container_width=True) and visual_query.strip():
        with st.spinner('🔄 Finding visually matching products...'):
            filters = build_filters()
            
            payload = {
                "text": visual_query,
                "top_k": top_k,
            }
            if price_min > 0:
                payload["price_min"] = price_min
            if price_max > 0:
                payload["price_max"] = price_max
            if category.strip():
                payload["category_name"] = category.strip()
            if stars_min > 0:
                payload["stars_min"] = stars_min
            
            try:
                resp = requests.post(f"{API_BASE}/search/visual-text", json=payload, timeout=30)
                
                if resp.ok:
                    data = resp.json()
                    
                    # Metrics
                    display_search_metrics(data)
                    
                    # Results
                    results = data.get('results', [])
                    if results:
                        st.markdown(f"### 🎨 Visually Matching Products")
                        st.caption(f"*Products that look like: '{visual_query}'*")
                        for i, prod in enumerate(results, 1):
                            display_product_card(prod, i)
                    else:
                        st.warning("😕 No visually matching products found")
                else:
                    error = resp.json().get('detail', 'Unknown error')
                    if "not available" in error.lower():
                        st.error("❌ Visual search not available")
                        st.code("python -m omnifind.embeddings.image_embedder")
                    else:
                        st.error(f"❌ {error}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==================== HYBRID SEARCH ====================
else:  # Hybrid Search
    st.markdown("## 🔀 Hybrid Search")
    st.caption("Combine image + text for ultra-precise results")
    
    st.info("💡 **Best of both worlds:** Upload an image AND add text refinements!")
    
    col_upload, col_text = st.columns([1, 1])
    
    with col_upload:
        hybrid_file = st.file_uploader(
            "Upload Base Image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="hybrid_upload",
            label_visibility="collapsed"
        )
        
        if hybrid_file:
            image = Image.open(hybrid_file)
            # FIX: Use width
            st.image(image, caption="Query Image", width=350)
    
    with col_text:
        hybrid_text = st.text_area(
            "Refinement Text",
            placeholder='e.g., "nike", "formal", "red color"',
            help="Add text to refine the image search",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🎚️ Weight Balance")
        hybrid_alpha = st.slider(
            "Image vs Text",
            0.0, 1.0, 0.5, 0.1,
            help="Adjust the balance",
            label_visibility="collapsed"
        )
        
        # Visual indicator with progress bar
        img_pct = int(hybrid_alpha * 100)
        txt_pct = 100 - img_pct
        st.caption(f"📷 Image: **{img_pct}%** | 📝 Text: **{txt_pct}%**")
        st.progress(hybrid_alpha)
    
    if st.button('🔍 Hybrid Search', type='primary', use_container_width=True) and hybrid_file:
        with st.spinner('🔄 Searching with image + text...'):
            try:
                # Prepare file
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
                data = {
                    'text': hybrid_text.strip(),
                    'top_k': top_k,
                    'alpha': hybrid_alpha,
                }
                
                # Add filters
                if price_min > 0:
                    data['price_min'] = price_min
                if price_max > 0:
                    data['price_max'] = price_max
                
                resp = requests.post(
                    f"{API_BASE}/search/hybrid-visual",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if resp.ok:
                    result = resp.json()
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📦 Results", result.get('count', 0))
                    with col2:
                        st.metric("⚡ Time", f"{result.get('latency_ms', 0):.0f}ms")
                    with col3:
                        st.metric("🎚️ Alpha", f"{result.get('alpha', 0):.1f}")
                    with col4:
                        text_used = "✅" if result.get('text_query') else "❌"
                        st.metric("📝 Text", text_used)
                    
                    # Results
                    results = result.get('results', [])
                    if results:
                        st.markdown("### 🔀 Hybrid Results")
                        if hybrid_text.strip():
                            st.caption(f"*Combining image with: '{hybrid_text}'*")
                        for i, prod in enumerate(results, 1):
                            display_product_card(prod, i)
                    else:
                        st.warning("😕 No matching products found")
                else:
                    error = resp.json().get('detail', 'Unknown error')
                    st.error(f"❌ {error}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;'>
    <h3 style='color: #FF9900; margin-bottom: 1rem;'>⚡ OmniFind AI v3.0</h3>
    <p style='color: #666; margin-bottom: 0.5rem;'><b>Production Multi-Modal Search Engine</b></p>
    <p style='font-size: 0.9rem; color: #999;'>
        Text: BGE-large-en + BM25 | Visual: CLIP-ViT-L-14 + TTA
    </p>
    <p style='font-size: 0.85rem; color: #aaa; margin-top: 1rem;'>
        Built with ❤️ using Streamlit • FastAPI • PyTorch
    </p>
</div>
""", unsafe_allow_html=True)