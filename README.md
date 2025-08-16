# OmniFind AI

OmniFind AI is a **multimodal product discovery copilot** that combines **Vision + RAG + Personalization** for intelligent product search across diverse domains.  
This repository provides an **industry-style starter skeleton** for building production-grade multimodal retrieval systems.

---

## 🚀 Quickstart (Development)

```bash
# 1. Create environment (conda recommended)
conda create -n omnifind python=3.10 -y
conda activate omnifind

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run backend (FastAPI)
uvicorn src.omnifind.api.main:app --reload

# 4. Run frontend (Streamlit) in another terminal
streamlit run src/omnifind/ui/streamlit_app.py
