from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from omnifind.retrieval.retriever import RetrieverService
from omnifind.logger import get_logger

logger = get_logger('api.main')

app = FastAPI(
    title="OmniFind AI API",
    version="0.1.0",
    description="Backend API for OmniFind AI - text and image retrieval"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

retriever = RetrieverService()

@app.get('/')
def read_root():
    return {'message': 'OmniFind AI backend is running'}

@app.post('/search/text')
def search_text(query: str):
    results = retriever.search_text(query, top_k=5)
    return {'query': query, 'results': results}

@app.post('/search/image')
async def search_image(file: UploadFile = File(...)):
    content = await file.read()
    results = retriever.search_image(content, top_k=5)
    return {'results': results}
