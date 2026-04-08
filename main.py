from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn, json
from rag_engine import RAGEngine, CLOUD_MODELS

app = FastAPI(title="RAG Agent API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

rag = RAGEngine()

class QueryRequest(BaseModel):
    question:           str
    provider:           str  = "ollama"   # ollama | openai | anthropic | gemini | groq
    model:              str  = ""
    api_key:            str  = ""         # sent from frontend, never stored on server
    mode:               str  = "chat"     # chat | legal
    web_search_enabled: bool = False      # enable live web search augmentation

@app.get("/")
async def root(): return {"message": "RAG Agent API v3 running"}

@app.get("/providers")
async def get_providers():
    """Return cloud provider model lists + live Ollama models"""
    ollama_models = await rag.get_ollama_models()
    return {
        "ollama":    {"models": ollama_models,           "needs_key": False},
        "openai":    {"models": CLOUD_MODELS["openai"],    "needs_key": True},
        "anthropic": {"models": CLOUD_MODELS["anthropic"], "needs_key": True},
        "gemini":    {"models": CLOUD_MODELS["gemini"],    "needs_key": True},
        "groq":      {"models": CLOUD_MODELS["groq"],      "needs_key": True},
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    return await rag.process_pdf(await file.read(), file.filename)

@app.get("/documents")
async def get_documents(): return {"documents": rag.get_documents()}

@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str): return rag.delete_document(doc_name)

@app.post("/query")
async def query(req: QueryRequest):
    # In chat mode with web search enabled, we allow querying without documents
    if not rag.has_documents() and not (req.web_search_enabled and req.mode == "chat"):
        raise HTTPException(400, "No documents uploaded. Please upload a PDF first.")

    async def generate():
        async for chunk in rag.query_stream(
            req.question, req.provider, req.model,
            req.api_key, req.mode, req.web_search_enabled
        ):
            # Pass through the special web-searching sentinel so frontend can show a status
            if chunk == "__WEB_SEARCHING__":
                yield f"data: {json.dumps({'searching': True})}\n\n"
            else:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "healthy", "ollama": await rag.check_ollama(),
            "documents_loaded": len(rag.get_documents())}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
