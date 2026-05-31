"""
code_share_routes.py  –  Code snippet sharing via MongoDB
Endpoints:
  POST /codeshare          → save snippet, returns { id }
  GET  /codeshare/{id}     → fetch snippet
  GET  /codeshare          → list recent public snippets (last 50)
"""

import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

# ── Mongo setup ───────────────────────────────────────────────────────────────
MONGO_URI1 = os.environ.get("MONGO_URI1", "mongodb://localhost:27017")
_client   = AsyncIOMotorClient(MONGO_URI1)
_db       = _client[os.environ.get("MONGO_DB", "qna_ai")]
snippets  = _db["code_snippets"]

code_share_router = APIRouter(prefix="/codeshare", tags=["codeshare"])

# ── Schema ────────────────────────────────────────────────────────────────────
ALLOWED_LANGS = {
    "auto", "python", "javascript", "jsx", "typescript", "bash",
    "c", "cpp", "csharp", "go", "css", "html", "java", "json",
    "kotlin", "php", "ruby", "rust", "sql", "vue", "yaml", "swift",
    "text",
}


class SnippetIn(BaseModel):
    title:    str  = "Untitled Snippet"
    language: str  = "auto"
    code:     str
    author:   str  = "Anonymous"


class SnippetOut(BaseModel):
    id:        str
    title:     str
    language:  str
    code:      str
    author:    str
    created_at: str
    views:     int


# ── Routes ────────────────────────────────────────────────────────────────────

@code_share_router.post("", status_code=201)
async def create_snippet(body: SnippetIn):
    lang = body.language.lower()
    if lang not in ALLOWED_LANGS:
        raise HTTPException(400, f"Language '{lang}' not supported.")

    doc = {
        "_id":        str(uuid.uuid4())[:8],   # short 8-char ID
        "title":      body.title[:120],
        "language":   lang,
        "code":       body.code[:500_000],      # 500 KB cap
        "author":     body.author[:60],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "views":      0,
    }
    await snippets.insert_one(doc)
    return {"id": doc["_id"]}


@code_share_router.get("/{snippet_id}", response_model=SnippetOut)
async def get_snippet(snippet_id: str):
    doc = await snippets.find_one_and_update(
        {"_id": snippet_id},
        {"$inc": {"views": 1}},
        return_document=True,
    )
    if not doc:
        raise HTTPException(404, "Snippet not found.")
    doc["id"] = doc.pop("_id")
    return doc


@code_share_router.get("", response_model=list[SnippetOut])
async def list_snippets():
    cursor = snippets.find({}).sort("created_at", -1).limit(50)
    results = []
    async for doc in cursor:
        doc["id"] = doc.pop("_id")
        results.append(doc)
    return results


@code_share_router.delete("/{snippet_id}")
async def delete_snippet(snippet_id: str):
    result = await snippets.delete_one({"_id": snippet_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Snippet not found.")
    return {"deleted": True}
