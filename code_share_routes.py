"""
code_share_routes.py  –  Code snippet sharing via MongoDB
Endpoints:
  POST   /codeshare          → save snippet, returns { id, edit_key }
  GET    /codeshare/{id}     → fetch snippet
  GET    /codeshare          → list recent public snippets (last 50)
  GET    /codeshare?ids=a,b  → list only snippets matching those IDs
  PUT    /codeshare/{id}     → update snippet (requires edit_key in body)
  DELETE /codeshare/{id}     → delete snippet

Multi-file snippets: include a `files` list in the body.
Each file: { filename, language, code }
Single-file snippets (no `files`) remain fully backward-compatible.
"""

import os
import uuid
import secrets
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
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

MAX_FILES      = 20        # max files per multi-file snippet
MAX_FILE_SIZE  = 500_000   # 500 KB per file
MAX_TOTAL_SIZE = 2_000_000 # 2 MB total across all files


class SnippetFile(BaseModel):
    filename: str
    language: str = "auto"
    code:     str


class SnippetIn(BaseModel):
    title:    str = "Untitled Snippet"
    language: str = "auto"          # used for single-file mode
    code:     str = ""              # used for single-file mode
    author:   str = "Anonymous"
    files:    Optional[List[SnippetFile]] = None  # multi-file mode


class SnippetUpdate(BaseModel):
    edit_key: str
    title:    str = "Untitled Snippet"
    language: str = "auto"
    code:     str = ""
    author:   str = "Anonymous"
    files:    Optional[List[SnippetFile]] = None


class SnippetFileOut(BaseModel):
    filename: str
    language: str
    code:     str


class SnippetOut(BaseModel):
    id:         str
    title:      str
    language:   str
    code:       str
    author:     str
    created_at: str
    views:      int
    files:      Optional[List[SnippetFileOut]] = None  # None = single-file


class SnippetCreated(BaseModel):
    id:       str
    edit_key: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_files(files: List[SnippetFile]) -> List[dict]:
    """Validate and normalise a list of SnippetFile objects."""
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Too many files. Maximum is {MAX_FILES}.")

    total = 0
    result = []
    for f in files:
        lang = f.language.lower()
        if lang not in ALLOWED_LANGS:
            raise HTTPException(400, f"Language '{lang}' not supported.")
        fname = (f.filename or "untitled").strip()[:120]
        if not fname:
            fname = "untitled"
        code = f.code[:MAX_FILE_SIZE]
        total += len(code.encode("utf-8"))
        if total > MAX_TOTAL_SIZE:
            raise HTTPException(413, "Total snippet size exceeds 2 MB limit.")
        result.append({"filename": fname, "language": lang, "code": code})
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@code_share_router.post("", status_code=201, response_model=SnippetCreated)
async def create_snippet(body: SnippetIn):
    edit_key = "ek_" + secrets.token_hex(8)

    # Multi-file mode
    if body.files is not None and len(body.files) > 0:
        validated_files = _validate_files(body.files)
        # Derive top-level language/code from first file for backward compat
        primary_lang = validated_files[0]["language"]
        primary_code = validated_files[0]["code"]
        doc = {
            "_id":        str(uuid.uuid4())[:8],
            "title":      body.title[:120],
            "language":   primary_lang,
            "code":       primary_code,
            "author":     body.author[:60],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "views":      0,
            "edit_key":   edit_key,
            "files":      validated_files,
        }
    else:
        # Single-file mode (original behaviour)
        lang = body.language.lower()
        if lang not in ALLOWED_LANGS:
            raise HTTPException(400, f"Language '{lang}' not supported.")
        doc = {
            "_id":        str(uuid.uuid4())[:8],
            "title":      body.title[:120],
            "language":   lang,
            "code":       body.code[:MAX_FILE_SIZE],
            "author":     body.author[:60],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "views":      0,
            "edit_key":   edit_key,
            "files":      None,
        }

    await snippets.insert_one(doc)
    return {"id": doc["_id"], "edit_key": edit_key}


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
    doc.pop("edit_key", None)
    return doc


@code_share_router.get("", response_model=list[SnippetOut])
async def list_snippets(ids: Optional[str] = Query(default=None)):
    if ids:
        id_list = [i.strip() for i in ids.split(",") if i.strip()]
        if not id_list:
            return []
        query  = {"_id": {"$in": id_list}}
        cursor = snippets.find(query).sort("created_at", -1)
    else:
        cursor = snippets.find({}).sort("created_at", -1).limit(50)

    results = []
    async for doc in cursor:
        doc["id"] = doc.pop("_id")
        doc.pop("edit_key", None)
        results.append(doc)
    return results


@code_share_router.put("/{snippet_id}", response_model=SnippetOut)
async def update_snippet(snippet_id: str, body: SnippetUpdate):
    existing = await snippets.find_one({"_id": snippet_id})
    if not existing:
        raise HTTPException(404, "Snippet not found.")
    if existing.get("edit_key") != body.edit_key:
        raise HTTPException(401, "Invalid edit key.")

    if body.files is not None and len(body.files) > 0:
        validated_files = _validate_files(body.files)
        primary_lang = validated_files[0]["language"]
        primary_code = validated_files[0]["code"]
        updates = {
            "title":    body.title[:120],
            "language": primary_lang,
            "code":     primary_code,
            "author":   body.author[:60],
            "files":    validated_files,
        }
    else:
        lang = body.language.lower()
        if lang not in ALLOWED_LANGS:
            raise HTTPException(400, f"Language '{lang}' not supported.")
        updates = {
            "title":    body.title[:120],
            "language": lang,
            "code":     body.code[:MAX_FILE_SIZE],
            "author":   body.author[:60],
            "files":    None,
        }

    updated = await snippets.find_one_and_update(
        {"_id": snippet_id},
        {"$set": updates},
        return_document=True,
    )
    updated["id"] = updated.pop("_id")
    updated.pop("edit_key", None)
    return updated


@code_share_router.delete("/{snippet_id}")
async def delete_snippet(snippet_id: str):
    result = await snippets.delete_one({"_id": snippet_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Snippet not found.")
    return {"deleted": True}