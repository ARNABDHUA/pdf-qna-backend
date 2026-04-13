"""
collab_routes.py — Collaborative Session API
Plug this into your existing FastAPI app with:
    from collab_routes import collab_router
    app.include_router(collab_router)
"""

import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

# ── MongoDB connection ─────────────────────────────────────────────────────────
MONGO_URL = os.getenv(
    "MONGODB_URL",
    ""
)

_mongo_client: Optional[AsyncIOMotorClient] = None

def get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=10000)
    return _mongo_client["qna_ai_collab"]


# ── Pydantic models ────────────────────────────────────────────────────────────

class CollabMessage(BaseModel):
    role: str                   # "user" | "ai"
    content: str
    author: Optional[str] = "Anonymous"
    provider: Optional[str] = None
    model: Optional[str] = None
    mode: Optional[str] = "chat"
    timestamp: Optional[float] = None
    msg_id: Optional[str] = None

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "Collaborative Session"
    owner: Optional[str] = "Anonymous"
    messages: Optional[List[dict]] = []

class AddMessageRequest(BaseModel):
    session_id: str
    message: CollabMessage

class PatchTitleRequest(BaseModel):
    title: str

# ── Router ─────────────────────────────────────────────────────────────────────
collab_router = APIRouter(prefix="/collab", tags=["Collaborative"])


@collab_router.post("/sessions")
async def create_collab_session(req: CreateSessionRequest):
    """Create a new collaborative session and return its share link ID."""
    db = get_db()
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()

    doc = {
        "_id":        session_id,
        "title":      req.title or "Collaborative Session",
        "owner":      req.owner or "Anonymous",
        "messages":   req.messages or [],
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(days=7),   # auto-expire after 7 days
        "participants": [],
        "active_users": 0,
    }

    try:
        await db.sessions.insert_one(doc)
    except Exception as e:
        raise HTTPException(500, f"Failed to create session: {e}")

    return {
        "session_id": session_id,
        "share_url":  f"/collab/{session_id}",
        "expires_at": doc["expires_at"].isoformat(),
    }


@collab_router.get("/sessions/{session_id}")
async def get_collab_session(session_id: str):
    """Fetch a collaborative session by ID."""
    db = get_db()
    doc = await db.sessions.find_one({"_id": session_id})
    if not doc:
        raise HTTPException(404, "Session not found or expired")

    # Check expiry
    if doc.get("expires_at") and datetime.utcnow() > doc["expires_at"]:
        raise HTTPException(410, "Session has expired")

    doc["id"] = doc.pop("_id")
    # Convert datetime objects
    for key in ("created_at", "updated_at", "expires_at"):
        if key in doc and isinstance(doc[key], datetime):
            doc[key] = doc[key].isoformat()

    return doc


@collab_router.post("/sessions/{session_id}/messages")
async def add_message_to_session(session_id: str, req: AddMessageRequest):
    """Append a message to a collaborative session."""
    db = get_db()
    doc = await db.sessions.find_one({"_id": session_id})
    if not doc:
        raise HTTPException(404, "Session not found")

    msg = req.message.dict()
    msg["timestamp"] = msg.get("timestamp") or datetime.utcnow().timestamp()
    msg["msg_id"]    = msg.get("msg_id") or str(uuid.uuid4())

    result = await db.sessions.update_one(
        {"_id": session_id},
        {
            "$push":  {"messages": msg},
            "$set":   {"updated_at": datetime.utcnow()},
        }
    )
    if result.modified_count == 0:
        raise HTTPException(500, "Failed to add message")

    return {"ok": True, "msg_id": msg["msg_id"]}


@collab_router.patch("/sessions/{session_id}/messages/{msg_id}")
async def update_message_in_session(session_id: str, msg_id: str, req: Request):
    """Stream-update an AI message (append chunks to content)."""
    db = get_db()
    body = await req.json()
    content = body.get("content", "")

    result = await db.sessions.update_one(
        {"_id": session_id, "messages.msg_id": msg_id},
        {
            "$set": {
                "messages.$.content": content,
                "updated_at": datetime.utcnow(),
            }
        }
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Message not found")

    return {"ok": True}


@collab_router.get("/sessions/{session_id}/poll")
async def poll_session(session_id: str, since: Optional[float] = None):
    """
    Long-poll endpoint — returns new messages since `since` (unix timestamp).
    Waits up to 20s for new content, then returns empty if nothing.
    """
    db = get_db()
    deadline = asyncio.get_event_loop().time() + 20  # 20-second long-poll

    while True:
        doc = await db.sessions.find_one({"_id": session_id})
        if not doc:
            raise HTTPException(404, "Session not found")

        messages = doc.get("messages", [])
        if since is not None:
            new_msgs = [m for m in messages if m.get("timestamp", 0) > since]
        else:
            new_msgs = messages

        if new_msgs:
            return {
                "messages":   new_msgs,
                "updated_at": doc.get("updated_at", datetime.utcnow()).timestamp()
                              if isinstance(doc.get("updated_at"), datetime)
                              else doc.get("updated_at"),
                "title": doc.get("title", "Collaborative Session"),
            }

        # Nothing yet — wait a bit before re-querying
        if asyncio.get_event_loop().time() >= deadline:
            return {"messages": [], "updated_at": since or 0, "title": doc.get("title", "")}

        await asyncio.sleep(1.5)


@collab_router.patch("/sessions/{session_id}/title")
async def update_session_title(session_id: str, req: PatchTitleRequest):
    """Rename a collaborative session."""
    db = get_db()
    result = await db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"title": req.title, "updated_at": datetime.utcnow()}}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@collab_router.post("/sessions/{session_id}/join")
async def join_session(session_id: str, req: Request):
    """Record a participant joining the session."""
    db = get_db()
    body = await req.json()
    name = body.get("name", "Anonymous")

    await db.sessions.update_one(
        {"_id": session_id},
        {
            "$addToSet": {"participants": name},
            "$set": {"updated_at": datetime.utcnow()},
        }
    )
    return {"ok": True}


@collab_router.delete("/sessions/{session_id}")
async def delete_collab_session(session_id: str):
    """Delete a collaborative session."""
    db = get_db()
    await db.sessions.delete_one({"_id": session_id})
    return {"ok": True}
