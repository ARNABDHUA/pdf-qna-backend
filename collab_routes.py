"""
collab_routes.py — Collaborative Session API
"""

import os
import uuid
import asyncio
import certifi
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

# ── MongoDB connection ─────────────────────────────────────────────────────────
MONGO_URL = os.getenv("MONGODB_URL", "")

_mongo_client: Optional[AsyncIOMotorClient] = None


def get_db():
    global _mongo_client
    if _mongo_client is None:
        if not MONGO_URL:
            raise RuntimeError("MONGODB_URL environment variable is not set")

        # Build connection URL with TLS params if not already present
        url = MONGO_URL
        if "tlsCAFile" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}tlsCAFile={certifi.where()}&tls=true&tlsAllowInvalidCertificates=true"

        _mongo_client = AsyncIOMotorClient(
            url,
            serverSelectionTimeoutMS=10000,
            socketTimeoutMS=20000,
            connectTimeoutMS=20000,
        )
    return _mongo_client["qna_ai_collab"]


# ── Pydantic models ────────────────────────────────────────────────────────────

class CollabMessage(BaseModel):
    role: str
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


# ── Helper ────────────────────────────────────────────────────────────────────
def _sanitise(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id", ""))
    for key in ("created_at", "updated_at", "expires_at"):
        if key in doc and isinstance(doc[key], datetime):
            doc[key] = doc[key].isoformat()
    for msg in doc.get("messages", []):
        for k, v in msg.items():
            if isinstance(v, datetime):
                msg[k] = v.isoformat()
    return doc


# ── Routes ────────────────────────────────────────────────────────────────────

@collab_router.post("/sessions")
async def create_collab_session(req: CreateSessionRequest):
    db = get_db()
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()

    doc = {
        "_id":          session_id,
        "title":        req.title or "Collaborative Session",
        "owner":        req.owner or "Anonymous",
        "messages":     req.messages or [],
        "created_at":   now,
        "updated_at":   now,
        "expires_at":   now + timedelta(days=7),
        "participants": [],
        "active_users": 0,
    }

    try:
        await db.sessions.insert_one(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")

    return {
        "session_id": session_id,
        "share_url":  f"/collab/{session_id}",
        "expires_at": doc["expires_at"].isoformat(),
    }


@collab_router.get("/sessions/{session_id}")
async def get_collab_session(session_id: str):
    db = get_db()

    try:
        doc = await db.sessions.find_one({"_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if not doc:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if doc.get("expires_at") and datetime.utcnow() > doc["expires_at"]:
        raise HTTPException(status_code=410, detail="Session has expired")

    return _sanitise(doc)


@collab_router.post("/sessions/{session_id}/messages")
async def add_message_to_session(session_id: str, req: AddMessageRequest):
    db = get_db()

    try:
        doc = await db.sessions.find_one({"_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    msg = req.message.dict()
    msg["timestamp"] = msg.get("timestamp") or datetime.utcnow().timestamp()
    msg["msg_id"]    = msg.get("msg_id") or str(uuid.uuid4())

    try:
        result = await db.sessions.update_one(
            {"_id": session_id},
            {
                "$push": {"messages": msg},
                "$set":  {"updated_at": datetime.utcnow()},
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {e}")

    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to add message — no document modified")

    return {"ok": True, "msg_id": msg["msg_id"]}


@collab_router.patch("/sessions/{session_id}/messages/{msg_id}")
async def update_message_in_session(session_id: str, msg_id: str, req: Request):
    db = get_db()

    try:
        body    = await req.json()
        content = body.get("content", "")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        result = await db.sessions.update_one(
            {"_id": session_id, "messages.msg_id": msg_id},
            {
                "$set": {
                    "messages.$.content": content,
                    "updated_at":         datetime.utcnow(),
                }
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")

    return {"ok": True}


@collab_router.get("/sessions/{session_id}/poll")
async def poll_session(session_id: str, since: Optional[float] = None):
    db = get_db()
    loop     = asyncio.get_event_loop()
    deadline = loop.time() + 20

    while True:
        try:
            doc = await db.sessions.find_one({"_id": session_id})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = doc.get("messages", [])
        new_msgs = (
            [m for m in messages if m.get("timestamp", 0) > since]
            if since is not None
            else messages
        )

        for msg in new_msgs:
            for k, v in msg.items():
                if isinstance(v, datetime):
                    msg[k] = v.isoformat()

        updated_at = doc.get("updated_at")
        updated_ts = (
            updated_at.timestamp() if isinstance(updated_at, datetime)
            else (updated_at or 0)
        )

        if new_msgs:
            return {
                "messages":   new_msgs,
                "updated_at": updated_ts,
                "title":      doc.get("title", "Collaborative Session"),
            }

        if loop.time() >= deadline:
            return {
                "messages":   [],
                "updated_at": since or 0,
                "title":      doc.get("title", ""),
            }

        await asyncio.sleep(1.5)


@collab_router.patch("/sessions/{session_id}/title")
async def update_session_title(session_id: str, req: PatchTitleRequest):
    db = get_db()

    try:
        result = await db.sessions.update_one(
            {"_id": session_id},
            {"$set": {"title": req.title, "updated_at": datetime.utcnow()}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"ok": True}


@collab_router.post("/sessions/{session_id}/join")
async def join_session(session_id: str, req: Request):
    db = get_db()

    try:
        body = await req.json()
        name = body.get("name", "Anonymous")
    except Exception:
        name = "Anonymous"

    try:
        await db.sessions.update_one(
            {"_id": session_id},
            {
                "$addToSet": {"participants": name},
                "$set":      {"updated_at": datetime.utcnow()},
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return {"ok": True}


@collab_router.delete("/sessions/{session_id}")
async def delete_collab_session(session_id: str):
    db = get_db()

    try:
        await db.sessions.delete_one({"_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return {"ok": True}