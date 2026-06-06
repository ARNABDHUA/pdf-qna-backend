"""
videocall_routes.py  –  WebRTC signaling + Web Push notifications
=================================================================
Add to main.py:
    from videocall_routes import videocall_router, videocall_ws_router
    app.include_router(videocall_router)
    app.include_router(videocall_ws_router)

Install deps:
    pip install pywebpush cryptography websockets
"""

import asyncio
import json
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Optional: pywebpush for Web Push ─────────────────────────────────────────
try:
    from pywebpush import webpush, WebPushException
    PUSH_AVAILABLE = True
except ImportError:
    PUSH_AVAILABLE = False
    print("Warning: pywebpush not installed. Run: pip install pywebpush")

# ─────────────────────────────────────────────────────────────────────────────
# VAPID keys – generate once and store in env vars:
#   python -c "from py_vapid import Vapid; v=Vapid(); v.generate_keys(); print(v.private_pem().decode()); print(v.public_key.public_bytes_raw().hex())"
# Or use: npx web-push generate-vapid-keys
# ─────────────────────────────────────────────────────────────────────────────
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
VAPID_PUBLIC_KEY  = os.environ.get("VAPID_PUBLIC_KEY",  "")
VAPID_EMAILTO       = os.environ.get("VAPID_EMAILTO", "mailto:admin@example.com")

# ── In-memory state (swap for Redis in production) ───────────────────────────
# rooms[room_id] = { "peers": {peer_id: WebSocket} }
rooms: dict[str, dict[str, Any]] = {}

# push_subscriptions[user_id] = subscription_info dict
push_subscriptions: dict[str, dict] = {}

# ── Routers ───────────────────────────────────────────────────────────────────
videocall_router    = APIRouter(prefix="/videocall", tags=["VideoCall"])
videocall_ws_router = APIRouter(tags=["VideoCall-WS"])


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@videocall_router.post("/rooms")
async def create_room():
    """Create a new meeting room and return its ID."""
    room_id = str(uuid.uuid4())[:8].upper()
    rooms[room_id] = {"peers": {}}
    return {"room_id": room_id}


@videocall_router.get("/rooms/{room_id}")
async def get_room(room_id: str):
    room_id = room_id.upper()
    if room_id not in rooms:
        raise HTTPException(404, "Room not found")
    peer_ids = list(rooms[room_id]["peers"].keys())
    return {"room_id": room_id, "peer_count": len(peer_ids), "peers": peer_ids}


@videocall_router.get("/vapid-public-key")
async def get_vapid_public_key():
    """Return the VAPID public key so the frontend can subscribe."""
    if not VAPID_PUBLIC_KEY:
        raise HTTPException(503, "VAPID keys not configured. Set VAPID_PUBLIC_KEY env var.")
    return {"public_key": VAPID_PUBLIC_KEY}


class PushSubscribeRequest(BaseModel):
    user_id:      str
    subscription: dict   # {endpoint, keys: {p256dh, auth}}


@videocall_router.post("/push/subscribe")
async def subscribe_push(req: PushSubscribeRequest):
    push_subscriptions[req.user_id] = req.subscription
    return {"ok": True, "message": f"Subscribed {req.user_id} to push notifications"}


class PushSendRequest(BaseModel):
    user_id: str
    title:   str = "Meeting Notification"
    body:    str = "You have a new notification"
    url:     str = "/"


@videocall_router.post("/push/send")
async def send_push(req: PushSendRequest):
    if not PUSH_AVAILABLE:
        raise HTTPException(503, "pywebpush not installed")
    if not VAPID_PRIVATE_KEY:
        raise HTTPException(503, "VAPID_PRIVATE_KEY not set")

    sub = push_subscriptions.get(req.user_id)
    if not sub:
        raise HTTPException(404, f"No push subscription for user '{req.user_id}'")

    payload = json.dumps({
        "title": req.title,
        "body":  req.body,
        "url":   req.url,
    })

    try:
        webpush(
            subscription_info=sub,
            data=payload,
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={"sub": VAPID_EMAILTO},
        )
        return {"ok": True}
    except WebPushException as e:
        raise HTTPException(500, f"Push failed: {e}")


class NotifyRoomRequest(BaseModel):
    room_id:  str
    title:    str = "Meeting Invite"
    body:     str = "You've been invited to a meeting"
    user_ids: list[str] = []


@videocall_router.post("/push/notify-room")
async def notify_room(req: NotifyRoomRequest):
    """Send push notifications to a list of users about a meeting."""
    if not PUSH_AVAILABLE or not VAPID_PRIVATE_KEY:
        raise HTTPException(503, "Push not configured")

    results = {}
    targets = req.user_ids or list(push_subscriptions.keys())
    room_url = f"/meet/{req.room_id}"

    for uid in targets:
        sub = push_subscriptions.get(uid)
        if not sub:
            results[uid] = "no_subscription"
            continue
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps({"title": req.title, "body": req.body, "url": room_url}),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_EMAILTO},
            )
            results[uid] = "sent"
        except WebPushException as e:
            results[uid] = f"failed: {e}"

    return {"results": results}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket signaling endpoint
# ─────────────────────────────────────────────────────────────────────────────
# Message protocol (JSON):
#   Client → Server: { type, room_id, peer_id, [to], [payload] }
#   Server → Client: { type, from, [payload] }
#
# Types:
#   join         – join a room
#   offer        – WebRTC offer  (peer-to-peer, relayed)
#   answer       – WebRTC answer (peer-to-peer, relayed)
#   ice          – ICE candidate (peer-to-peer, relayed)
#   leave        – peer left
#   peers        – server → joiner: list of existing peers in room
#   error        – server error message

@videocall_ws_router.websocket("/ws/videocall/{room_id}/{peer_id}")
async def videocall_ws(websocket: WebSocket, room_id: str, peer_id: str):
    room_id = room_id.upper()
    await websocket.accept()

    # Auto-create room if it doesn't exist
    if room_id not in rooms:
        rooms[room_id] = {"peers": {}}

    room = rooms[room_id]
    existing_peers = list(room["peers"].keys())

    # Register this peer
    room["peers"][peer_id] = websocket

    # Tell the new joiner about existing peers
    await websocket.send_json({
        "type":  "peers",
        "peers": existing_peers,
    })

    # Tell everyone else that a new peer joined
    await _broadcast(room, peer_id, {"type": "peer_joined", "from": peer_id})

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type in ("offer", "answer", "ice"):
                # Relay directly to the target peer
                to = msg.get("to")
                if to and to in room["peers"]:
                    await room["peers"][to].send_json({
                        "type":    msg_type,
                        "from":    peer_id,
                        "payload": msg.get("payload"),
                    })
                else:
                    await websocket.send_json({
                        "type":  "error",
                        "error": f"Peer '{to}' not found in room",
                    })

            elif msg_type == "leave":
                break

            elif msg_type == "chat":
                # Broadcast chat message to everyone in room
                await _broadcast(room, peer_id, {
                    "type":    "chat",
                    "from":    peer_id,
                    "payload": msg.get("payload"),
                })

    except WebSocketDisconnect:
        pass
    finally:
        room["peers"].pop(peer_id, None)
        await _broadcast(room, peer_id, {"type": "peer_left", "from": peer_id})
        # Clean up empty rooms
        if not room["peers"]:
            rooms.pop(room_id, None)


async def _broadcast(room: dict, sender_id: str, message: dict):
    """Send a message to all peers in a room except the sender."""
    dead = []
    for pid, ws in room["peers"].items():
        if pid == sender_id:
            continue
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(pid)
    for pid in dead:
        room["peers"].pop(pid, None)
