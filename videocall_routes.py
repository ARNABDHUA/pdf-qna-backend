"""
videocall_routes.py  –  WebRTC signaling + Web Push + Private Rooms
====================================================================
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
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

try:
    from pywebpush import webpush, WebPushException
    PUSH_AVAILABLE = True
except ImportError:
    PUSH_AVAILABLE = False
    print("Warning: pywebpush not installed. Run: pip install pywebpush")

VAPID_PRIVATE_KEY1 = os.environ.get("VAPID_PRIVATE_KEY1", "")
VAPID_PUBLIC_KEY1  = os.environ.get("VAPID_PUBLIC_KEY1",  "")
VAPID_EMAIL1       = os.environ.get("VAPID_EMAIL1", "mailto:admin@example.com")

# rooms[room_id] = {
#   "peers":    { peer_id: WebSocket },
#   "names":    { peer_id: display_name },
#   "password": str | None,
#   "host":     peer_id of creator
# }
rooms: dict[str, dict[str, Any]] = {}
push_subscriptions: dict[str, dict] = {}

videocall_router    = APIRouter(prefix="/videocall", tags=["VideoCall"])
videocall_ws_router = APIRouter(tags=["VideoCall-WS"])


# ── REST ──────────────────────────────────────────────────────────────────────

class CreateRoomRequest(BaseModel):
    password: Optional[str] = None
    host_id:  str = ""

@videocall_router.post("/rooms")
async def create_room(req: CreateRoomRequest):
    room_id = str(uuid.uuid4())[:8].upper()
    rooms[room_id] = {
        "peers":    {},
        "names":    {},
        "password": req.password.strip() if req.password and req.password.strip() else None,
        "host":     req.host_id,
    }
    return {"room_id": room_id, "private": rooms[room_id]["password"] is not None}


@videocall_router.get("/rooms/{room_id}")
async def get_room(room_id: str):
    room_id = room_id.upper()
    if room_id not in rooms:
        raise HTTPException(404, "Room not found")
    r = rooms[room_id]
    return {
        "room_id":    room_id,
        "peer_count": len(r["peers"]),
        "peers":      list(r["peers"].keys()),
        "names":      r["names"],
        "private":    r["password"] is not None,
    }


@videocall_router.post("/rooms/{room_id}/verify")
async def verify_room_password(room_id: str, body: dict):
    room_id = room_id.upper()
    if room_id not in rooms:
        raise HTTPException(404, "Room not found")
    room = rooms[room_id]
    if room["password"] is None:
        return {"ok": True}
    if body.get("password", "") == room["password"]:
        return {"ok": True}
    raise HTTPException(403, "Wrong password")


@videocall_router.get("/vapid-public-key")
async def get_VAPID_PUBLIC_KEY1():
    if not VAPID_PUBLIC_KEY1:
        raise HTTPException(503, "VAPID keys not configured.")
    return {"public_key": VAPID_PUBLIC_KEY1}


class PushSubscribeRequest(BaseModel):
    user_id:      str
    subscription: dict

@videocall_router.post("/push/subscribe")
async def subscribe_push(req: PushSubscribeRequest):
    push_subscriptions[req.user_id] = req.subscription
    return {"ok": True}


class PushSendRequest(BaseModel):
    user_id: str
    title:   str = "Meeting Notification"
    body:    str = "You have a new notification"
    url:     str = "/"

@videocall_router.post("/push/send")
async def send_push(req: PushSendRequest):
    if not PUSH_AVAILABLE:
        raise HTTPException(503, "pywebpush not installed")
    if not VAPID_PRIVATE_KEY1:
        raise HTTPException(503, "VAPID_PRIVATE_KEY1 not set")
    sub = push_subscriptions.get(req.user_id)
    if not sub:
        raise HTTPException(404, f"No push subscription for user '{req.user_id}'")
    try:
        webpush(
            subscription_info=sub,
            data=json.dumps({"title": req.title, "body": req.body, "url": req.url}),
            VAPID_PRIVATE_KEY1=VAPID_PRIVATE_KEY1,   # fixed: was VAPID_PRIVATE_KEY11
            vapid_claims={"sub": VAPID_EMAIL1},
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
    if not PUSH_AVAILABLE or not VAPID_PRIVATE_KEY1:
        raise HTTPException(503, "Push not configured")
    results = {}
    targets  = req.user_ids or list(push_subscriptions.keys())
    room_url = f"/meet?room={req.room_id}"
    for uid in targets:
        sub = push_subscriptions.get(uid)
        if not sub:
            results[uid] = "no_subscription"
            continue
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps({"title": req.title, "body": req.body, "url": room_url}),
                VAPID_PRIVATE_KEY1=VAPID_PRIVATE_KEY1,   # fixed: was VAPID_PRIVATE_KEY11
                vapid_claims={"sub": VAPID_EMAIL1},
            )
            results[uid] = "sent"
        except WebPushException as e:
            results[uid] = f"failed: {e}"
    return {"results": results}


# ── WebSocket signaling ───────────────────────────────────────────────────────

@videocall_ws_router.websocket("/ws/videocall/{room_id}/{peer_id}")
async def videocall_ws(
    websocket: WebSocket,
    room_id:   str,
    peer_id:   str,
    name:      str = Query(default=""),
    password:  str = Query(default=""),
):
    room_id = room_id.upper()
    await websocket.accept()

    if room_id not in rooms:
        rooms[room_id] = {"peers": {}, "names": {}, "password": None, "host": peer_id}

    room = rooms[room_id]

    if room["password"] and password != room["password"]:
        await websocket.send_json({"type": "error", "error": "wrong_password"})
        await websocket.close(code=4003)
        return

    existing_peers = [
        {"id": pid, "name": room["names"].get(pid, pid)}
        for pid in room["peers"]
    ]

    room["peers"][peer_id] = websocket
    room["names"][peer_id] = name or peer_id

    await websocket.send_json({"type": "peers", "peers": existing_peers})

    await _broadcast(room, peer_id, {
        "type": "peer_joined",
        "from": peer_id,
        "name": room["names"][peer_id],
    })

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type in ("offer", "answer", "ice"):
                to = msg.get("to")
                if to and to in room["peers"]:
                    await room["peers"][to].send_json({
                        "type":    msg_type,
                        "from":    peer_id,
                        "payload": msg.get("payload"),
                    })

            elif msg_type == "chat":
                await _broadcast(room, peer_id, {
                    "type":    "chat",
                    "from":    peer_id,
                    "name":    room["names"].get(peer_id, peer_id),
                    "payload": msg.get("payload"),
                })

            elif msg_type == "leave":
                break

    except WebSocketDisconnect:
        pass
    finally:
        room["peers"].pop(peer_id, None)
        room["names"].pop(peer_id, None)
        await _broadcast(room, peer_id, {"type": "peer_left", "from": peer_id})
        if not room["peers"]:
            rooms.pop(room_id, None)


async def _broadcast(room: dict, sender_id: str, message: dict):
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