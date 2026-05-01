"""
group_splits_routes.py — Group-based split expenses for ExpenseTracker
Features:
  - Create groups, invite via link, join via token
  - Add members by username, remove members (admin only)
  - Add group expenses (split equally by default, editable shares)
  - Month/year wise breakdown with category filter
  - You-owe / others-owe tracking
  - WebSocket for real-time updates
  - Web Push notifications (works outside browser/Chrome)

Add to main.py:
    from group_splits_routes import group_splits_router, group_ws_router
    app.include_router(group_splits_router)
    app.include_router(group_ws_router)

Requirements:
    pip install pywebpush

Env vars needed:
    VAPID_PUBLIC_KEY   = BEl62iU...   (base64url string from vapidkeys.com)
    VAPID_PRIVATE_KEY  = HkA9a3_...  (base64url string from vapidkeys.com)
    VAPID_MAILTO       = mailto:you@yourdomain.com
"""

import os
import hashlib
import hmac
import secrets
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

# ── Web Push (pywebpush) ───────────────────────────────────────────────────────
try:
    from pywebpush import webpush, WebPushException
    WEBPUSH_AVAILABLE = True
except ImportError:
    WEBPUSH_AVAILABLE = False
    print("Warning: pywebpush not installed. Run: pip install pywebpush")

MONGO_URL         = os.getenv("MONGODB_URL", "")
SECRET            = os.getenv("EXPENSE_SECRET", "expense_tracker_secret_2025")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_PUBLIC_KEY  = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_MAILTO      = os.getenv("VAPID_MAILTO", "mailto:admin@expenseai.com")

_client = None
_db     = None


def get_db():
    global _client, _db
    if not MONGO_AVAILABLE:
        raise HTTPException(500, "motor not installed. Run: pip install motor")
    if not MONGO_URL:
        raise HTTPException(500, "MONGODB_URL env var not set.")
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
        _db = _client["expense_tracker"]
    return _db


def hash_pw(pw: str) -> str:
    return hmac.new(SECRET.encode(), pw.encode(), hashlib.sha256).hexdigest()


def verify_pw(pw: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_pw(pw), hashed)


# ── WebSocket connection manager ───────────────────────────────────────────────
class GroupConnectionManager:
    def __init__(self):
        # group_id -> list of (username, websocket)
        self.active: Dict[str, List[tuple]] = {}

    async def connect(self, group_id: str, username: str, ws: WebSocket):
        await ws.accept()
        if group_id not in self.active:
            self.active[group_id] = []
        self.active[group_id].append((username, ws))

    def disconnect(self, group_id: str, ws: WebSocket):
        if group_id in self.active:
            self.active[group_id] = [
                (u, w) for u, w in self.active[group_id] if w != ws
            ]

    async def broadcast(
        self, group_id: str, message: dict, exclude_ws: WebSocket = None
    ):
        if group_id not in self.active:
            return
        dead = []
        for username, ws in self.active[group_id]:
            if ws == exclude_ws:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(group_id, ws)


manager = GroupConnectionManager()


# ── Web Push helper ────────────────────────────────────────────────────────────
async def send_push_to_members(
    db,
    group_id: str,
    exclude_username: str,
    payload: dict,
):
    """
    Send a Web Push notification to every group member except the sender.
    Silently skips if VAPID keys are not configured or pywebpush not installed.
    Cleans up expired (410) subscriptions automatically.
    """
    if not WEBPUSH_AVAILABLE or not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        return

    group = await db.groups.find_one({"group_id": group_id})
    if not group:
        return

    members = [m for m in group.get("members", []) if m != exclude_username]
    if not members:
        return

    subs_cursor = db.push_subscriptions.find({"username": {"$in": members}})
    subs = [s async for s in subs_cursor]

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    expired_ids = []

    for sub_doc in subs:
        try:
            webpush(
                subscription_info=sub_doc["subscription"],
                data=data,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_MAILTO},
            )
        except WebPushException as e:
            # 410 Gone = subscription no longer valid, remove it
            if e.response is not None and e.response.status_code == 410:
                expired_ids.append(sub_doc["_id"])
        except Exception as e:
            print(f"Push error for {sub_doc.get('username')}: {e}")

    if expired_ids:
        await db.push_subscriptions.delete_many({"_id": {"$in": expired_ids}})


# ── Pydantic Models ────────────────────────────────────────────────────────────
class AuthBase(BaseModel):
    username: str
    password: str


class CreateGroupRequest(AuthBase):
    group_name: str
    description: Optional[str] = ""
    currency: Optional[str] = "INR"


class JoinGroupRequest(AuthBase):
    invite_token: str


class AddMemberRequest(AuthBase):
    group_id: str
    member_username: str


class RemoveMemberRequest(AuthBase):
    group_id: str
    member_username: str


class SplitShare(BaseModel):
    username: str
    share: float
    paid: bool = False


class AddGroupExpenseRequest(AuthBase):
    group_id: str
    amount: float
    category: str
    description: str
    reason: Optional[str] = ""
    timestamp: Optional[int] = None
    splits: Optional[List[SplitShare]] = None   # None = equal split
    account_id: Optional[str] = None
    account_name: Optional[str] = None


class SettleRequest(AuthBase):
    group_id: str
    expense_id: str
    settled_for_username: str
    amount: float


class DeleteExpenseRequest(AuthBase):
    group_id: str
    expense_id: str


class PushSubscribeRequest(AuthBase):
    subscription: dict   # { endpoint, keys: { p256dh, auth } }


class PushUnsubscribeRequest(AuthBase):
    endpoint: str


# ── Helpers ────────────────────────────────────────────────────────────────────
def uid() -> str:
    return secrets.token_hex(12)


async def auth_user(db, username: str, password: str):
    uname = username.strip().lower()
    user  = await db.users.find_one({"username": uname})
    if not user:
        raise HTTPException(
            401, "User not found. Please save expenses to register first."
        )
    if not verify_pw(password, user["password_hash"]):
        raise HTTPException(401, "Incorrect password.")
    return user, uname


async def get_group_or_raise(db, group_id: str):
    group = await db.groups.find_one({"group_id": group_id})
    if not group:
        raise HTTPException(404, "Group not found.")
    return group


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def fmt_inr(n: float) -> str:
    return f"₹{n:,.2f}"


# ── Routers ────────────────────────────────────────────────────────────────────
group_splits_router = APIRouter(prefix="/groups", tags=["groups"])
group_ws_router     = APIRouter(prefix="/groups", tags=["groups-ws"])


# ── Push subscription endpoints ────────────────────────────────────────────────
@group_splits_router.post("/push-subscribe")
async def push_subscribe(req: PushSubscribeRequest):
    """Store a browser push subscription for this user."""
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    endpoint = req.subscription.get("endpoint", "")
    if not endpoint:
        raise HTTPException(400, "Invalid subscription: missing endpoint.")

    await db.push_subscriptions.update_one(
        {"username": uname, "endpoint": endpoint},
        {
            "$set": {
                "username":     uname,
                "subscription": req.subscription,
                "endpoint":     endpoint,
                "updated_at":   iso_now(),
            }
        },
        upsert=True,
    )
    return {"success": True, "message": "Push subscription saved."}


@group_splits_router.post("/push-unsubscribe")
async def push_unsubscribe(req: PushUnsubscribeRequest):
    """Remove a push subscription (called when user disables notifications)."""
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    await db.push_subscriptions.delete_many(
        {"username": uname, "endpoint": req.endpoint}
    )
    return {"success": True}


@group_splits_router.get("/vapid-public-key")
async def get_vapid_public_key():
    """Return the VAPID public key so the frontend can subscribe."""
    if not VAPID_PUBLIC_KEY:
        raise HTTPException(503, "Push notifications not configured on this server.")
    return {"vapid_public_key": VAPID_PUBLIC_KEY}


# ── Create Group ───────────────────────────────────────────────────────────────
@group_splits_router.post("/create")
async def create_group(req: CreateGroupRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    name = req.group_name.strip()
    if not name or len(name) < 2:
        raise HTTPException(400, "Group name must be at least 2 characters.")

    group_id     = uid()
    invite_token = secrets.token_urlsafe(16)

    group_doc = {
        "group_id":      group_id,
        "name":          name,
        "description":   req.description.strip(),
        "currency":      req.currency or "INR",
        "admin":         uname,
        "members":       [uname],
        "invite_token":  invite_token,
        "created_at":    iso_now(),
        "expense_count": 0,
    }
    await db.groups.insert_one(group_doc)

    return {
        "success":      True,
        "group_id":     group_id,
        "invite_token": invite_token,
        "invite_link":  f"join/{invite_token}",
        "message":      f"Group '{name}' created.",
    }


# ── Join via invite token ──────────────────────────────────────────────────────
@group_splits_router.post("/join")
async def join_group(req: JoinGroupRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await db.groups.find_one({"invite_token": req.invite_token.strip()})
    if not group:
        raise HTTPException(404, "Invalid or expired invite link.")

    if uname in group["members"]:
        return {
            "success":    True,
            "message":    "Already a member.",
            "group_id":   group["group_id"],
            "group_name": group["name"],
        }

    await db.groups.update_one(
        {"invite_token": req.invite_token.strip()},
        {"$addToSet": {"members": uname}},
    )

    await manager.broadcast(group["group_id"], {
        "type":     "member_joined",
        "username": uname,
        "group_id": group["group_id"],
    })

    # Push to existing members
    await send_push_to_members(db, group["group_id"], uname, {
        "title":   f"👋 {uname} joined {group['name']}",
        "body":    f"{uname} is now a member of the group.",
        "groupId": group["group_id"],
        "url":     "/expenses",
    })

    return {
        "success":    True,
        "message":    f"Joined group '{group['name']}'.",
        "group_id":   group["group_id"],
        "group_name": group["name"],
    }


# ── Add member by username ─────────────────────────────────────────────────────
@group_splits_router.post("/add-member")
async def add_member(req: AddMemberRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, req.group_id)
    if group["admin"] != uname:
        raise HTTPException(403, "Only the group admin can add members.")

    new_member = req.member_username.strip().lower()
    target = await db.users.find_one({"username": new_member})
    if not target:
        raise HTTPException(
            404, f"User '{new_member}' not found. They must have a saved account."
        )

    if new_member in group["members"]:
        return {"success": True, "message": f"{new_member} is already a member."}

    await db.groups.update_one(
        {"group_id": req.group_id},
        {"$addToSet": {"members": new_member}},
    )

    await manager.broadcast(req.group_id, {
        "type":     "member_added",
        "username": new_member,
        "by":       uname,
        "group_id": req.group_id,
    })

    await send_push_to_members(db, req.group_id, uname, {
        "title":   f"👋 {new_member} was added to {group['name']}",
        "body":    f"{uname} added {new_member} to the group.",
        "groupId": req.group_id,
        "url":     "/expenses",
    })

    return {"success": True, "message": f"Added {new_member} to the group."}


# ── Remove member (admin only) ─────────────────────────────────────────────────
@group_splits_router.post("/remove-member")
async def remove_member(req: RemoveMemberRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, req.group_id)
    if group["admin"] != uname:
        raise HTTPException(403, "Only the group admin can remove members.")

    target = req.member_username.strip().lower()
    if target == uname:
        raise HTTPException(400, "Admin cannot remove themselves.")
    if target not in group["members"]:
        return {"success": True, "message": f"{target} is not in this group."}

    await db.groups.update_one(
        {"group_id": req.group_id},
        {"$pull": {"members": target}},
    )

    await manager.broadcast(req.group_id, {
        "type":     "member_removed",
        "username": target,
        "by":       uname,
        "group_id": req.group_id,
    })

    return {"success": True, "message": f"Removed {target} from the group."}


# ── Get user's groups ──────────────────────────────────────────────────────────
@group_splits_router.post("/my-groups")
async def my_groups(req: AuthBase):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    cursor = db.groups.find({"members": uname}, {"_id": 0})
    groups = [g async for g in cursor]

    result = []
    for g in groups:
        gid = g["group_id"]
        exp_cursor = db.group_expenses.find({"group_id": gid}, {"_id": 0})
        expenses   = [e async for e in exp_cursor]

        you_owe_amount = 0
        others_owe_you = 0
        for e in expenses:
            paid_by = e.get("paid_by", "")
            for s in e.get("splits", []):
                if (
                    s["username"] == uname
                    and paid_by != uname
                    and not s.get("paid", False)
                ):
                    you_owe_amount += s["share"]
                if (
                    paid_by == uname
                    and s["username"] != uname
                    and not s.get("paid", False)
                ):
                    others_owe_you += s["share"]

        result.append({
            **{k: v for k, v in g.items() if k != "_id"},
            "expense_count": len(expenses),
            "you_owe":       round(you_owe_amount, 2),
            "others_owe":    round(others_owe_you, 2),
        })

    return {"success": True, "groups": result}


# ── Get single group detail ────────────────────────────────────────────────────
@group_splits_router.post("/{group_id}/detail")
async def group_detail(group_id: str, req: AuthBase):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, group_id)
    if uname not in group["members"]:
        raise HTTPException(403, "You are not a member of this group.")

    exp_cursor = db.group_expenses.find({"group_id": group_id}, {"_id": 0})
    expenses   = sorted(
        [e async for e in exp_cursor],
        key=lambda x: x["timestamp"],
        reverse=True,
    )

    return {
        "success":      True,
        "group":        {k: v for k, v in group.items() if k != "_id"},
        "expenses":     expenses,
        "current_user": uname,
    }


# ── Add group expense ──────────────────────────────────────────────────────────
@group_splits_router.post("/add-expense")
async def add_group_expense(req: AddGroupExpenseRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, req.group_id)
    if uname not in group["members"]:
        raise HTTPException(403, "You are not a member of this group.")

    if req.amount <= 0:
        raise HTTPException(400, "Amount must be positive.")

    members = group["members"]
    n       = len(members)

    # Build splits
    if req.splits:
        total_shares = sum(s.share for s in req.splits)
        if abs(total_shares - req.amount) > 0.01:
            raise HTTPException(
                400,
                f"Split shares ({total_shares}) must sum to amount ({req.amount}).",
            )
        splits = [
            {"username": s.username, "share": round(s.share, 2), "paid": s.paid}
            for s in req.splits
        ]
    else:
        # Equal split — payer's share is auto-settled
        per_person = round(req.amount / n, 2)
        splits     = []
        running    = 0
        for i, member in enumerate(members):
            share = per_person
            if i == n - 1:
                share = round(req.amount - running, 2)
            running += share
            splits.append({
                "username": member,
                "share":    share,
                "paid":     member == uname,
            })

    expense_id = uid()
    ts = (
        req.timestamp
        if req.timestamp
        else int(datetime.now(timezone.utc).timestamp() * 1000)
    )

    expense_doc = {
        "expense_id":   expense_id,
        "group_id":     req.group_id,
        "paid_by":      uname,
        "amount":       req.amount,
        "category":     req.category,
        "description":  req.description[:80],
        "reason":       (req.reason or "")[:100],
        "timestamp":    ts,
        "splits":       splits,
        "account_id":   req.account_id,
        "account_name": req.account_name,
        "created_at":   iso_now(),
    }

    await db.group_expenses.insert_one(expense_doc)
    await db.groups.update_one(
        {"group_id": req.group_id}, {"$inc": {"expense_count": 1}}
    )

    broadcast_doc = {k: v for k, v in expense_doc.items() if k != "_id"}

    # WebSocket broadcast (real-time for open tabs)
    await manager.broadcast(req.group_id, {
        "type":       "new_expense",
        "expense":    broadcast_doc,
        "group_id":   req.group_id,
        "group_name": group["name"],
    })

    # Web Push (for closed tabs / background)
    per_person_str = fmt_inr(req.amount / n) if n > 0 else fmt_inr(req.amount)
    await send_push_to_members(db, req.group_id, uname, {
        "title":   f"💸 New expense in {group['name']}",
        "body":    f"{uname} added {req.description} — {fmt_inr(req.amount)} (your share: {per_person_str})",
        "groupId": req.group_id,
        "url":     "/expenses",
    })

    return {
        "success":    True,
        "expense_id": expense_id,
        "expense":    broadcast_doc,
        "message":    f"Expense added. Split among {n} members.",
    }


# ── Settle a share ─────────────────────────────────────────────────────────────
@group_splits_router.post("/settle")
async def settle_share(req: SettleRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, req.group_id)
    if uname not in group["members"]:
        raise HTTPException(403, "Not a group member.")

    expense = await db.group_expenses.find_one({
        "expense_id": req.expense_id,
        "group_id":   req.group_id,
    })
    if not expense:
        raise HTTPException(404, "Expense not found.")

    new_splits = []
    found      = False
    for s in expense.get("splits", []):
        if s["username"] == req.settled_for_username and not s.get("paid", False):
            new_splits.append({**s, "paid": True})
            found = True
        else:
            new_splits.append(s)

    if not found:
        raise HTTPException(400, "Share not found or already settled.")

    await db.group_expenses.update_one(
        {"expense_id": req.expense_id},
        {"$set": {"splits": new_splits}},
    )

    await manager.broadcast(req.group_id, {
        "type":        "share_settled",
        "expense_id":  req.expense_id,
        "settled_for": req.settled_for_username,
        "settled_by":  uname,
        "amount":      req.amount,
        "group_id":    req.group_id,
    })

    # Push to the person whose share was settled
    await send_push_to_members(db, req.group_id, uname, {
        "title":   f"✅ Share settled in {group['name']}",
        "body":    f"{uname} marked {fmt_inr(req.amount)} as paid for {req.settled_for_username}",
        "groupId": req.group_id,
        "url":     "/expenses",
    })

    return {
        "success": True,
        "message": f"Settled {fmt_inr(req.amount)} for {req.settled_for_username}.",
    }


# ── Delete group expense (admin or payer) ──────────────────────────────────────
@group_splits_router.post("/delete-expense")
async def delete_expense(req: DeleteExpenseRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group   = await get_group_or_raise(db, req.group_id)
    expense = await db.group_expenses.find_one({
        "expense_id": req.expense_id,
        "group_id":   req.group_id,
    })
    if not expense:
        raise HTTPException(404, "Expense not found.")

    if expense["paid_by"] != uname and group["admin"] != uname:
        raise HTTPException(403, "Only the payer or admin can delete this expense.")

    await db.group_expenses.delete_one({"expense_id": req.expense_id})
    await db.groups.update_one(
        {"group_id": req.group_id}, {"$inc": {"expense_count": -1}}
    )

    await manager.broadcast(req.group_id, {
        "type":       "expense_deleted",
        "expense_id": req.expense_id,
        "group_id":   req.group_id,
        "by":         uname,
    })

    await send_push_to_members(db, req.group_id, uname, {
        "title":   f"🗑 Expense deleted in {group['name']}",
        "body":    f"{uname} deleted: {expense.get('description', 'an expense')}",
        "groupId": req.group_id,
        "url":     "/expenses",
    })

    return {"success": True, "message": "Expense deleted."}


# ── Get group invite link ──────────────────────────────────────────────────────
@group_splits_router.post("/{group_id}/invite")
async def get_invite(group_id: str, req: AuthBase):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, group_id)
    if uname not in group["members"]:
        raise HTTPException(403, "Not a member.")

    return {
        "success":      True,
        "invite_token": group["invite_token"],
        "invite_link":  f"join/{group['invite_token']}",
    }


# ── Regenerate invite token (admin) ───────────────────────────────────────────
@group_splits_router.post("/{group_id}/regen-invite")
async def regen_invite(group_id: str, req: AuthBase):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, group_id)
    if group["admin"] != uname:
        raise HTTPException(403, "Only admin can regenerate invite link.")

    new_token = secrets.token_urlsafe(16)
    await db.groups.update_one(
        {"group_id": group_id}, {"$set": {"invite_token": new_token}}
    )

    return {
        "success":      True,
        "invite_token": new_token,
        "invite_link":  f"join/{new_token}",
    }


# ── Leave group ────────────────────────────────────────────────────────────────
@group_splits_router.post("/{group_id}/leave")
async def leave_group(group_id: str, req: AuthBase):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, group_id)
    if uname not in group["members"]:
        raise HTTPException(400, "You are not a member of this group.")
    if group["admin"] == uname:
        raise HTTPException(
            400, "Admin cannot leave. Transfer admin or delete the group."
        )

    await db.groups.update_one(
        {"group_id": group_id}, {"$pull": {"members": uname}}
    )
    await manager.broadcast(group_id, {
        "type":     "member_left",
        "username": uname,
        "group_id": group_id,
    })

    return {"success": True, "message": f"Left group '{group['name']}'."}


# ── Delete group (admin only) ──────────────────────────────────────────────────
@group_splits_router.post("/{group_id}/delete")
async def delete_group(group_id: str, req: AuthBase):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, group_id)
    if group["admin"] != uname:
        raise HTTPException(403, "Only admin can delete the group.")

    await db.groups.delete_one({"group_id": group_id})
    await db.group_expenses.delete_many({"group_id": group_id})

    await manager.broadcast(group_id, {
        "type":     "group_deleted",
        "group_id": group_id,
    })

    return {"success": True, "message": "Group deleted."}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@group_ws_router.websocket("/ws/{group_id}/{username}")
async def group_websocket(ws: WebSocket, group_id: str, username: str):
    uname = username.strip().lower()
    try:
        await manager.connect(group_id, uname, ws)
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(group_id, ws)
    except Exception:
        try:
            manager.disconnect(group_id, ws)
        except Exception:
            pass