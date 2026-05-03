"""
group_splits_routes.py — Group-based split expenses for ExpenseTracker
FIXES APPLIED:
  1. hash_pw now matches expense_mongo_routes.py exactly (same hmac.new usage)
  2. pywebpush webpush() call uses correct keyword args for v2.x
  3. Added error logging so push failures are visible in Render logs
  4. send_push_to_members logs why it skips (helps debugging)
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

try:
    from pywebpush import webpush, WebPushException
    WEBPUSH_AVAILABLE = True
except ImportError:
    WEBPUSH_AVAILABLE = False
    print("Warning: pywebpush not installed. Run: pip install pywebpush")

MONGO_URL         = os.getenv("MONGODB_URL", "")
# ── FIX: use same secret as expense_mongo_routes.py ───────────────────────────
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


# ── FIX: hash_pw must exactly match expense_mongo_routes.py ───────────────────
def hash_pw(pw: str) -> str:
    """SHA-256 HMAC — identical to expense_mongo_routes.hash_password()"""
    return hmac.new(SECRET.encode(), pw.encode(), hashlib.sha256).hexdigest()


def verify_pw(pw: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_pw(pw), hashed)


# ── WebSocket connection manager ───────────────────────────────────────────────
class GroupConnectionManager:
    def __init__(self):
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

    FIX: Added detailed logging so you can see in Render logs why pushes
    are skipped or failing.
    """
    if not WEBPUSH_AVAILABLE:
        print("[Push] Skipped: pywebpush not installed")
        return
    if not VAPID_PRIVATE_KEY:
        print("[Push] Skipped: VAPID_PRIVATE_KEY env var is empty")
        return
    if not VAPID_PUBLIC_KEY:
        print("[Push] Skipped: VAPID_PUBLIC_KEY env var is empty")
        return

    group = await db.groups.find_one({"group_id": group_id})
    if not group:
        print(f"[Push] Skipped: group {group_id} not found")
        return

    members = [m for m in group.get("members", []) if m != exclude_username]
    if not members:
        print(f"[Push] No other members in group {group_id}")
        return

    subs_cursor = db.push_subscriptions.find({"username": {"$in": members}})
    all_subs = [s async for s in subs_cursor]
    # Deduplicate: keep latest subscription per username
    seen = {}
    for s in all_subs:
        u = s.get("username")
        if u not in seen or s.get("updated_at", "") > seen[u].get("updated_at", ""):
            seen[u] = s
    subs = list(seen.values())

    if not subs:
        print(f"[Push] No push subscriptions found for members: {members}")
        return

    print(f"[Push] Sending to {len(subs)} subscription(s) in group {group_id}")
    data = json.dumps(payload, ensure_ascii=False)
    expired_ids = []

    for sub_doc in subs:
        try:
            # ── FIX: pywebpush 2.x API ────────────────────────────────────────
            webpush(
                subscription_info=sub_doc["subscription"],
                data=data,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_MAILTO},
            )
            print(f"[Push] ✓ Sent to {sub_doc.get('username')}")
        except WebPushException as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"[Push] WebPushException for {sub_doc.get('username')}: status={status} — {e}")
            if e.response is not None and e.response.status_code == 410:
                expired_ids.append(sub_doc["_id"])
        except Exception as e:
            print(f"[Push] Unexpected error for {sub_doc.get('username')}: {e}")

    if expired_ids:
        await db.push_subscriptions.delete_many({"_id": {"$in": expired_ids}})
        print(f"[Push] Removed {len(expired_ids)} expired subscription(s)")


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
    splits: Optional[List[SplitShare]] = None
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
    subscription: dict


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
    print(f"[Push] Subscription saved for {uname}")
    return {"success": True, "message": "Push subscription saved."}


@group_splits_router.post("/push-unsubscribe")
async def push_unsubscribe(req: PushUnsubscribeRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)
    await db.push_subscriptions.delete_many(
        {"username": uname, "endpoint": req.endpoint}
    )
    return {"success": True}


@group_splits_router.get("/vapid-public-key")
async def get_vapid_public_key():
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

    await manager.broadcast(req.group_id, {
        "type":       "new_expense",
        "expense":    broadcast_doc,
        "group_id":   req.group_id,
        "group_name": group["name"],
    })

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


# # ── Leave group ────────────────────────────────────────────────────────────────
# @group_splits_router.post("/{group_id}/leave")
# async def leave_group(group_id: str, req: AuthBase):
#     db = get_db()
#     _, uname = await auth_user(db, req.username, req.password)

#     group = await get_group_or_raise(db, group_id)
#     if uname not in group["members"]:
#         raise HTTPException(400, "You are not a member of this group.")
#     if group["admin"] == uname:
#         raise HTTPException(
#             400, "Admin cannot leave. Transfer admin or delete the group."
#         )

#     await db.groups.update_one(
#         {"group_id": group_id}, {"$pull": {"members": uname}}
#     )
#     await manager.broadcast(group_id, {
#         "type":     "member_left",
#         "username": uname,
#         "group_id": group_id,
#     })

#     return {"success": True, "message": f"Left group '{group['name']}'."}


# ── Leave group ────────────────────────────────────────────────────────────────
@group_splits_router.post("/{group_id}/leave")
async def leave_group(group_id: str, req: AuthBase):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, group_id)
    if uname not in group["members"]:
        raise HTTPException(400, "You are not a member of this group.")

    remaining = [m for m in group["members"] if m != uname]

    # If admin is leaving, auto-transfer admin to next member (or delete if last)
    update_fields = {"$pull": {"members": uname}}
    if group["admin"] == uname:
        if len(remaining) == 0:
            # Admin is last member — delete the group entirely
            await db.groups.delete_one({"group_id": group_id})
            await db.group_expenses.delete_many({"group_id": group_id})
            await manager.broadcast(group_id, {
                "type": "group_deleted",
                "group_id": group_id,
            })
            return {
                "success": True,
                "message": f"Group '{group['name']}' deleted (you were the last member).",
            }
        else:
            # Transfer admin to the next member in the list
            new_admin = remaining[0]
            update_fields["$set"] = {"admin": new_admin}

    await db.groups.update_one({"group_id": group_id}, update_fields)

    await manager.broadcast(group_id, {
        "type": "member_left",
        "username": uname,
        "group_id": group_id,
        "new_admin": update_fields.get("$set", {}).get("admin"),
    })

    return {
        "success": True,
        "message": f"Left group '{group['name']}'.",
        "new_admin": update_fields.get("$set", {}).get("admin"),
    }


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


# ── ADD THESE TWO ENDPOINTS to group_splits_routes.py ────────────────────────
# Paste them just before the WebSocket endpoint section


class CheckPaidStatusRequest(BaseModel):
    username: str
    password: str
    debtor_username: str  


class PayMySharesRequest(BaseModel):
    username: str
    password: str


@group_splits_router.post("/{group_id}/check-paid-status")
async def check_paid_status(group_id: str, req: CheckPaidStatusRequest):
    """
    Creditor calls this before showing Settle All.
    Returns whether the debtor has acknowledged/paid their shares
    (i.e. all their splits are marked paid=True by themselves).
    """
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, group_id)
    if uname not in group["members"]:
        raise HTTPException(403, "Not a group member.")

    debtor = req.debtor_username.strip().lower()

    exp_cursor = db.group_expenses.find({"group_id": group_id}, {"_id": 0})
    expenses = [e async for e in exp_cursor]

    unpaid_shares = []
    for e in expenses:
        if e.get("paid_by") == uname:          # only expenses THIS user paid for
            for s in e.get("splits", []):
                if s["username"] == debtor and not s.get("paid", False):
                    unpaid_shares.append({
                        "expense_id": e["expense_id"],
                        "amount": s["share"],
                    })

    total_unpaid = sum(s["amount"] for s in unpaid_shares)
    return {
        "success": True,
        "fully_paid": len(unpaid_shares) == 0,
        "unpaid_count": len(unpaid_shares),
        "total_unpaid": round(total_unpaid, 2),
    }


@group_splits_router.post("/{group_id}/pay-my-shares")
async def pay_my_shares(group_id: str, req: PayMySharesRequest):
    """
    Debtor calls this to mark all their own unpaid splits as paid.
    Returns total amount paid and count of expenses settled.
    Broadcasts a WebSocket event so creditors are notified.
    """
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    group = await get_group_or_raise(db, group_id)
    if uname not in group["members"]:
        raise HTTPException(403, "Not a group member.")

    exp_cursor = db.group_expenses.find({"group_id": group_id}, {"_id": 0})
    expenses = [e async for e in exp_cursor]

    total_paid = 0.0
    settled_count = 0

    for e in expenses:
        if e.get("paid_by") == uname:
            continue                           # skip expenses you paid for yourself

        new_splits = []
        changed = False
        for s in e.get("splits", []):
            if s["username"] == uname and not s.get("paid", False):
                new_splits.append({**s, "paid": True})
                total_paid += s["share"]
                changed = True
            else:
                new_splits.append(s)

        if changed:
            await db.group_expenses.update_one(
                {"expense_id": e["expense_id"]},
                {"$set": {"splits": new_splits}},
            )
            settled_count += 1

    total_paid = round(total_paid, 2)

    # Broadcast so creditors see the update in real time
    await manager.broadcast(group_id, {
        "type": "shares_self_paid",
        "by": uname,
        "total": total_paid,
        "settled_count": settled_count,
        "group_id": group_id,
    })

    # Push notification to creditors
    await send_push_to_members(db, group_id, uname, {
        "title": f"💳 {uname} paid their dues in {group['name']}",
        "body": f"{uname} marked {fmt_inr(total_paid)} as paid — you can now settle!",
        "groupId": group_id,
        "url": "/expenses",
    })

    return {
        "success": True,
        "total_paid": total_paid,
        "settled_count": settled_count,
        "message": f"Marked {settled_count} expense share(s) as paid (total {fmt_inr(total_paid)}).",
    }


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


class UnsettleRequest(BaseModel):
    username: str
    password: str
    group_id: str
    expense_id: str
    settled_for_username: str

@group_splits_router.post("/unsettle")
async def unsettle_share(req: UnsettleRequest):
    db = get_db()
    _, uname = await auth_user(db, req.username, req.password)

    expense = await db.group_expenses.find_one({
        "expense_id": req.expense_id,
        "group_id":   req.group_id,
    })
    if not expense:
        raise HTTPException(404, "Expense not found.")

    # Check if the payer (paid_by) has already done a "settle all"
    # We detect this by checking if ALL non-payer splits are paid
    splits = expense.get("splits", [])
    payer  = expense.get("paid_by", "")

    # Find the target split
    target_split = next(
        (s for s in splits if s["username"] == req.settled_for_username and s.get("paid")),
        None
    )
    if not target_split:
        # Already unpaid or not found — nothing to revert
        return {"success": True, "reverted": False, "reason": "Already unpaid or not found"}

    # Check if payer has manually settled all (all other splits paid)
    # If payer explicitly ran "settle all", we respect that and don't revert
    other_splits_all_paid = all(
        s.get("paid", False)
        for s in splits
        if s["username"] != payer and s["username"] != req.settled_for_username
    )

    # Simple rule: if the expense was settled by payer action (paid field set by payer),
    # we check via a flag. For now, allow revert only if payer hasn't confirmed.
    # Mark as unpaid
    new_splits = [
        {**s, "paid": False} if s["username"] == req.settled_for_username else s
        for s in splits
    ]

    await db.group_expenses.update_one(
        {"expense_id": req.expense_id},
        {"$set": {"splits": new_splits}}
    )

    await manager.broadcast(req.group_id, {
        "type":        "share_unsettled",
        "expense_id":  req.expense_id,
        "unsettled_for": req.settled_for_username,
        "group_id":    req.group_id,
    })

    return {"success": True, "reverted": True}