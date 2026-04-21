"""
expense_mongo_routes.py  —  MongoDB save/sync for ExpenseTracker
Add to main.py:
    from expense_mongo_routes import expense_mongo_router
    app.include_router(expense_mongo_router)
"""

import os
import hashlib
import hmac
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ── MongoDB (Motor async driver) ──────────────────────────────────────────────
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("Warning: motor not installed. Run: pip install motor")

MONGO_URL = os.getenv("MONGODB_URL", "")
_client = None
_db = None


def get_db():
    global _client, _db
    if not MONGO_AVAILABLE:
        raise HTTPException(500, "motor driver not installed. Run: pip install motor")
    if not MONGO_URL:
        raise HTTPException(500, "MONGODB_URL env var is not set.")
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
        _db = _client["expense_tracker"]
    return _db


# ── Password hashing (no bcrypt dependency — plain SHA-256 HMAC) ──────────────
SECRET = os.getenv("EXPENSE_SECRET", "expense_tracker_secret_2025")


def hash_password(password: str) -> str:
    return hmac.new(SECRET.encode(), password.encode(), hashlib.sha256).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_password(password), hashed)


# ── Pydantic models ───────────────────────────────────────────────────────────
class ExpenseItem(BaseModel):
    id: str
    amount: float
    category: str
    description: str
    reason: Optional[str] = ""
    type: str  # "expense" | "income"
    timestamp: int
    provider: Optional[str] = ""
    model: Optional[str] = ""


class SaveRequest(BaseModel):
    username: str
    password: str
    expenses: List[ExpenseItem]


class SyncRequest(BaseModel):
    username: str
    password: str


class AuthCheckRequest(BaseModel):
    username: str


# ── Router ────────────────────────────────────────────────────────────────────
expense_mongo_router = APIRouter(prefix="/expenses", tags=["expenses"])


@expense_mongo_router.post("/check-user")
async def check_user(req: AuthCheckRequest):
    """Returns whether the username already exists (to show login vs register)."""
    db = get_db()
    user = await db.users.find_one({"username": req.username.strip().lower()})
    return {"exists": user is not None}


@expense_mongo_router.post("/save")
async def save_expenses(req: SaveRequest):
    """
    Register (first time) or login + overwrite cloud expenses with local data.
    Merges by id — adds new items, keeps existing ones not in local list.
    """
    db = get_db()
    uname = req.username.strip().lower()

    if not uname or len(uname) < 2:
        raise HTTPException(400, "Username must be at least 2 characters.")
    if not req.password or len(req.password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters.")

    user = await db.users.find_one({"username": uname})

    if user is None:
        # ── Register ──
        await db.users.insert_one({
            "username": uname,
            "password_hash": hash_password(req.password),
            "created_at": datetime.utcnow(),
        })
    else:
        # ── Login ──
        if not verify_password(req.password, user["password_hash"]):
            raise HTTPException(401, "Incorrect password.")

    # Fetch existing cloud expenses
    existing_cursor = db.expenses.find({"username": uname}, {"_id": 0})
    existing = {e["id"]: e async for e in existing_cursor}

    # Merge: local list wins for overlapping ids
    for item in req.expenses:
        existing[item.id] = {"username": uname, **item.dict()}

    merged = list(existing.values())

    if merged:
        # Replace all expenses for this user
        await db.expenses.delete_many({"username": uname})
        await db.expenses.insert_many(merged)

    return {
        "success": True,
        "saved": len(merged),
        "message": f"{'Registered and saved' if user is None else 'Saved'} {len(merged)} expenses.",
    }


@expense_mongo_router.post("/sync")
async def sync_expenses(req: SyncRequest):
    """Login and return all cloud expenses for this user."""
    db = get_db()
    uname = req.username.strip().lower()

    user = await db.users.find_one({"username": uname})
    if user is None:
        raise HTTPException(404, "Username not found. Please save data first to register.")
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Incorrect password.")

    cursor = db.expenses.find({"username": uname}, {"_id": 0, "username": 0})
    expenses = [e async for e in cursor]

    return {
        "success": True,
        "expenses": expenses,
        "count": len(expenses),
    }
