"""
community_routes.py  —  Community expense leaderboard for ExpenseTracker
Attach to main.py:
    from community_routes import community_router
    app.include_router(community_router)

MongoDB collections used:
    community_shares   — one doc per (username, month_key) holding category totals
    users              — existing collection (password check reused from expense_mongo_routes)
"""

import os
import hashlib
import hmac
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

MONGO_URL = os.getenv("MONGODB_URL", "")
SECRET    = os.getenv("EXPENSE_SECRET", "expense_tracker_secret_2025")

_client = None
_db     = None


def get_db():
    global _client, _db
    if not MONGO_AVAILABLE:
        raise HTTPException(500, "motor not installed — pip install motor")
    if not MONGO_URL:
        raise HTTPException(500, "MONGODB_URL env var not set")
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
        _db     = _client["expense_tracker"]
    return _db


def verify_password(password: str, hashed: str) -> bool:
    h = hmac.new(SECRET.encode(), password.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(h, hashed)


# ── Pydantic models ───────────────────────────────────────────────────────────

class CategoryTotal(BaseModel):
    category: str
    total: float     # total expense for this category in the month


class ShareRequest(BaseModel):
    username:          str
    password:          str
    month_key:         str                  # "2025-05"
    category_totals:   List[CategoryTotal]
    total_expense:     float
    show_real_name:    bool = True
    exclude_income:    bool = True          # always true — safety default


class UnshareRequest(BaseModel):
    username:  str
    password:  str
    month_key: str


class LeaderboardRequest(BaseModel):
    month_key: str
    category:  Optional[str] = None        # None = overall ranking


class MyRankRequest(BaseModel):
    username:  str
    password:  str
    month_key: str


# ── Router ────────────────────────────────────────────────────────────────────

community_router = APIRouter(prefix="/community", tags=["community"])


# ── Helper: authenticate user ─────────────────────────────────────────────────

async def _auth(db, username: str, password: str):
    user = await db.users.find_one({"username": username.strip().lower()})
    if not user:
        raise HTTPException(404, "Username not found. Save your expenses first to register.")
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(401, "Incorrect password.")
    return user


# ── POST /community/share ─────────────────────────────────────────────────────

@community_router.post("/share")
async def share_month(req: ShareRequest):
    """
    Publish (or update) a user's monthly category-level expense totals to the
    community leaderboard.  Individual transactions are NEVER stored here —
    only aggregated category totals and the grand total.
    """
    db    = get_db()
    uname = req.username.strip().lower()
    await _auth(db, uname, req.password)

    if not req.month_key or len(req.month_key) != 7:
        raise HTTPException(400, "month_key must be YYYY-MM format (e.g. '2025-05')")

    display_name = uname if req.show_real_name else f"anon_{abs(hash(uname)) % 10000:04d}"

    doc = {
        "username":       uname,
        "display_name":   display_name,
        "month_key":      req.month_key,
        "total_expense":  round(req.total_expense, 2),
        "category_totals": [
            {"category": ct.category, "total": round(ct.total, 2)}
            for ct in req.category_totals
            if ct.total > 0
        ],
        "show_real_name": req.show_real_name,
        "shared_at":      datetime.utcnow().isoformat(),
    }

    await db.community_shares.replace_one(
        {"username": uname, "month_key": req.month_key},
        doc,
        upsert=True,
    )

    # Rebuild indexes (idempotent — safe to call every time)
    await db.community_shares.create_index(
        [("month_key", 1), ("total_expense", -1)]
    )
    await db.community_shares.create_index(
        [("month_key", 1), ("category_totals.category", 1), ("category_totals.total", -1)]
    )

    return {"success": True, "message": f"Shared {req.month_key} successfully."}


# ── POST /community/unshare ───────────────────────────────────────────────────

@community_router.post("/unshare")
async def unshare_month(req: UnshareRequest):
    """Remove a previously shared month from the leaderboard."""
    db    = get_db()
    uname = req.username.strip().lower()
    await _auth(db, uname, req.password)

    result = await db.community_shares.delete_one(
        {"username": uname, "month_key": req.month_key}
    )
    if result.deleted_count == 0:
        raise HTTPException(404, "No shared data found for that month.")
    return {"success": True, "message": f"Removed {req.month_key} from leaderboard."}


# ── POST /community/leaderboard ───────────────────────────────────────────────

@community_router.post("/leaderboard")
async def get_leaderboard(req: LeaderboardRequest):
    """
    Returns the top-50 leaderboard for a given month.
    If `category` is provided, ranks by that category's spend.
    Otherwise ranks by total_expense.

    Also returns community stats: participant count, average spend,
    per-category winners.
    """
    db = get_db()

    if req.category:
        # ── Category-level leaderboard ────────────────────────────────────────
        pipeline = [
            {"$match": {"month_key": req.month_key}},
            {"$unwind": "$category_totals"},
            {"$match": {"category_totals.category": req.category}},
            {"$project": {
                "_id": 0,
                "username":     1,
                "display_name": 1,
                "month_key":    1,
                "amount":       "$category_totals.total",
            }},
            {"$sort": {"amount": -1}},
            {"$limit": 50},
        ]
        docs = await db.community_shares.aggregate(pipeline).to_list(length=50)

        # participant count for this category
        cat_count_pipeline = [
            {"$match": {"month_key": req.month_key}},
            {"$unwind": "$category_totals"},
            {"$match": {"category_totals.category": req.category}},
            {"$count": "total"},
        ]
        cnt_result = await db.community_shares.aggregate(cat_count_pipeline).to_list(length=1)
        participant_count = cnt_result[0]["total"] if cnt_result else 0

        entries = [
            {
                "rank":         i + 1,
                "username":     d["username"],
                "display_name": d["display_name"],
                "amount":       d["amount"],
            }
            for i, d in enumerate(docs)
        ]

        return {
            "success":          True,
            "month_key":        req.month_key,
            "category":         req.category,
            "entries":          entries,
            "participant_count": participant_count,
        }

    # ── Overall leaderboard ───────────────────────────────────────────────────
    cursor = db.community_shares.find(
        {"month_key": req.month_key},
        {"_id": 0, "username": 1, "display_name": 1,
         "total_expense": 1, "category_totals": 1},
    ).sort("total_expense", -1).limit(50)

    docs = await cursor.to_list(length=50)

    participant_count = await db.community_shares.count_documents(
        {"month_key": req.month_key}
    )

    # Compute average
    avg_pipeline = [
        {"$match": {"month_key": req.month_key}},
        {"$group": {"_id": None, "avg": {"$avg": "$total_expense"}}},
    ]
    avg_result = await db.community_shares.aggregate(avg_pipeline).to_list(length=1)
    avg_spend = round(avg_result[0]["avg"], 2) if avg_result else 0

    entries = [
        {
            "rank":          i + 1,
            "username":      d["username"],
            "display_name":  d["display_name"],
            "total_expense": d["total_expense"],
        }
        for i, d in enumerate(docs)
    ]

    # Per-category winners (top spender per category across all participants)
    cat_winners_pipeline = [
        {"$match": {"month_key": req.month_key}},
        {"$unwind": "$category_totals"},
        {"$sort": {"category_totals.total": -1}},
        {"$group": {
            "_id":          "$category_totals.category",
            "display_name": {"$first": "$display_name"},
            "username":     {"$first": "$username"},
            "top_amount":   {"$first": "$category_totals.total"},
            "participant_count": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]
    cat_winners_raw = await db.community_shares.aggregate(
        cat_winners_pipeline
    ).to_list(length=50)

    category_winners = [
        {
            "category":     cw["_id"],
            "winner_name":  cw["display_name"],
            "top_amount":   cw["top_amount"],
            "participant_count": cw["participant_count"],
        }
        for cw in cat_winners_raw
    ]

    # Available months (for the month picker)
    months_pipeline = [
        {"$group": {"_id": "$month_key"}},
        {"$sort": {"_id": -1}},
        {"$limit": 12},
    ]
    months_raw = await db.community_shares.aggregate(months_pipeline).to_list(length=12)
    available_months = [m["_id"] for m in months_raw]

    return {
        "success":           True,
        "month_key":         req.month_key,
        "category":          None,
        "entries":           entries,
        "participant_count": participant_count,
        "avg_spend":         avg_spend,
        "category_winners":  category_winners,
        "available_months":  available_months,
    }


# ── POST /community/my-rank ───────────────────────────────────────────────────

@community_router.post("/my-rank")
async def get_my_rank(req: MyRankRequest):
    """
    Returns the authenticated user's rank in the overall leaderboard and in
    each category for the given month.  Also includes percentile.
    """
    db    = get_db()
    uname = req.username.strip().lower()
    await _auth(db, uname, req.password)

    my_doc = await db.community_shares.find_one(
        {"username": uname, "month_key": req.month_key},
        {"_id": 0},
    )

    if not my_doc:
        return {
            "success":    True,
            "shared":     False,
            "message":    "You haven't shared this month yet. Use 'Share Month' to join the leaderboard.",
        }

    total_participants = await db.community_shares.count_documents(
        {"month_key": req.month_key}
    )

    # Overall rank = count of people who spent MORE than me
    overall_rank = await db.community_shares.count_documents({
        "month_key":     req.month_key,
        "total_expense": {"$gt": my_doc["total_expense"]},
    }) + 1

    percentile = round((1 - (overall_rank - 1) / max(total_participants, 1)) * 100, 1)

    # Per-category ranks
    category_ranks = []
    for ct in my_doc.get("category_totals", []):
        cat  = ct["category"]
        amt  = ct["total"]
        rank_in_cat = await db.community_shares.count_documents({
            "month_key":             req.month_key,
            "category_totals":       {
                "$elemMatch": {"category": cat, "total": {"$gt": amt}}
            },
        }) + 1

        cat_participant_count_pipeline = [
            {"$match": {"month_key": req.month_key}},
            {"$unwind": "$category_totals"},
            {"$match": {"category_totals.category": cat}},
            {"$count": "total"},
        ]
        cnt = await db.community_shares.aggregate(
            cat_participant_count_pipeline
        ).to_list(length=1)
        cat_total = cnt[0]["total"] if cnt else 1

        category_ranks.append({
            "category":          cat,
            "amount":            amt,
            "rank":              rank_in_cat,
            "total_in_category": cat_total,
            "percentile":        round((1 - (rank_in_cat - 1) / max(cat_total, 1)) * 100, 1),
        })

    category_ranks.sort(key=lambda x: x["rank"])

    return {
        "success":            True,
        "shared":             True,
        "month_key":          req.month_key,
        "total_expense":      my_doc["total_expense"],
        "overall_rank":       overall_rank,
        "total_participants": total_participants,
        "percentile":         percentile,
        "category_ranks":     category_ranks,
        "shared_at":          my_doc.get("shared_at"),
    }


# ── GET /community/available-months ──────────────────────────────────────────

@community_router.get("/available-months")
async def available_months():
    """Returns the list of months that have at least one shared entry."""
    db = get_db()
    pipeline = [
        {"$group": {"_id": "$month_key", "count": {"$sum": 1}}},
        {"$sort": {"_id": -1}},
        {"$limit": 24},
    ]
    months_raw = await db.community_shares.aggregate(pipeline).to_list(length=24)
    return {
        "months": [{"month_key": m["_id"], "participants": m["count"]} for m in months_raw]
    }


# ── GET /community/stats/{month_key} ─────────────────────────────────────────

@community_router.get("/stats/{month_key}")
async def month_stats(month_key: str):
    """Quick stats for a month — participant count, avg, median, top categories."""
    db = get_db()

    count = await db.community_shares.count_documents({"month_key": month_key})
    if count == 0:
        return {"success": True, "month_key": month_key, "participant_count": 0}

    pipeline = [
        {"$match": {"month_key": month_key}},
        {"$group": {
            "_id":  None,
            "avg":  {"$avg":  "$total_expense"},
            "min":  {"$min":  "$total_expense"},
            "max":  {"$max":  "$total_expense"},
            "sum":  {"$sum":  "$total_expense"},
        }},
    ]
    agg = await db.community_shares.aggregate(pipeline).to_list(length=1)
    stats = agg[0] if agg else {}

    return {
        "success":           True,
        "month_key":         month_key,
        "participant_count": count,
        "avg_expense":       round(stats.get("avg", 0), 2),
        "min_expense":       round(stats.get("min", 0), 2),
        "max_expense":       round(stats.get("max", 0), 2),
        "total_community":   round(stats.get("sum", 0), 2),
    }

# added by arnab