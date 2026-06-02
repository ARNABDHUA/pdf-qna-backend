"""
game_routes.py  –  Multiplayer game WebSocket + REST routes
Add to main.py:
    from game_routes import game_router, game_ws_router
    app.include_router(game_router)
    app.include_router(game_ws_router)
"""

import asyncio
import json
import random
import string
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

# ── Routers ───────────────────────────────────────────────────────────────────
game_router    = APIRouter(prefix="/games", tags=["games"])
game_ws_router = APIRouter(tags=["game-ws"])

# ── In-memory store ───────────────────────────────────────────────────────────
rooms: Dict[str, "Room"] = {}

GAME_CONFIGS = {
    "business": {"min_players": 2, "max_players": 8,  "rounds": 6},
    "spy":      {"min_players": 4, "max_players": 15, "rounds": None},
    "memory":   {"min_players": 2, "max_players": 8,  "rounds": None},
}

AVATARS = [
    "🦊","🐯","🦁","🐻","🦝","🐺","🦄","🐸","🐧","🦋",
    "🦊","🐙","🦈","🐲","🦩","🐳","🦅","🦎","🦔","🐝",
]


def make_room_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


# ── Room model ────────────────────────────────────────────────────────────────
class Player:
    def __init__(self, pid: str, name: str, avatar: str):
        self.id      = pid
        self.name    = name
        self.avatar  = avatar
        self.score   = 0
        self.ready   = False
        self.ws: Optional[WebSocket] = None
        self.role: Optional[str] = None        # for spy game
        self.eliminated = False


class Room:
    def __init__(self, room_id: str, game_type: str, host_id: str):
        self.id        = room_id
        self.game_type = game_type
        self.host_id   = host_id
        self.players:  Dict[str, Player] = {}
        self.state     = "lobby"   # lobby | playing | finished
        self.round     = 0
        self.phase     = ""        # game-specific phase
        self.data:     Dict[str, Any] = {}   # game-specific state
        self.chat:     List[Dict]    = []
        self.created   = time.time()

    # ── broadcast ──
    async def broadcast(self, msg: dict, exclude: Optional[str] = None):
        dead = []
        for pid, p in self.players.items():
            if pid == exclude or p.ws is None:
                continue
            try:
                await p.ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(pid)
        for pid in dead:
            self.players.pop(pid, None)

    async def send_to(self, pid: str, msg: dict):
        p = self.players.get(pid)
        if p and p.ws:
            try:
                await p.ws.send_text(json.dumps(msg))
            except Exception:
                pass

    # ── room snapshot (no ws objects) ──
    def snapshot(self, for_pid: Optional[str] = None) -> dict:
        players_out = []
        for p in self.players.values():
            pdata = {
                "id":          p.id,
                "name":        p.name,
                "avatar":      p.avatar,
                "score":       p.score,
                "ready":       p.ready,
                "eliminated":  p.eliminated,
                "isHost":      p.id == self.host_id,
            }
            # only reveal spy role to the player themselves
            if self.game_type == "spy" and for_pid == p.id:
                pdata["role"] = p.role
            elif self.game_type == "spy" and p.eliminated:
                pdata["role"] = p.role  # revealed on elimination
            players_out.append(pdata)

        return {
            "roomId":    self.id,
            "gameType":  self.game_type,
            "state":     self.state,
            "round":     self.round,
            "phase":     self.phase,
            "players":   players_out,
            "gameData":  self.data,
            "hostId":    self.host_id,
        }


# ── Game logic helpers ────────────────────────────────────────────────────────

def init_business_game(room: Room):
    cfg = GAME_CONFIGS["business"]
    room.data = {
        "maxRounds": cfg["rounds"],
        "market": {"demand": 70, "recession": False, "boom": False},
        "businesses": {
            pid: {
                "cash":          10000,
                "revenue":       0,
                "expenses":      0,
                "marketShare":   round(100 / len(room.players), 1),
                "satisfaction":  75,
                "employees":     5,
                "products":      1,
                "loans":         0,
                "brand":         50,
                "actions":       [],   # actions taken this round
                "submitted":     False,
            }
            for pid in room.players
        },
    }
    room.round = 1
    room.phase = "planning"


def init_spy_game(room: Room):
    n = len(room.players)
    num_spies = 1 if n <= 6 else 2 if n <= 10 else 3
    pids = list(room.players.keys())
    random.shuffle(pids)
    spies = set(pids[:num_spies])
    for pid, p in room.players.items():
        p.role = "spy" if pid in spies else "civilian"
    room.data = {
        "phase":       "day",
        "votes":       {},
        "numSpies":    num_spies,
        "eliminated":  [],
        "nightTarget": None,
        "spyIds":      list(spies),
    }
    room.round = 1
    room.phase = "day"


def init_memory_game(room: Room):
    emojis = ["🎯","🎸","🎺","🎻","🎹","🎨","🖼️","🎭","🎪","🎢",
              "🎡","🎠","🎲","🎯","🀄","🎴","🃏","🎰","🧩","🎮"]
    pairs  = emojis[:12]  # 12 pairs = 24 cards
    cards  = pairs * 2
    random.shuffle(cards)
    room.data = {
        "cards":       [{"id": i, "emoji": e, "matched": False, "flippedBy": None} for i, e in enumerate(cards)],
        "currentTurn": list(room.players.keys())[0],
        "flipped":     [],       # indices currently face-up (unmatched)
        "scores":      {pid: 0 for pid in room.players},
        "combo":       {pid: 0 for pid in room.players},
    }
    room.phase = "playing"


def apply_business_round(room: Room):
    """Resolve one business quarter."""
    mkt = room.data["market"]
    event = random.choice(["normal","normal","boom","recession","inflation","surge"])
    mkt["boom"]      = event == "boom"
    mkt["recession"] = event == "recession"
    demand_mod = 1.3 if mkt["boom"] else 0.7 if mkt["recession"] else 1.0

    for pid, biz in room.data["businesses"].items():
        actions = biz.get("actions", [])
        # base revenue
        base = biz["products"] * 1500 * demand_mod * (biz["marketShare"] / 100)
        if "advertise" in actions:
            biz["brand"]       = min(100, biz["brand"] + 8)
            biz["expenses"]   += 500
        if "hire" in actions:
            biz["employees"]  += 2
            biz["expenses"]   += 800
        if "expand" in actions:
            biz["products"]   += 1
            biz["expenses"]   += 1200
        if "train" in actions:
            biz["satisfaction"] = min(100, biz["satisfaction"] + 5)
        # revenue affected by brand & satisfaction
        rev = base * (1 + biz["brand"] / 200) * (biz["satisfaction"] / 100)
        biz["revenue"] += round(rev)
        net = round(rev - biz["expenses"])
        biz["cash"]    += net
        biz["expenses"] = round(biz["expenses"] * 0.1)  # reset partial
        biz["score"]    = round(biz["cash"] + biz["revenue"] * 0.3 + biz["brand"] * 50)
        biz["actions"]  = []
        biz["submitted"] = False
        # update score on player
        if pid in room.players:
            room.players[pid].score = biz["score"]

    room.data["market"]["event"] = event


def check_spy_winner(room: Room) -> Optional[str]:
    alive = [p for p in room.players.values() if not p.eliminated]
    spies = [p for p in alive if p.role == "spy"]
    civs  = [p for p in alive if p.role == "civilian"]
    if not spies:
        return "civilians"
    if len(spies) >= len(civs):
        return "spies"
    return None


# ── REST endpoints ────────────────────────────────────────────────────────────

class CreateRoomBody(BaseModel):
    gameType:   str
    playerName: str
    avatar:     str

class JoinRoomBody(BaseModel):
    playerName: str
    avatar:     str


@game_router.post("/create")
async def create_room(body: CreateRoomBody):
    if body.gameType not in GAME_CONFIGS:
        raise HTTPException(400, "Invalid game type")
    room_id  = make_room_id()
    player_id = "p_" + "".join(random.choices(string.ascii_lowercase, k=8))
    room     = Room(room_id, body.gameType, player_id)
    player   = Player(player_id, body.playerName, body.avatar)
    player.ready = True
    room.players[player_id] = player
    rooms[room_id] = room
    return {"roomId": room_id, "playerId": player_id, "room": room.snapshot(player_id)}


@game_router.post("/join/{room_id}")
async def join_room(room_id: str, body: JoinRoomBody):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "Room not found")
    if room.state != "lobby":
        raise HTTPException(400, "Game already started")
    cfg = GAME_CONFIGS[room.game_type]
    if len(room.players) >= cfg["max_players"]:
        raise HTTPException(400, "Room is full")

    player_id = "p_" + "".join(random.choices(string.ascii_lowercase, k=8))
    player    = Player(player_id, body.playerName, body.avatar)
    room.players[player_id] = player
    return {"roomId": room_id, "playerId": player_id, "room": room.snapshot(player_id)}


@game_router.get("/{room_id}")
async def get_room(room_id: str, playerId: str = ""):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "Room not found")
    return room.snapshot(playerId)


@game_router.get("/avatars/list")
async def list_avatars():
    return {"avatars": AVATARS}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@game_ws_router.websocket("/ws/game/{room_id}/{player_id}")
async def game_ws(ws: WebSocket, room_id: str, player_id: str):
    await ws.accept()

    room   = rooms.get(room_id)
    player = room.players.get(player_id) if room else None

    if not room or not player:
        await ws.send_text(json.dumps({"type": "error", "msg": "Room or player not found"}))
        await ws.close()
        return

    player.ws = ws

    # announce join
    await room.broadcast({"type": "player_joined", "player": {
        "id": player.id, "name": player.name, "avatar": player.avatar,
        "score": player.score, "ready": player.ready, "isHost": player.id == room.host_id,
        "eliminated": player.eliminated,
    }}, exclude=player_id)

    # send full state to joining player
    await ws.send_text(json.dumps({"type": "room_state", **room.snapshot(player_id)}))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            # ── chat ──────────────────────────────────────────────────────────
            if mtype == "chat":
                entry = {"from": player.name, "avatar": player.avatar, "text": msg.get("text",""), "ts": time.time()}
                room.chat.append(entry)
                await room.broadcast({"type": "chat", **entry})

            # ── ready toggle ──────────────────────────────────────────────────
            elif mtype == "ready":
                player.ready = not player.ready
                await room.broadcast({"type": "player_ready", "playerId": player_id, "ready": player.ready})

            # ── start game (host only) ────────────────────────────────────────
            elif mtype == "start_game":
                if player_id != room.host_id:
                    continue
                cfg = GAME_CONFIGS[room.game_type]
                if len(room.players) < cfg["min_players"]:
                    await ws.send_text(json.dumps({"type": "error", "msg": f"Need at least {cfg['min_players']} players"}))
                    continue
                room.state = "playing"
                if room.game_type == "business":
                    init_business_game(room)
                elif room.game_type == "spy":
                    init_spy_game(room)
                elif room.game_type == "memory":
                    init_memory_game(room)
                # send personalised state to each player
                for pid, p in room.players.items():
                    if p.ws:
                        await p.ws.send_text(json.dumps({"type": "game_started", **room.snapshot(pid)}))

            # ── BUSINESS actions ──────────────────────────────────────────────
            elif mtype == "biz_action":
                if room.game_type != "business" or room.state != "playing":
                    continue
                biz = room.data["businesses"].get(player_id, {})
                action = msg.get("action")
                if action and action not in biz.get("actions", []):
                    biz.setdefault("actions", []).append(action)
                await ws.send_text(json.dumps({"type": "biz_update", "biz": biz}))

            elif mtype == "biz_submit":
                if room.game_type != "business" or room.state != "playing":
                    continue
                room.data["businesses"][player_id]["submitted"] = True
                # check if all submitted
                all_done = all(b["submitted"] for b in room.data["businesses"].values())
                if all_done:
                    apply_business_round(room)
                    cfg = GAME_CONFIGS["business"]
                    if room.round >= cfg["rounds"]:
                        room.state = "finished"
                        await room.broadcast({"type": "game_over", **room.snapshot()})
                    else:
                        room.round += 1
                        await room.broadcast({"type": "round_result", **room.snapshot()})
                else:
                    await room.broadcast({"type": "biz_waiting", "submitted": [
                        pid for pid, b in room.data["businesses"].items() if b["submitted"]
                    ]})

            # ── SPY actions ───────────────────────────────────────────────────
            elif mtype == "spy_vote":
                if room.game_type != "spy" or room.phase != "day":
                    continue
                target = msg.get("target")
                if target in room.players:
                    room.data["votes"][player_id] = target
                # check all alive non-eliminated players voted
                alive_ids = [p.id for p in room.players.values() if not p.eliminated]
                if len(room.data["votes"]) >= len(alive_ids):
                    # tally
                    tally: Dict[str, int] = {}
                    for v in room.data["votes"].values():
                        tally[v] = tally.get(v, 0) + 1
                    eliminated_id = max(tally, key=lambda k: tally[k])
                    elim = room.players[eliminated_id]
                    elim.eliminated = True
                    room.data["eliminated"].append({"id": elim.id, "name": elim.name, "role": elim.role})
                    room.data["votes"] = {}
                    winner = check_spy_winner(room)
                    if winner:
                        room.state = "finished"
                        await room.broadcast({"type": "game_over", "winner": winner, **room.snapshot()})
                    else:
                        room.phase = "night"
                        room.data["phase"] = "night"
                        await room.broadcast({"type": "day_result", "eliminated": {"id": elim.id, "name": elim.name, "role": elim.role, "avatar": elim.avatar}, **room.snapshot()})

            elif mtype == "spy_night_target":
                if room.game_type != "spy" or room.phase != "night":
                    continue
                if room.players[player_id].role != "spy":
                    continue
                room.data["nightTarget"] = msg.get("target")
                # if all spies agreed (simplified: first vote wins)
                target_id = room.data["nightTarget"]
                if target_id and target_id in room.players:
                    victim = room.players[target_id]
                    victim.eliminated = True
                    room.data["eliminated"].append({"id": victim.id, "name": victim.name, "role": victim.role})
                    room.data["nightTarget"] = None
                    room.round += 1
                    room.phase = "day"
                    room.data["phase"] = "day"
                    winner = check_spy_winner(room)
                    if winner:
                        room.state = "finished"
                        await room.broadcast({"type": "game_over", "winner": winner, **room.snapshot()})
                    else:
                        for pid, p in room.players.items():
                            if p.ws:
                                await p.ws.send_text(json.dumps({"type": "night_result", "victim": {"id": victim.id, "name": victim.name, "avatar": victim.avatar}, **room.snapshot(pid)}))

            # ── MEMORY actions ────────────────────────────────────────────────
            elif mtype == "memory_flip":
                if room.game_type != "memory" or room.state != "playing":
                    continue
                if room.data["currentTurn"] != player_id:
                    continue
                card_idx = msg.get("cardIdx")
                cards    = room.data["cards"]
                flipped  = room.data["flipped"]
                if card_idx is None or cards[card_idx]["matched"]:
                    continue
                if card_idx in flipped:
                    continue
                flipped.append(card_idx)
                cards[card_idx]["flippedBy"] = player_id
                await room.broadcast({"type": "card_flipped", "cardIdx": card_idx, "emoji": cards[card_idx]["emoji"], "playerId": player_id})

                if len(flipped) == 2:
                    a, b = flipped
                    if cards[a]["emoji"] == cards[b]["emoji"]:
                        # match!
                        cards[a]["matched"] = cards[b]["matched"] = True
                        combo = room.data["combo"][player_id] + 1
                        room.data["combo"][player_id] = combo
                        pts = 1 + (2 if combo > 1 else 0)
                        room.data["scores"][player_id] = room.data["scores"].get(player_id, 0) + pts
                        room.players[player_id].score = room.data["scores"][player_id]
                        room.data["flipped"] = []
                        await room.broadcast({"type": "match_found", "cardA": a, "cardB": b, "playerId": player_id, "points": pts, "scores": room.data["scores"]})
                        # check all matched
                        if all(c["matched"] for c in cards):
                            room.state = "finished"
                            await room.broadcast({"type": "game_over", **room.snapshot()})
                    else:
                        # no match – pass turn
                        for cid in [a, b]:
                            room.data["combo"][player_id] = 0
                        room.data["flipped"] = []
                        # next player
                        alive = [pid for pid in room.players if not room.players[pid].eliminated]
                        cur_idx = alive.index(player_id) if player_id in alive else 0
                        next_pid = alive[(cur_idx + 1) % len(alive)]
                        room.data["currentTurn"] = next_pid
                        await room.broadcast({"type": "no_match", "cardA": a, "cardB": b, "nextTurn": next_pid})

            # ── mic signal (relay peer info) ──────────────────────────────────
            elif mtype == "mic_signal":
                target_id = msg.get("to")
                if target_id in room.players:
                    await room.send_to(target_id, {"type": "mic_signal", "from": player_id, "signal": msg.get("signal")})

    except WebSocketDisconnect:
        player.ws = None
        await room.broadcast({"type": "player_left", "playerId": player_id, "name": player.name})
        # if room empty, clean up
        active = [p for p in room.players.values() if p.ws is not None]
        if not active:
            rooms.pop(room_id, None)