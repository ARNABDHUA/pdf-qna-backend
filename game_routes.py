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
    "business":  {"min_players": 2, "max_players": 8,  "rounds": 6},
    "spy":       {"min_players": 4, "max_players": 15, "rounds": None},
    "memory":    {"min_players": 2, "max_players": 8,  "rounds": None},
    "monopoly":  {"min_players": 2, "max_players": 8,  "rounds": None},
    "trivia":    {"min_players": 2, "max_players": 15, "rounds": 10},
}

AVATARS = [
    "🦊","🐯","🦁","🐻","🦝","🐺","🦄","🐸","🐧","🦋",
    "🐙","🦈","🐲","🦩","🐳","🦅","🦎","🦔","🐝","🦖",
]

# ── Monopoly Board Data ───────────────────────────────────────────────────────
BOARD_TILES = [
    {"id": 0,  "type": "corner",    "name": "GO"},
    {"id": 1,  "type": "property",  "name": "Mediterranean Ave", "group": "brown",    "price": 60,  "rent": [2,10,30,90,160,250],    "houseCost": 50},
    {"id": 2,  "type": "community", "name": "Community Chest"},
    {"id": 3,  "type": "property",  "name": "Baltic Avenue",     "group": "brown",    "price": 60,  "rent": [4,20,60,180,320,450],   "houseCost": 50},
    {"id": 4,  "type": "tax",       "name": "Income Tax",        "amount": 200},
    {"id": 5,  "type": "railroad",  "name": "Reading Railroad",  "price": 200},
    {"id": 6,  "type": "property",  "name": "Oriental Ave",      "group": "lightBlue","price": 100, "rent": [6,30,90,270,400,550],   "houseCost": 50},
    {"id": 7,  "type": "chance",    "name": "Chance"},
    {"id": 8,  "type": "property",  "name": "Vermont Ave",       "group": "lightBlue","price": 100, "rent": [6,30,90,270,400,550],   "houseCost": 50},
    {"id": 9,  "type": "property",  "name": "Connecticut Ave",   "group": "lightBlue","price": 120, "rent": [8,40,100,300,450,600],  "houseCost": 50},
    {"id": 10, "type": "corner",    "name": "Jail / Visiting"},
    {"id": 11, "type": "property",  "name": "St. Charles Place", "group": "pink",     "price": 140, "rent": [10,50,150,450,625,750], "houseCost": 100},
    {"id": 12, "type": "utility",   "name": "Electric Company",  "price": 150},
    {"id": 13, "type": "property",  "name": "States Ave",        "group": "pink",     "price": 140, "rent": [10,50,150,450,625,750], "houseCost": 100},
    {"id": 14, "type": "property",  "name": "Virginia Ave",      "group": "pink",     "price": 160, "rent": [12,60,180,500,700,900], "houseCost": 100},
    {"id": 15, "type": "railroad",  "name": "Pennsylvania RR",   "price": 200},
    {"id": 16, "type": "property",  "name": "St. James Place",   "group": "orange",   "price": 180, "rent": [14,70,200,550,750,950], "houseCost": 100},
    {"id": 17, "type": "community", "name": "Community Chest"},
    {"id": 18, "type": "property",  "name": "Tennessee Ave",     "group": "orange",   "price": 180, "rent": [14,70,200,550,750,950], "houseCost": 100},
    {"id": 19, "type": "property",  "name": "New York Ave",      "group": "orange",   "price": 200, "rent": [16,80,220,600,800,1000],"houseCost": 100},
    {"id": 20, "type": "corner",    "name": "Free Parking"},
    {"id": 21, "type": "property",  "name": "Kentucky Ave",      "group": "red",      "price": 220, "rent": [18,90,250,700,875,1050],"houseCost": 150},
    {"id": 22, "type": "chance",    "name": "Chance"},
    {"id": 23, "type": "property",  "name": "Indiana Ave",       "group": "red",      "price": 220, "rent": [18,90,250,700,875,1050],"houseCost": 150},
    {"id": 24, "type": "property",  "name": "Illinois Ave",      "group": "red",      "price": 240, "rent": [20,100,300,750,925,1100],"houseCost": 150},
    {"id": 25, "type": "railroad",  "name": "B&O Railroad",      "price": 200},
    {"id": 26, "type": "property",  "name": "Atlantic Ave",      "group": "yellow",   "price": 260, "rent": [22,110,330,800,975,1150],"houseCost": 150},
    {"id": 27, "type": "property",  "name": "Ventnor Ave",       "group": "yellow",   "price": 260, "rent": [22,110,330,800,975,1150],"houseCost": 150},
    {"id": 28, "type": "utility",   "name": "Water Works",       "price": 150},
    {"id": 29, "type": "property",  "name": "Marvin Gardens",    "group": "yellow",   "price": 280, "rent": [24,120,360,850,1025,1200],"houseCost": 150},
    {"id": 30, "type": "corner",    "name": "Go to Jail"},
    {"id": 31, "type": "property",  "name": "Pacific Ave",       "group": "green",    "price": 300, "rent": [26,130,390,900,1100,1275],"houseCost": 200},
    {"id": 32, "type": "property",  "name": "North Carolina Ave","group": "green",    "price": 300, "rent": [26,130,390,900,1100,1275],"houseCost": 200},
    {"id": 33, "type": "community", "name": "Community Chest"},
    {"id": 34, "type": "property",  "name": "Pennsylvania Ave",  "group": "green",    "price": 320, "rent": [28,150,450,1000,1200,1400],"houseCost": 200},
    {"id": 35, "type": "railroad",  "name": "Short Line RR",     "price": 200},
    {"id": 36, "type": "chance",    "name": "Chance"},
    {"id": 37, "type": "property",  "name": "Park Place",        "group": "darkBlue", "price": 350, "rent": [35,175,500,1100,1300,1500],"houseCost": 200},
    {"id": 38, "type": "tax",       "name": "Luxury Tax",        "amount": 75},
    {"id": 39, "type": "property",  "name": "Boardwalk",         "group": "darkBlue", "price": 400, "rent": [50,200,600,1400,1700,2000],"houseCost": 200},
]

CHANCE_CARDS = [
    {"text": "Advance to GO. Collect $200.", "action": "move", "target": 0, "reward": 200},
    {"text": "Bank pays you a dividend of $50.", "action": "earn", "amount": 50},
    {"text": "Get out of Jail Free.", "action": "jailFree"},
    {"text": "Go to Jail. Do not pass GO.", "action": "jail"},
    {"text": "Pay poor tax of $15.", "action": "pay", "amount": 15},
    {"text": "You won a crossword competition! Collect $100.", "action": "earn", "amount": 100},
    {"text": "Your building loan matures. Collect $150.", "action": "earn", "amount": 150},
    {"text": "Pay school fees of $150.", "action": "pay", "amount": 150},
    {"text": "Speeding fine. Pay $15.", "action": "pay", "amount": 15},
    {"text": "You are assessed for street repairs: $25 per house.", "action": "repairs", "cost": 25},
    {"text": "Advance to Illinois Ave.", "action": "move", "target": 24, "reward": 0},
    {"text": "Advance to St. Charles Place.", "action": "move", "target": 11, "reward": 0},
    {"text": "Go back 3 spaces.", "action": "moveRel", "delta": -3},
    {"text": "Bank error in your favor. Collect $200.", "action": "earn", "amount": 200},
]

COMMUNITY_CARDS = [
    {"text": "Advance to GO. Collect $200.", "action": "move", "target": 0, "reward": 200},
    {"text": "Bank error in your favor. Collect $200.", "action": "earn", "amount": 200},
    {"text": "Doctor's fee. Pay $50.", "action": "pay", "amount": 50},
    {"text": "From sale of stock you get $50.", "action": "earn", "amount": 50},
    {"text": "Get out of Jail Free.", "action": "jailFree"},
    {"text": "Go to Jail.", "action": "jail"},
    {"text": "Holiday fund matures. Collect $100.", "action": "earn", "amount": 100},
    {"text": "Income tax refund. Collect $20.", "action": "earn", "amount": 20},
    {"text": "Life insurance matures. Collect $100.", "action": "earn", "amount": 100},
    {"text": "Pay hospital fees of $100.", "action": "pay", "amount": 100},
    {"text": "Pay school fees of $50.", "action": "pay", "amount": 50},
    {"text": "Receive $25 consultancy fee.", "action": "earn", "amount": 25},
    {"text": "You inherit $100.", "action": "earn", "amount": 100},
    {"text": "You have won second prize in a beauty contest. Collect $10.", "action": "earn", "amount": 10},
]

STARTING_CASH = 1500
JAIL_BAIL = 50
GO_REWARD = 200


def make_room_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


# ── Room model ────────────────────────────────────────────────────────────────
class Player:
    def __init__(self, pid: str, name: str, avatar: str):
        self.id          = pid
        self.name        = name
        self.avatar      = avatar
        self.score       = 0
        self.ready       = False
        self.ws: Optional[WebSocket] = None
        self.role: Optional[str] = None
        self.eliminated  = False


class Room:
    def __init__(self, room_id: str, game_type: str, host_id: str):
        self.id        = room_id
        self.game_type = game_type
        self.host_id   = host_id
        self.players:  Dict[str, Player] = {}
        self.state     = "lobby"
        self.round     = 0
        self.phase     = ""
        self.data:     Dict[str, Any] = {}
        self.chat:     List[Dict]    = []
        self.created   = time.time()

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

    def snapshot(self, for_pid: Optional[str] = None) -> dict:
        players_out = []
        for p in self.players.values():
            pdata = {
                "id":         p.id,
                "name":       p.name,
                "avatar":     p.avatar,
                "score":      p.score,
                "ready":      p.ready,
                "eliminated": p.eliminated,
                "isHost":     p.id == self.host_id,
            }
            if self.game_type == "spy" and for_pid == p.id:
                pdata["role"] = p.role
            elif self.game_type == "spy" and p.eliminated:
                pdata["role"] = p.role
            players_out.append(pdata)

        return {
            "roomId":   self.id,
            "gameType": self.game_type,
            "state":    self.state,
            "round":    self.round,
            "phase":    self.phase,
            "players":  players_out,
            "gameData": self.data,
            "hostId":   self.host_id,
        }


# ── Monopoly helpers ──────────────────────────────────────────────────────────

def init_monopoly_game(room: Room):
    pids = list(room.players.keys())
    properties = {}
    for tile in BOARD_TILES:
        if tile["type"] in ("property", "railroad", "utility"):
            properties[str(tile["id"])] = {
                "ownerId":   None,
                "houses":    0,
                "hotel":     False,
                "mortgaged": False,
            }

    players_state = []
    for pid in pids:
        players_state.append({
            "id":           pid,
            "cash":         STARTING_CASH,
            "position":     0,
            "inJail":       False,
            "jailTurns":    0,
            "hasJailCard":  False,
            "properties":   [],
            "netWorth":     STARTING_CASH,
            "eliminated":   False,
            "turnsPlayed":  0,
        })

    room.data = {
        "players":          players_state,
        "properties":       properties,
        "currentPlayerIdx": 0,
        "dice":             [0, 0],
        "phase":            "roll",      # roll | buy | auction | trade | gameover
        "log":              ["🎲 Monopoly started! Each player receives $1,500."],
        "round":            1,
        "winner":           None,
        "freeParking":      0,
        "lastCard":         None,
        "doublesCount":     0,
        "auction":          None,        # {tileId, bids:{pid:amount}, timer, highBidder}
        "tradeOffer":       None,        # {fromId, toId, offerCash, offerProps, wantCash, wantProps, status}
        "gameTimer":        time.time() + 3600,  # 1 hour game timer
    }
    room.state = "playing"
    room.phase = "roll"


def monopoly_check_color_monopoly(gs: dict, owner_id: str, group: str) -> bool:
    group_tiles = [t for t in BOARD_TILES if t.get("group") == group]
    return all(gs["properties"].get(str(t["id"]), {}).get("ownerId") == owner_id for t in group_tiles)


def monopoly_calc_net_worth(player: dict, properties: dict) -> int:
    prop_value = 0
    for pid in player.get("properties", []):
        tile = next((t for t in BOARD_TILES if t["id"] == pid), None)
        prop = properties.get(str(pid), {})
        if not tile or not prop:
            continue
        base = tile.get("price", 0) * (0.5 if prop.get("mortgaged") else 1)
        buildings = (tile.get("houseCost", 0) * 5 if prop.get("hotel")
                     else prop.get("houses", 0) * tile.get("houseCost", 0))
        prop_value += base + buildings
    return int(player["cash"] + prop_value)


def monopoly_next_turn(gs: dict) -> dict:
    players = gs["players"]
    idx = (gs["currentPlayerIdx"] + 1) % len(players)
    while players[idx].get("eliminated"):
        idx = (idx + 1) % len(players)
    gs["currentPlayerIdx"] = idx
    gs["phase"] = "roll"
    gs["doublesCount"] = 0
    if idx == 0:
        gs["round"] = gs.get("round", 1) + 1
    return gs


def monopoly_eliminate(gs: dict, pid: str) -> dict:
    idx = next((i for i, p in enumerate(gs["players"]) if p["id"] == pid), -1)
    if idx == -1:
        return gs
    player = gs["players"][idx]
    # return properties to bank
    for prop_id in player.get("properties", []):
        gs["properties"][str(prop_id)] = {"ownerId": None, "houses": 0, "hotel": False, "mortgaged": False}
    player["properties"] = []
    player["eliminated"] = True
    player["cash"] = 0
    gs["log"].append(f"💀 {player['name']} went bankrupt and is eliminated!")
    # check winner
    active = [p for p in gs["players"] if not p.get("eliminated")]
    if len(active) == 1:
        gs["winner"] = active[0]["id"]
        gs["phase"] = "gameover"
    return gs


def monopoly_process_tile(gs: dict, pidx: int, tile: dict, dice: list) -> dict:
    player = gs["players"][pidx]
    ttype  = tile["type"]
    doubles = dice[0] == dice[1]

    if ttype == "corner":
        if tile["name"] == "Go to Jail":
            player["inJail"] = True
            player["position"] = 10
            gs["log"].append(f"👮 {player['name']} sent to Jail!")
            return monopoly_next_turn(gs)
        return gs

    if ttype == "tax":
        amount = tile.get("amount", 0)
        player["cash"] -= amount
        gs["log"].append(f"💸 {player['name']} paid {tile['name']}: ${amount}")
        if player["cash"] < 0:
            gs = monopoly_eliminate(gs, player["id"])
        else:
            if not doubles:
                gs = monopoly_next_turn(gs)
        return gs

    if ttype in ("chance", "community"):
        deck = CHANCE_CARDS if ttype == "chance" else COMMUNITY_CARDS
        card = random.choice(deck)
        gs["lastCard"] = {"deck": "Chance" if ttype == "chance" else "Community", **card}
        gs["log"].append(f"{'❓' if ttype == 'chance' else '📦'} {player['name']}: \"{card['text']}\"")

        action = card["action"]
        if action == "earn":
            player["cash"] += card["amount"]
        elif action == "pay":
            player["cash"] -= card["amount"]
            if player["cash"] < 0:
                gs = monopoly_eliminate(gs, player["id"])
                return gs
        elif action == "move":
            if player["position"] > card["target"] and card["target"] == 0:
                player["cash"] += GO_REWARD
            player["position"] = card["target"]
            new_tile = BOARD_TILES[card["target"]]
            gs = monopoly_process_tile(gs, pidx, new_tile, dice)
            return gs
        elif action == "moveRel":
            player["position"] = (player["position"] + card["delta"]) % 40
            new_tile = BOARD_TILES[player["position"]]
            gs = monopoly_process_tile(gs, pidx, new_tile, dice)
            return gs
        elif action == "jail":
            player["inJail"] = True
            player["position"] = 10
            gs = monopoly_next_turn(gs)
            return gs
        elif action == "jailFree":
            player["hasJailCard"] = True
        elif action == "repairs":
            houses = sum(gs["properties"].get(str(pid), {}).get("houses", 0) for pid in player.get("properties", []))
            cost = houses * card["cost"]
            player["cash"] -= cost
            if player["cash"] < 0:
                gs = monopoly_eliminate(gs, player["id"])
                return gs

        if not doubles:
            gs = monopoly_next_turn(gs)
        else:
            gs["phase"] = "roll"
        return gs

    if ttype in ("property", "railroad", "utility"):
        prop = gs["properties"].get(str(tile["id"]))
        if not prop:
            if not doubles:
                gs = monopoly_next_turn(gs)
            return gs

        if prop["ownerId"] is None:
            gs["phase"] = "buy"
            return gs

        if prop["ownerId"] == player["id"]:
            gs["log"].append(f"🏠 {player['name']} owns {tile['name']}.")
            if not doubles:
                gs = monopoly_next_turn(gs)
            return gs

        # Pay rent
        owner = next((p for p in gs["players"] if p["id"] == prop["ownerId"]), None)
        if not owner or prop.get("mortgaged"):
            if not doubles:
                gs = monopoly_next_turn(gs)
            return gs

        rent = 0
        if ttype == "railroad":
            rr_count = sum(1 for pid in owner.get("properties", []) if BOARD_TILES[pid]["type"] == "railroad")
            rent = 25 * (2 ** (rr_count - 1))
        elif ttype == "utility":
            util_count = sum(1 for pid in owner.get("properties", []) if BOARD_TILES[pid]["type"] == "utility")
            roll = dice[0] + dice[1]
            rent = roll * (10 if util_count == 2 else 4)
        else:
            level = 5 if prop.get("hotel") else prop.get("houses", 0)
            rent = tile.get("rent", [0] * 6)[level]
            if level == 0 and monopoly_check_color_monopoly(gs, owner["id"], tile.get("group", "")):
                rent *= 2

        player["cash"] -= rent
        owner["cash"] += rent
        gs["log"].append(f"💰 {player['name']} paid ${rent} rent to {owner['name']} for {tile['name']}")

        if player["cash"] < 0:
            # transfer player assets to creditor before elimination
            owner["cash"] += player["cash"]  # net (negative portion)
            player["cash"] = 0
            gs = monopoly_eliminate(gs, player["id"])
            return gs

        if not doubles:
            gs = monopoly_next_turn(gs)
        return gs

    if not doubles:
        gs = monopoly_next_turn(gs)
    return gs


def monopoly_apply_roll(gs: dict, pid: str) -> dict:
    pidx = gs["currentPlayerIdx"]
    player = gs["players"][pidx]
    if player["id"] != pid:
        return gs

    dice = [random.randint(1, 6), random.randint(1, 6)]
    doubles = dice[0] == dice[1]
    gs["dice"] = dice
    gs["doublesCount"] = (gs.get("doublesCount", 0) + 1) if doubles else 0

    # 3 doubles in a row → jail
    if gs["doublesCount"] >= 3:
        player["inJail"] = True
        player["position"] = 10
        gs["log"].append(f"🎲 {player['name']} rolled three doubles — Jail!")
        gs = monopoly_next_turn(gs)
        return gs

    # In jail
    if player["inJail"]:
        if doubles:
            player["inJail"] = False
            player["jailTurns"] = 0
            gs["log"].append(f"🎲 {player['name']} rolled doubles and escaped jail! ({dice[0]}+{dice[1]})")
        else:
            player["jailTurns"] = player.get("jailTurns", 0) + 1
            if player["jailTurns"] >= 3:
                player["cash"] -= JAIL_BAIL
                player["inJail"] = False
                player["jailTurns"] = 0
                gs["log"].append(f"💸 {player['name']} paid ${JAIL_BAIL} bail after 3 turns.")
            else:
                gs["log"].append(f"🔒 {player['name']} is in jail (turn {player['jailTurns']}/3). Rolled {dice[0]}+{dice[1]}.")
                gs = monopoly_next_turn(gs)
                return gs

    steps = dice[0] + dice[1]
    old_pos = player["position"]
    new_pos = (old_pos + steps) % 40
    if new_pos < old_pos and new_pos != 0:
        player["cash"] += GO_REWARD
        gs["log"].append(f"🚦 {player['name']} passed GO! +${GO_REWARD}")
    player["position"] = new_pos
    player["turnsPlayed"] = player.get("turnsPlayed", 0) + 1

    tile = BOARD_TILES[new_pos]
    gs["log"].append(f"🎲 {player['name']} rolled {dice[0]}+{dice[1]}={steps} → {tile['name']}")

    gs = monopoly_process_tile(gs, pidx, tile, dice)

    # Update net worths
    for p in gs["players"]:
        p["netWorth"] = monopoly_calc_net_worth(p, gs["properties"])

    return gs


def monopoly_start_auction(gs: dict, tile_id: int, skip_pid: str) -> dict:
    gs["auction"] = {
        "tileId":     tile_id,
        "startPrice": max(10, BOARD_TILES[tile_id].get("price", 60) // 2),
        "bids":       {},
        "highBidder": None,
        "highBid":    0,
        "startTime":  time.time(),
        "duration":   30,  # seconds
        "skipPid":    skip_pid,
    }
    gs["phase"] = "auction"
    gs["log"].append(f"🔨 Auction started for {BOARD_TILES[tile_id]['name']}!")
    return gs


def monopoly_resolve_auction(gs: dict) -> dict:
    auction = gs.get("auction")
    if not auction:
        return gs
    if auction["highBidder"] and auction["highBid"] > 0:
        winner = next((p for p in gs["players"] if p["id"] == auction["highBidder"]), None)
        tile = BOARD_TILES[auction["tileId"]]
        if winner and winner["cash"] >= auction["highBid"]:
            winner["cash"] -= auction["highBid"]
            winner["properties"].append(auction["tileId"])
            gs["properties"][str(auction["tileId"])]["ownerId"] = winner["id"]
            gs["log"].append(f"🏆 {winner['name']} won auction for {tile['name']} at ${auction['highBid']}!")
        else:
            gs["log"].append(f"🔨 Auction ended with no valid winner for {tile['name']}.")
    else:
        tile = BOARD_TILES[auction["tileId"]]
        gs["log"].append(f"🔨 No bids for {tile['name']}. Property returned to bank.")

    gs["auction"] = None
    gs = monopoly_next_turn(gs)
    return gs


# ── Business game helpers ─────────────────────────────────────────────────────

def init_business_game(room: Room):
    cfg = GAME_CONFIGS["business"]
    room.data = {
        "maxRounds": cfg["rounds"],
        "market": {"demand": 70, "recession": False, "boom": False},
        "businesses": {
            pid: {
                "cash":         10000,
                "revenue":      0,
                "expenses":     0,
                "marketShare":  round(100 / len(room.players), 1),
                "satisfaction": 75,
                "employees":    5,
                "products":     1,
                "loans":        0,
                "brand":        50,
                "actions":      [],
                "submitted":    False,
            }
            for pid in room.players
        },
    }
    room.round = 1
    room.phase = "planning"


def apply_business_round(room: Room):
    mkt = room.data["market"]
    event = random.choice(["normal","normal","boom","recession","inflation","surge"])
    mkt["boom"]      = event == "boom"
    mkt["recession"] = event == "recession"
    demand_mod = 1.3 if mkt["boom"] else 0.7 if mkt["recession"] else 1.0

    for pid, biz in room.data["businesses"].items():
        actions = biz.get("actions", [])
        base = biz["products"] * 1500 * demand_mod * (biz["marketShare"] / 100)
        if "advertise" in actions:
            biz["brand"]     = min(100, biz["brand"] + 8)
            biz["expenses"] += 500
        if "hire" in actions:
            biz["employees"] += 2
            biz["expenses"]  += 800
        if "expand" in actions:
            biz["products"]  += 1
            biz["expenses"]  += 1200
        if "train" in actions:
            biz["satisfaction"] = min(100, biz["satisfaction"] + 5)
        rev = base * (1 + biz["brand"] / 200) * (biz["satisfaction"] / 100)
        biz["revenue"] += round(rev)
        net = round(rev - biz["expenses"])
        biz["cash"]    += net
        biz["expenses"] = round(biz["expenses"] * 0.1)
        biz["score"]    = round(biz["cash"] + biz["revenue"] * 0.3 + biz["brand"] * 50)
        biz["actions"]  = []
        biz["submitted"] = False
        if pid in room.players:
            room.players[pid].score = biz["score"]
    room.data["market"]["event"] = event


# ── Spy game helpers ──────────────────────────────────────────────────────────

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


def check_spy_winner(room: Room) -> Optional[str]:
    alive = [p for p in room.players.values() if not p.eliminated]
    spies = [p for p in alive if p.role == "spy"]
    civs  = [p for p in alive if p.role == "civilian"]
    if not spies:
        return "civilians"
    if len(spies) >= len(civs):
        return "spies"
    return None


# ── Memory game helpers ───────────────────────────────────────────────────────

def init_memory_game(room: Room):
    emojis = ["🎯","🎸","🎺","🎻","🎹","🎨","🖼️","🎭","🎪","🎢",
              "🎡","🎠","🎲","🀄","🎴","🃏","🎰","🧩","🎮","🏆"]
    pairs  = emojis[:12]
    cards  = pairs * 2
    random.shuffle(cards)
    room.data = {
        "cards":       [{"id": i, "emoji": e, "matched": False, "flippedBy": None} for i, e in enumerate(cards)],
        "currentTurn": list(room.players.keys())[0],
        "flipped":     [],
        "scores":      {pid: 0 for pid in room.players},
        "combo":       {pid: 0 for pid in room.players},
    }
    room.phase = "playing"


# ── Trivia game helpers ───────────────────────────────────────────────────────

TRIVIA_QUESTIONS = [
    {"q": "What is the capital of France?", "opts": ["Berlin","Madrid","Paris","Rome"], "ans": 2},
    {"q": "Which planet is closest to the Sun?", "opts": ["Venus","Mercury","Earth","Mars"], "ans": 1},
    {"q": "What is 12 × 12?", "opts": ["132","144","124","148"], "ans": 1},
    {"q": "Who painted the Mona Lisa?", "opts": ["Van Gogh","Picasso","Da Vinci","Rembrandt"], "ans": 2},
    {"q": "What is the largest ocean?", "opts": ["Atlantic","Indian","Arctic","Pacific"], "ans": 3},
    {"q": "How many sides does a hexagon have?", "opts": ["5","6","7","8"], "ans": 1},
    {"q": "What year did WW2 end?", "opts": ["1943","1944","1945","1946"], "ans": 2},
    {"q": "What is the chemical symbol for gold?", "opts": ["Go","Gd","Au","Ag"], "ans": 2},
    {"q": "Which country invented pizza?", "opts": ["Greece","Italy","Spain","France"], "ans": 1},
    {"q": "What is the speed of light (approx)?", "opts": ["200,000 km/s","300,000 km/s","400,000 km/s","150,000 km/s"], "ans": 1},
    {"q": "What is the largest continent?", "opts": ["Africa","Americas","Asia","Europe"], "ans": 2},
    {"q": "How many bones does an adult human have?", "opts": ["196","206","216","186"], "ans": 1},
]


def init_trivia_game(room: Room):
    questions = random.sample(TRIVIA_QUESTIONS, min(10, len(TRIVIA_QUESTIONS)))
    room.data = {
        "questions":    questions,
        "currentQ":     0,
        "scores":       {pid: 0 for pid in room.players},
        "answers":      {},       # {pid: answerIdx} for current question
        "phase":        "question",
        "questionStart": time.time(),
        "timePerQ":     20,
    }
    room.round = 1
    room.phase = "question"


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
    room_id   = make_room_id()
    player_id = "p_" + "".join(random.choices(string.ascii_lowercase, k=8))
    room      = Room(room_id, body.gameType, player_id)
    player    = Player(player_id, body.playerName, body.avatar)
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


@game_router.get("/avatars/list")
async def list_avatars():
    return {"avatars": AVATARS}


@game_router.get("/{room_id}")
async def get_room(room_id: str, playerId: str = ""):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "Room not found")
    return room.snapshot(playerId)


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

    await room.broadcast({"type": "player_joined", "player": {
        "id": player.id, "name": player.name, "avatar": player.avatar,
        "score": player.score, "ready": player.ready, "isHost": player.id == room.host_id,
        "eliminated": player.eliminated,
    }}, exclude=player_id)

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

            # ── start game ────────────────────────────────────────────────────
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
                elif room.game_type == "monopoly":
                    init_monopoly_game(room)
                elif room.game_type == "trivia":
                    init_trivia_game(room)
                for pid, p in room.players.items():
                    if p.ws:
                        await p.ws.send_text(json.dumps({"type": "game_started", **room.snapshot(pid)}))

            # ══════════════════════════════════════════════════════════════════
            # ── MONOPOLY ACTIONS ──────────────────────────────────────────────
            # ══════════════════════════════════════════════════════════════════

            elif mtype == "mono_roll":
                if room.game_type != "monopoly":
                    continue
                gs = room.data
                pidx = gs["currentPlayerIdx"]
                if gs["players"][pidx]["id"] != player_id:
                    continue
                if gs.get("phase") != "roll":
                    continue
                room.data = monopoly_apply_roll(gs, player_id)
                await room.broadcast({"type": "mono_state", "gameData": room.data, **room.snapshot()})

            elif mtype == "mono_buy":
                if room.game_type != "monopoly":
                    continue
                gs = room.data
                pidx = gs["currentPlayerIdx"]
                player_state = gs["players"][pidx]
                if player_state["id"] != player_id or gs["phase"] != "buy":
                    continue
                tile = BOARD_TILES[player_state["position"]]
                buy  = msg.get("buy", True)
                if buy and player_state["cash"] >= tile["price"]:
                    player_state["cash"] -= tile["price"]
                    player_state["properties"].append(tile["id"])
                    gs["properties"][str(tile["id"])]["ownerId"] = player_id
                    gs["log"].append(f"🏠 {player_state['name']} bought {tile['name']} for ${tile['price']}")
                    doubles = gs["dice"][0] == gs["dice"][1]
                    gs = monopoly_next_turn(gs) if not doubles else {**gs, "phase": "roll"}
                else:
                    gs["log"].append(f"⏭ {player_state['name']} declined {tile['name']} — starting auction")
                    gs = monopoly_start_auction(gs, tile["id"], player_id)
                for p in gs["players"]:
                    p["netWorth"] = monopoly_calc_net_worth(p, gs["properties"])
                room.data = gs
                await room.broadcast({"type": "mono_state", "gameData": room.data, **room.snapshot()})

            elif mtype == "mono_auction_bid":
                if room.game_type != "monopoly":
                    continue
                gs = room.data
                if gs.get("phase") != "auction" or not gs.get("auction"):
                    continue
                amount = int(msg.get("amount", 0))
                auction = gs["auction"]
                if amount > auction.get("highBid", 0) and amount <= (next((p for p in gs["players"] if p["id"] == player_id), {}) or {}).get("cash", 0):
                    auction["bids"][player_id] = amount
                    auction["highBid"] = amount
                    auction["highBidder"] = player_id
                    bidder_name = next((p["name"] for p in gs["players"] if p["id"] == player_id), "?")
                    gs["log"].append(f"🔨 {bidder_name} bids ${amount} on {BOARD_TILES[auction['tileId']]['name']}")
                    await room.broadcast({"type": "mono_auction_update", "auction": auction, "log": gs["log"][-5:]})

            elif mtype == "mono_auction_end":
                # host or timer triggers this
                if room.game_type != "monopoly":
                    continue
                if player_id != room.host_id:
                    continue
                gs = room.data
                gs = monopoly_resolve_auction(gs)
                for p in gs["players"]:
                    p["netWorth"] = monopoly_calc_net_worth(p, gs["properties"])
                room.data = gs
                await room.broadcast({"type": "mono_state", "gameData": room.data, **room.snapshot()})

            elif mtype == "mono_build":
                if room.game_type != "monopoly":
                    continue
                gs = room.data
                pidx = gs["currentPlayerIdx"]
                if gs["players"][pidx]["id"] != player_id:
                    continue
                tile_id = msg.get("tileId")
                action  = msg.get("action")  # house | hotel | sell | mortgage | unmortgage
                tile    = BOARD_TILES[tile_id]
                prop    = gs["properties"].get(str(tile_id))
                player_state = gs["players"][pidx]

                if not prop or prop["ownerId"] != player_id:
                    continue

                house_cost = tile.get("houseCost", 0)

                if action == "house" and not prop["hotel"] and prop["houses"] < 4:
                    if player_state["cash"] < house_cost:
                        continue
                    if not monopoly_check_color_monopoly(gs, player_id, tile.get("group", "")):
                        continue
                    player_state["cash"] -= house_cost
                    prop["houses"] += 1
                    gs["log"].append(f"🏗 {player_state['name']} built a house on {tile['name']} (-${house_cost})")

                elif action == "hotel" and prop["houses"] == 4 and not prop["hotel"]:
                    if player_state["cash"] < house_cost:
                        continue
                    player_state["cash"] -= house_cost
                    prop["houses"] = 0
                    prop["hotel"] = True
                    gs["log"].append(f"🏨 {player_state['name']} built a hotel on {tile['name']} (-${house_cost})")

                elif action == "sell":
                    refund = house_cost // 2
                    if prop["hotel"]:
                        prop["hotel"] = False
                        prop["houses"] = 4
                        player_state["cash"] += refund
                    elif prop["houses"] > 0:
                        prop["houses"] -= 1
                        player_state["cash"] += refund
                    gs["log"].append(f"🔨 {player_state['name']} sold a building on {tile['name']} (+${refund})")

                elif action == "mortgage" and not prop["mortgaged"] and prop["houses"] == 0 and not prop["hotel"]:
                    val = tile.get("price", 0) // 2
                    prop["mortgaged"] = True
                    player_state["cash"] += val
                    gs["log"].append(f"📋 {player_state['name']} mortgaged {tile['name']} (+${val})")

                elif action == "unmortgage" and prop["mortgaged"]:
                    cost = int(tile.get("price", 0) / 2 * 1.1)
                    if player_state["cash"] < cost:
                        continue
                    player_state["cash"] -= cost
                    prop["mortgaged"] = False
                    gs["log"].append(f"✅ {player_state['name']} lifted mortgage on {tile['name']} (-${cost})")

                for p in gs["players"]:
                    p["netWorth"] = monopoly_calc_net_worth(p, gs["properties"])
                room.data = gs
                await room.broadcast({"type": "mono_state", "gameData": room.data, **room.snapshot()})

            elif mtype == "mono_jail_bail":
                if room.game_type != "monopoly":
                    continue
                gs = room.data
                pidx = gs["currentPlayerIdx"]
                player_state = gs["players"][pidx]
                if player_state["id"] != player_id or not player_state.get("inJail"):
                    continue
                if player_state["cash"] < JAIL_BAIL:
                    continue
                player_state["cash"] -= JAIL_BAIL
                player_state["inJail"] = False
                player_state["jailTurns"] = 0
                gs["log"].append(f"🔓 {player_state['name']} paid ${JAIL_BAIL} bail.")
                room.data = gs
                await room.broadcast({"type": "mono_state", "gameData": room.data, **room.snapshot()})

            elif mtype == "mono_jail_card":
                if room.game_type != "monopoly":
                    continue
                gs = room.data
                pidx = gs["currentPlayerIdx"]
                player_state = gs["players"][pidx]
                if player_state["id"] != player_id or not player_state.get("inJail"):
                    continue
                if not player_state.get("hasJailCard"):
                    continue
                player_state["hasJailCard"] = False
                player_state["inJail"] = False
                player_state["jailTurns"] = 0
                gs["log"].append(f"🃏 {player_state['name']} used Get Out of Jail Free card!")
                room.data = gs
                await room.broadcast({"type": "mono_state", "gameData": room.data, **room.snapshot()})

            elif mtype == "mono_trade_offer":
                if room.game_type != "monopoly":
                    continue
                gs = room.data
                trade = {
                    "fromId":      player_id,
                    "toId":        msg.get("toId"),
                    "offerCash":   msg.get("offerCash", 0),
                    "offerProps":  msg.get("offerProps", []),
                    "wantCash":    msg.get("wantCash", 0),
                    "wantProps":   msg.get("wantProps", []),
                    "status":      "pending",
                }
                from_name = next((p["name"] for p in gs["players"] if p["id"] == player_id), "?")
                to_name   = next((p["name"] for p in gs["players"] if p["id"] == trade["toId"]), "?")
                gs["tradeOffer"] = trade
                gs["log"].append(f"🤝 {from_name} offered a trade to {to_name}")
                room.data = gs
                await room.send_to(trade["toId"], {"type": "mono_trade_incoming", "trade": trade})
                await ws.send_text(json.dumps({"type": "mono_trade_sent"}))

            elif mtype == "mono_trade_respond":
                if room.game_type != "monopoly":
                    continue
                gs = room.data
                trade = gs.get("tradeOffer")
                if not trade or trade["toId"] != player_id:
                    continue
                accept = msg.get("accept", False)
                if accept:
                    from_player = next((p for p in gs["players"] if p["id"] == trade["fromId"]), None)
                    to_player   = next((p for p in gs["players"] if p["id"] == trade["toId"]), None)
                    if from_player and to_player:
                        # execute trade
                        from_player["cash"] -= trade["offerCash"]
                        from_player["cash"] += trade["wantCash"]
                        to_player["cash"]   += trade["offerCash"]
                        to_player["cash"]   -= trade["wantCash"]
                        for pid in trade["offerProps"]:
                            if pid in from_player["properties"]:
                                from_player["properties"].remove(pid)
                                to_player["properties"].append(pid)
                                gs["properties"][str(pid)]["ownerId"] = to_player["id"]
                        for pid in trade["wantProps"]:
                            if pid in to_player["properties"]:
                                to_player["properties"].remove(pid)
                                from_player["properties"].append(pid)
                                gs["properties"][str(pid)]["ownerId"] = from_player["id"]
                        gs["log"].append(f"🤝 Trade accepted between {from_player['name']} and {to_player['name']}!")
                else:
                    trade_from = next((p["name"] for p in gs["players"] if p["id"] == trade["fromId"]), "?")
                    gs["log"].append(f"❌ Trade declined by {player.name}")
                gs["tradeOffer"] = None
                for p in gs["players"]:
                    p["netWorth"] = monopoly_calc_net_worth(p, gs["properties"])
                room.data = gs
                await room.broadcast({"type": "mono_state", "gameData": room.data, **room.snapshot()})

            elif mtype == "mono_dismiss_card":
                if room.game_type != "monopoly":
                    continue
                room.data["lastCard"] = None
                await room.broadcast({"type": "mono_card_dismissed"})

            # ══════════════════════════════════════════════════════════════════
            # ── BUSINESS ACTIONS ──────────────────────────────────────────────
            # ══════════════════════════════════════════════════════════════════

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

            # ══════════════════════════════════════════════════════════════════
            # ── SPY ACTIONS ───────────────────────────────────────────────────
            # ══════════════════════════════════════════════════════════════════

            elif mtype == "spy_vote":
                if room.game_type != "spy" or room.phase != "day":
                    continue
                target = msg.get("target")
                if target in room.players:
                    room.data["votes"][player_id] = target
                alive_ids = [p.id for p in room.players.values() if not p.eliminated]
                if len(room.data["votes"]) >= len(alive_ids):
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

            # ══════════════════════════════════════════════════════════════════
            # ── MEMORY ACTIONS ────────────────────────────────────────────────
            # ══════════════════════════════════════════════════════════════════

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
                        cards[a]["matched"] = cards[b]["matched"] = True
                        combo = room.data["combo"][player_id] + 1
                        room.data["combo"][player_id] = combo
                        pts = 1 + (2 if combo > 1 else 0)
                        room.data["scores"][player_id] = room.data["scores"].get(player_id, 0) + pts
                        room.players[player_id].score = room.data["scores"][player_id]
                        room.data["flipped"] = []
                        await room.broadcast({"type": "match_found", "cardA": a, "cardB": b, "playerId": player_id, "points": pts, "scores": room.data["scores"]})
                        if all(c["matched"] for c in cards):
                            room.state = "finished"
                            await room.broadcast({"type": "game_over", **room.snapshot()})
                    else:
                        room.data["combo"][player_id] = 0
                        room.data["flipped"] = []
                        alive = [pid for pid in room.players if not room.players[pid].eliminated]
                        cur_idx = alive.index(player_id) if player_id in alive else 0
                        next_pid = alive[(cur_idx + 1) % len(alive)]
                        room.data["currentTurn"] = next_pid
                        await room.broadcast({"type": "no_match", "cardA": a, "cardB": b, "nextTurn": next_pid})

            # ── TRIVIA ACTIONS ────────────────────────────────────────────────
            elif mtype == "trivia_answer":
                if room.game_type != "trivia" or room.state != "playing":
                    continue
                gs = room.data
                if gs.get("phase") != "question":
                    continue
                ans_idx = msg.get("answerIdx")
                if player_id not in gs["answers"]:
                    gs["answers"][player_id] = ans_idx
                    elapsed = time.time() - gs["questionStart"]
                    q = gs["questions"][gs["currentQ"]]
                    if ans_idx == q["ans"]:
                        speed_bonus = max(0, int(10 - elapsed))
                        gs["scores"][player_id] = gs["scores"].get(player_id, 0) + 10 + speed_bonus
                        room.players[player_id].score = gs["scores"][player_id]
                    await ws.send_text(json.dumps({"type": "trivia_answer_ack", "correct": ans_idx == q["ans"]}))
                    # check if all answered
                    active_pids = [p.id for p in room.players.values() if not p.eliminated]
                    if len(gs["answers"]) >= len(active_pids):
                        # reveal
                        q = gs["questions"][gs["currentQ"]]
                        await room.broadcast({"type": "trivia_reveal", "correctIdx": q["ans"], "scores": gs["scores"], "answers": gs["answers"]})
                        gs["currentQ"] += 1
                        gs["answers"] = {}
                        if gs["currentQ"] >= len(gs["questions"]):
                            room.state = "finished"
                            await room.broadcast({"type": "game_over", **room.snapshot()})
                        else:
                            gs["phase"] = "question"
                            gs["questionStart"] = time.time()
                            room.round = gs["currentQ"] + 1
                            await room.broadcast({"type": "trivia_next", "questionIdx": gs["currentQ"], "question": gs["questions"][gs["currentQ"]], "scores": gs["scores"]})
                room.data = gs

            # ── mic signal ────────────────────────────────────────────────────
            elif mtype == "mic_signal":
                target_id = msg.get("to")
                if target_id in room.players:
                    await room.send_to(target_id, {"type": "mic_signal", "from": player_id, "signal": msg.get("signal")})

    except WebSocketDisconnect:
        player.ws = None
        await room.broadcast({"type": "player_left", "playerId": player_id, "name": player.name})
        active = [p for p in room.players.values() if p.ws is not None]
        if not active:
            rooms.pop(room_id, None)