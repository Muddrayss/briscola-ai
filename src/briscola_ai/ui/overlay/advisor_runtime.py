from __future__ import annotations
import sys, hashlib, traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, Future

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from .advisor_overlay import AdvisorOverlay
from .correction_panel import CorrectionPanel
from ...vision.sisal_reader import SisalReader
from ...vision.recognizer_full import FullCardLibrary, Recognizer
from ...core.cards import Card, Suit, POINTS
from ...core.env import BriscolaEnv, Observation
from ...agents.mcts import ISMCTSAgent

SUIT_FROM_NAME = {"denari": Suit.DENARI, "coppe": Suit.COPPE, "bastoni": Suit.BASTONI, "spade": Suit.SPADE}
RANKS = ["A","3","K","C","F","7","6","5","4","2"]

# ----------------- helpers -----------------
def suit_from_string(s: str) -> Optional[Suit]:
    return SUIT_FROM_NAME.get((s or "").lower())

def apply_rank_from_name(nm: str, default: str="7") -> str:
    low = (nm or "").lower()
    for r in RANKS:
        if f"_{r}".lower() in low or low.endswith(r.lower()):
            return r
    return default

def build_live_env_and_obs(my_cards: List[Card], briscola_suit: Suit,
                           lead_card: Optional[Card],
                           trump_card_known: Optional[Card],
                           played: List[Card]) -> Tuple[BriscolaEnv, Observation]:
    env = BriscolaEnv(seed=123)
    env.points = [0,0]
    env.briscola = briscola_suit
    env.trump_card = trump_card_known
    env.hands[0] = list(my_cards)
    env.hands[1] = []
    if lead_card is None:
        env.leader = 0; env.turn_player = 0; env.lead_card = None; env.follow_card = None
    else:
        env.leader = 1; env.turn_player = 0; env.lead_card = lead_card; env.follow_card = None
    env.played = list(played)
    env.deck = []
    trick_pts = POINTS[lead_card.rank] if lead_card and lead_card.rank in POINTS else 0
    obs = Observation(
        player=0, hand=list(my_cards), briscola=briscola_suit, lead_card=lead_card,
        deck_count=0, trick_points_so_far=trick_pts, played=list(env.played),
        opponent_count=(2 if lead_card else 3)
    )
    return env, obs

def phash(arr) -> str:
    import cv2, hashlib
    if arr is None:
        return "NONE"
    small = cv2.resize(arr, (64,64), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()

# ----------------- MAIN -----------------
def main():
    if len(sys.argv) < 3:
        print("Uso: python -m briscola_ai.ui.overlay.advisor_runtime <calibration.json> <templates_full_dir> [iters]")
        sys.exit(1)
    cal_path = sys.argv[1]
    templ_dir = Path(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) >= 4 else 500

    app = QApplication(sys.argv)

    overlay = AdvisorOverlay(cal_path)
    overlay.set_hud_anchor("tr")
    overlay.set_hud_scale(1.2)
    overlay.show()

    try:
        lib = FullCardLibrary(templ_dir)          # legge manifest + templates
    except Exception as e:
        # Mostra subito errore blocking (manifest/templates)
        overlay.set_debug_info({"ERROR":"Template/manifest", "detail":str(e)})
        overlay.show()
        sys.exit(app.exec())

    recog = Recognizer(lib, thr=0.56)
    panel = CorrectionPanel([name for name,_ in lib.templates]); panel.show()
    reader = SisalReader(cal_path)
    agent = ISMCTSAgent(iterations=iters, c=1.0, discount=0.7, seed=7)

    # ---- stato ----
    scanning_active = True
    lock_bottom = False
    locked_names: List[Optional[str]] = [None, None, None]

    # overrides dal pannello
    bottom_over: Dict[int,str] = {0:"",1:"",2:""}
    trump_suit_override: str = ""
    trump_rank_override: str = ""
    lead_enabled: bool = True
    lead_suit_override: str = ""
    lead_rank_override: str = ""
    last_lead_s: str = ""; last_lead_r: str = ""
    last_follow_s: str = ""; last_follow_r: str = ""

    # anti-flicker
    current_highlight: Optional[str] = None
    last_raw: Optional[str] = None
    stable_count: int = 0
    MIN_STABLE = 2

    # executor + cache roi
    executor = ThreadPoolExecutor(max_workers=1)
    in_flight: Optional[Future] = None
    last_result: Optional[Dict] = None
    last_roi_hash: Dict[str,str] = {}

    # wiring pannello
    panel.changed_bottom.connect(lambda slot, name: bottom_over.__setitem__(slot, name if name!="(auto)" else ""))

    def set_trump_card(suit: str, rank: str):
        nonlocal trump_suit_override, trump_rank_override
        trump_suit_override = suit; trump_rank_override = rank
    panel.changed_trump_card.connect(set_trump_card)

    def set_lead_card(enabled: bool, suit: str, rank: str):
        nonlocal lead_enabled, lead_suit_override, lead_rank_override
        lead_enabled = enabled; lead_suit_override = suit; lead_rank_override = rank
    panel.changed_lead_card.connect(set_lead_card)

    def set_last_lead(s: str, r: str):
        nonlocal last_lead_s, last_lead_r
        last_lead_s, last_lead_r = s, r
    panel.changed_last_lead.connect(set_last_lead)

    def set_last_follow(s: str, r: str):
        nonlocal last_follow_s, last_follow_r
        last_follow_s, last_follow_r = s, r
    panel.changed_last_follow.connect(set_last_follow)

    def do_start(): 
        nonlocal scanning_active
        scanning_active = True
    panel.start_clicked.connect(do_start)

    def do_pause():
        nonlocal scanning_active, current_highlight, last_raw, stable_count
        scanning_active = False
        current_highlight = None; last_raw = None; stable_count = 0
        overlay.set_suggestion(None)
    panel.pause_clicked.connect(do_pause)

    def do_reset():
        nonlocal scanning_active, lock_bottom, locked_names
        nonlocal current_highlight, last_raw, stable_count
        scanning_active = False; lock_bottom = False; locked_names = [None,None,None]
        for k in bottom_over: bottom_over[k] = ""
        set_trump_card("", ""); set_lead_card(True, "", ""); set_last_lead("", ""); set_last_follow("", "")
        current_highlight = None; last_raw = None; stable_count = 0
        overlay.set_suggestion(None); overlay.set_debug_info({}); overlay.set_thumbnails({})
        last_roi_hash.clear()
    panel.reset_clicked.connect(do_reset)

    def do_new_hand():
        nonlocal lock_bottom, locked_names
        set_lead_card(False, lead_suit_override, lead_rank_override)
        lock_bottom = False; locked_names = [None,None,None]
    panel.new_hand_clicked.connect(do_new_hand)

    def do_new_trick():
        nonlocal lock_bottom, locked_names
        set_lead_card(True, lead_suit_override, lead_rank_override)
        lock_bottom = False; locked_names = [None,None,None]
    panel.new_trick_clicked.connect(do_new_trick)

    panel.lock_bottom_toggled.connect(lambda on: (
        locked_names.__setitem__(0,None), locked_names.__setitem__(1,None), locked_names.__setitem__(2,None)
    ))

    # -------------- worker (robusto) --------------
    def worker_job(snap: Dict[str, Optional[any]]):
        debug = {"status":"RUN"}
        thumbs = {}
        try:
            # bottom
            names: List[Optional[str]] = [None,None,None]
            for i,k in enumerate(["bottom_slot_1","bottom_slot_2","bottom_slot_3"]):
                if lock_bottom and locked_names[i] is not None:
                    names[i] = locked_names[i]; debug[k] = f"locked:{locked_names[i]}"; continue
                if bottom_over[i]:
                    names[i] = bottom_over[i]; debug[k] = f"override:{bottom_over[i]}"; continue
                img = snap.get(k)
                if img is None:
                    debug[k] = "MISSING ROI"
                    continue
                h = phash(img)
                if last_roi_hash.get(k) == h and last_result and last_result.get("names"):
                    names[i] = last_result["names"][i]; debug[k] = "(cached)"
                else:
                    nm, sc, tag = recog.best_match(img)
                    names[i] = nm; debug[k] = tag; last_roi_hash[k] = h
                if names[i]:
                    thumbs[k] = str(templ_dir / names[i])

            if lock_bottom and all(n is not None for n in names):
                for i in range(3):
                    if locked_names[i] is None: locked_names[i] = names[i]

            if any(n is None for n in names):
                debug["BLOCK"] = "missing bottom card(s)"
                return {"ok": False, "debug": debug, "thumbs": thumbs, "names": names}

            # trump
            tnm = None; img_t = snap.get("trump")
            if trump_suit_override and trump_suit_override!="(auto)":
                debug["trump"] = f"OVERRIDE SUIT {trump_suit_override}"
            else:
                if img_t is not None:
                    h = phash(img_t)
                    if last_roi_hash.get("trump") == h and last_result:
                        tnm = last_result.get("trump_nm"); debug["trump"] = "(cached)"
                    else:
                        tnm, sc, tag = recog.best_match(img_t); debug["trump"] = tag; last_roi_hash["trump"] = h
                    if tnm: thumbs["trump"] = str(templ_dir / tnm)

            br_suit = suit_from_string(trump_suit_override) or (suit_from_string(tnm) if tnm else None)
            if br_suit is None:
                debug["BLOCK"] = "missing briscola suit (set in panel or ensure ROI 'trump')"
                return {"ok": False, "debug": debug, "thumbs": thumbs, "names": names, "trump_nm": tnm}

            # trump rank (opzionale)
            trump_rank = None
            if trump_rank_override and trump_rank_override!="(auto)":
                trump_rank = trump_rank_override
            elif tnm:
                s_txt, r_txt = lib.semantics(tnm)
                trump_rank = r_txt or apply_rank_from_name(tnm, "")
                if trump_rank == "": trump_rank = None
            trump_card_known = Card(br_suit, trump_rank) if trump_rank else None

            # lead
            lead_card = None
            if lead_enabled:
                if lead_suit_override and lead_suit_override!="(auto)":
                    ls = suit_from_string(lead_suit_override)
                    lr = lead_rank_override if (lead_rank_override and lead_rank_override!="(auto)") else "7"
                    if ls: lead_card = Card(ls, lr)
                    debug["lead"] = f"OVERRIDE {lead_suit_override}/{lr}"
                else:
                    limg = snap.get("lead")
                    if limg is not None:
                        h = phash(limg)
                        if last_roi_hash.get("lead") == h and last_result:
                            lnm = last_result.get("lead_nm"); debug["lead"] = "(cached)"
                        else:
                            lnm, sc, tag = recog.best_match(limg); debug["lead"] = tag; last_roi_hash["lead"] = h
                        if lnm:
                            thumbs["lead"] = str(templ_dir / lnm)
                            s_txt, r_txt = lib.semantics(lnm)
                            ls = suit_from_string(s_txt) or suit_from_string(lnm) or br_suit
                            lr = r_txt or apply_rank_from_name(lnm, "7")
                            lead_card = Card(ls, lr)
                    else:
                        debug["lead"] = "(none)"

            # last trick (se servono: opzionale)
            played: List[Card] = []
            # (si possono popolare da pannello se vuoi; al momento lasciamo vuoto)

            # bottom -> Card dal manifest
            def card_from_name(nm: str, fb: Suit) -> Card:
                s_txt, r_txt = lib.semantics(nm)
                s = suit_from_string(s_txt) or suit_from_string(nm) or fb
                r = r_txt or apply_rank_from_name(nm, "7")
                return Card(s, r)

            my_cards = [card_from_name(n, br_suit) for n in names]  # type: ignore

            env, obs = build_live_env_and_obs(my_cards, br_suit, lead_card, trump_card_known, played)
            idx = int(ISMCTSAgent(iterations=agent.iterations, c=agent.c, discount=agent.discount, seed=agent.seed).act(env, obs))
            if idx not in (0,1,2):
                debug["ERROR"] = f"bad index {idx}"
                return {"ok": False, "debug": debug, "thumbs": thumbs, "names": names, "trump_nm": tnm}

            return {"ok": True, "debug": debug, "thumbs": thumbs, "names": names, "trump_nm": tnm, "lead_nm": None, "suggest": f"bottom_slot_{idx+1}"}

        except Exception as e:
            tb = traceback.format_exc(limit=2)
            debug["ERROR"] = f"{e.__class__.__name__}: {str(e)}"
            debug["TRACE"] = tb.strip().replace("\n"," | ")
            return {"ok": False, "debug": debug, "thumbs": thumbs}

    # -------------- UI loop --------------
    def tick():
        nonlocal scanning_active, current_highlight, last_raw, stable_count, in_flight, last_result

        if not scanning_active:
            overlay.set_debug_info({"status":"PAUSED"})
            return

        if in_flight and not in_flight.done():
            if last_result:
                overlay.set_debug_info({**last_result.get("debug", {}), "status":"RUN"})
            return

        snap = reader.read_snapshot()
        fut = executor.submit(worker_job, snap)
        in_flight = fut

        def done(fu: Future):
            nonlocal current_highlight, last_raw, stable_count, last_result
            try:
                res = fu.result()
            except Exception as e:
                overlay.set_debug_info({"ERROR":"worker crash", "detail":str(e)})
                return
            last_result = res
            dbg = {**res.get("debug", {}), "status":"RUN"}
            overlay.set_debug_info(dbg)
            overlay.set_thumbnails(res.get("thumbs", {}))
            print("[ADVISOR]", dbg)  # console mirror

            if not res.get("ok"):
                overlay.set_suggestion(None)
                return

            raw_key = res.get("suggest")
            if not raw_key:
                overlay.set_suggestion(None)
                return

            if raw_key == last_raw:
                stable_count += 1
            else:
                last_raw = raw_key; stable_count = 1
            if current_highlight != raw_key and stable_count >= MIN_STABLE:
                current_highlight = raw_key
                overlay.set_suggestion(current_highlight)

        fut.add_done_callback(done)

    QTimer.singleShot(0, tick)
    timer = QTimer(); timer.timeout.connect(tick); timer.start(140)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
