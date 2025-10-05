from __future__ import annotations
import os, sys, time, logging, logging.handlers, threading, hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QtMsgType, qInstallMessageHandler

from .advisor_overlay import AdvisorOverlay
from .correction_panel import CorrectionPanel
from .debug_panel import DebugPanel
from ...vision.sisal_reader import SisalReader
from ...vision.recognizer_full import FullCardLibrary, Recognizer
from ...core.cards import Card, Suit
from ...core.env import BriscolaEnv, Observation

# -------------------- Logging --------------------
def setup_logging() -> logging.Logger:
    log_dir = Path.cwd() / "_logs"
    log_dir.mkdir(exist_ok=True)
    lvl = logging.DEBUG if os.getenv("BRISCOLA_DEBUG","0") == "1" else logging.INFO
    logger = logging.getLogger("advisor")
    logger.setLevel(lvl); logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr); sh.setLevel(lvl); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(log_dir/"overlay.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(lvl); fh.setFormatter(fmt); logger.addHandler(fh)

    def qt_handler(mode, context, message):
        msg = f"{context.file}:{context.line} [{context.function}] {message}"
        if mode == QtMsgType.QtDebugMsg:   logger.debug("QT: " + msg)
        elif mode == QtMsgType.QtInfoMsg:  logger.info("QT: " + msg)
        elif mode == QtMsgType.QtWarningMsg: logger.warning("QT: " + msg)
        elif mode == QtMsgType.QtCriticalMsg: logger.error("QT: " + msg)
        elif mode == QtMsgType.QtFatalMsg: logger.critical("QT: " + msg)
    qInstallMessageHandler(qt_handler)

    try:
        m = {"SILENT":0, "FATAL":1, "ERROR":2, "WARNING":3, "INFO":4, "DEBUG":5, "VERBOSE":6}
        lvl_str = os.getenv("OPENCV_LOG_LEVEL", "ERROR").upper()
        cv2.utils.logging.setLogLevel(m.get(lvl_str, 2))
    except Exception:
        pass
    return logger

def md5_of(gray120x180: np.ndarray) -> str:
    return hashlib.md5(gray120x180.tobytes()).hexdigest()

def roi_is_empty(img_bgr: np.ndarray) -> bool:
    """Heuristic emptiness test: low edge density + low Laplacian variance."""
    if img_bgr is None: return True
    g = cv2.cvtColor(cv2.resize(img_bgr, (160,120)), cv2.COLOR_BGR2GRAY)
    # edge density
    edges = cv2.Canny(g, 50, 150)  # docs recommend Gaussian smoothing + gradient; we use defaults for speed
    ed = float(np.count_nonzero(edges)) / float(edges.size)
    # focus/texture via Laplacian variance
    lapv = cv2.Laplacian(g, cv2.CV_64F).var()
    return (ed < 0.012) and (lapv < 18.0)

# ---------- helpers ----------
SUIT_FROM_NAME = {"denari": Suit.DENARI, "coppe": Suit.COPPE, "bastoni": Suit.BASTONI, "spade": Suit.SPADE}
BOTTOM_KEYS = ["bottom_slot_1","bottom_slot_2","bottom_slot_3"]
AUX_KEYS = ["lead","follow","trump"]

# ---------- main ----------
def main():
    try:
        cv2.setNumThreads(int(os.getenv("BRISCOLA_CV_THREADS","1")))
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    if len(sys.argv) < 3:
        print("Uso: python -m briscola_ai.ui.overlay.advisor_runtime <calibration.json> <templates_full_dir> [iters]")
        sys.exit(1)
    cal_path = sys.argv[1]; templ_dir = Path(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) >= 4 else 500

    log = setup_logging()
    log.info("=== Briscola Advisor start ===")
    log.info(f"Calibration: {cal_path}")
    log.info(f"Templates:   {templ_dir}")

    app = QApplication(sys.argv)
    overlay = AdvisorOverlay(cal_path); overlay.set_hud_anchor("tr"); overlay.set_hud_scale(1.06); overlay.show()

    dbg_panel = DebugPanel(out_dir=str(Path.cwd()/"_debug"))
    lib = FullCardLibrary(templ_dir)
    recog = Recognizer(lib, thr=0.60, margin=0.055, topk=4)

    panel = CorrectionPanel([name for name in lib.gray.keys()]); panel.show()

    scanning_active = True
    last_md5: Dict[str,str] = {}
    last_caption: Dict[str,str] = {}
    last_name: Dict[str, Optional[str]] = {}
    last_score: Dict[str, float] = {}
    last_seen_ms: Dict[str, int] = {}
    overrides: Dict[str, str] = {}
    locked: Dict[str, bool] = {}
    thumbs_dir = Path.cwd()/"_debug"/"thumbs"; thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Hysteresis + TTLs
    THR_ON = float(os.getenv("BRISCOLA_THR_ON","0.63"))
    THR_OFF = float(os.getenv("BRISCOLA_THR_OFF","0.58"))
    TTL_MS_AUX = int(os.getenv("BRISCOLA_TTL_AUX_MS","1800"))  # hold lead/follow/trump briefly after they disappear
    STICKY_OVERRIDE = (os.getenv("BRISCOLA_OVERRIDE_STICKY","1") == "1")

    def now_ms() -> int: return int(time.time()*1000)

    def on_start(): 
        nonlocal scanning_active; scanning_active = True; overlay.set_debug_info({"status":"RUN"}); log.info("UI: start")
    def on_pause(): 
        nonlocal scanning_active; scanning_active = False; overlay.set_debug_info({"status":"PAUSE"}); log.info("UI: pause")
    def on_reset(): 
        nonlocal last_md5, last_caption, last_name, last_score, overrides, locked, last_seen_ms
        last_md5.clear(); last_caption.clear(); last_name.clear(); last_score.clear()
        overrides.clear(); locked.clear(); last_seen_ms.clear()
        overlay.set_debug_info({"status":"RESET"}); log.info("UI: reset")
    panel.start_clicked.connect(on_start); panel.pause_clicked.connect(on_pause); panel.reset_clicked.connect(on_reset)

    # --- Robust override plumbing ---
    def _apply_override(slot_key: str, card_name: str, sticky: bool = True):
        overrides[slot_key] = card_name
        locked[slot_key] = bool(sticky)
        last_name[slot_key] = card_name
        last_score[slot_key] = 0.99
        last_caption[slot_key] = f"OVERRIDE: {card_name}"
        overlay.set_debug_info({**overlay.debug_texts, slot_key: last_caption[slot_key]})
        log.info("OVERRIDE %s := %s (sticky=%s)", slot_key, card_name, sticky)

    def _clear_override(slot_key: str):
        overrides.pop(slot_key, None)
        locked.pop(slot_key, None)
        log.info("OVERRIDE cleared for %s", slot_key)

    # Try to auto-connect common signals from the panel
    connected = False
    for sig_name in ("override_selected", "card_selected", "overrideChanged", "apply_override", "overrideTrump", "overrideLead", "overrideFollow"):
        if hasattr(panel, sig_name):
            sig = getattr(panel, sig_name)
            try:
                # Accept either (slot, card) or (card,) signatures
                def _handler(*args, _sig=sig_name):
                    if not args: return
                    if len(args) == 2:
                        sk, cn = args[0], args[1]
                    else:
                        # If single arg, map to currently suggested slot or explicit 'trump/lead/follow'
                        cn = args[0]
                        sk = overlay.suggest_key or "bottom_slot_2"
                    _apply_override(sk, cn, STICKY_OVERRIDE)
                sig.connect(_handler)
                log.info("Connected override signal: %s", sig_name)
                connected = True
            except Exception:
                pass
    if not connected:
        log.warning("No known override signal found on CorrectionPanel; overrides might need manual wiring.")

    # Optional: env-specified startup overrides (e.g. trump)
    env_trump = os.getenv("BRISCOLA_LOCK_TRUMP","").strip()
    if env_trump:
        _apply_override("trump", env_trump, True)

    pool = ThreadPoolExecutor(max_workers=1); pending = {"fut": None}
    _tls = threading.local()

    def _get_reader():
        r = getattr(_tls, "reader", None)
        if r is None:
            _tls.reader = SisalReader(cal_path)
            log.debug("Initialized SisalReader in worker thread")
            r = _tls.reader
        return r

    def _grab_all_rois(r: SisalReader) -> Dict[str, "object"]:
        if hasattr(r, "capture_rois"): return r.capture_rois()
        return r.read_snapshot()

    def _recognize_slot(k: str, img, force: bool = False):
        """Update last_* for slot k using recognition with hysteresis; obey overrides/locks and presence gating."""
        t_ms = now_ms()
        # Presence gating for aux (lead/follow/trump)
        aux = (k in AUX_KEYS)
        present = (not aux) or (not roi_is_empty(img))
        was_recent = (t_ms - last_seen_ms.get(k, 0) <= TTL_MS_AUX)

        if aux:
            if present:
                last_seen_ms[k] = t_ms
            elif not was_recent:
                # Timeout -> clear unless locked/override
                if k not in overrides:
                    last_name[k] = None; last_score[k] = 0.0
                    last_caption[k] = "—"
                return  # don't recognize when empty/expired

        # Overrides win unless explicitly non-sticky and ROI changed
        if k in overrides:
            nm = overrides[k]; sc = 0.99
            last_name[k] = nm; last_score[k] = sc; last_caption[k] = f"OVERRIDE: {nm} | S={sc:.2f}"
            return

        # Normal recognition with hysteresis
        g = cv2.cvtColor(cv2.resize(img, (120,180)), cv2.COLOR_BGR2GRAY)
        m = md5_of(g)
        changed = (m != last_md5.get(k))
        if changed: last_md5[k] = m

        nm0, sc0 = last_name.get(k), last_score.get(k, 0.0)
        need_rec = force or changed or (nm0 is None) or (sc0 < THR_OFF)
        if not need_rec:
            return

        n2, s2, tag = recog.best_match(img, suit_hint=None)
        accept = False
        if n2 and s2 >= THR_ON: accept = True
        elif nm0 is None and n2: accept = True  # bootstrap from nothing
        if accept:
            last_name[k] = n2; last_score[k] = float(s2)
        # Caption always updated for UI
        nm = last_name.get(k); sc = last_score.get(k, 0.0)
        cap = f"{nm or '—'}"
        if n2 and s2: cap += f" | S={s2:.2f}"
        last_caption[k] = cap

    def worker_job(frame_idx: int):
        debug: Dict[str,str] = {}
        thumbs: Dict[str,str] = {}

        # --- Thread-safe reader ---
        r = _get_reader()
        tried_reinit = False
        while True:
            try:
                roi = _grab_all_rois(r)
                break
            except AttributeError as e:
                if ("_thread._local" in str(e)) and (not tried_reinit):
                    log.warning("Reader had thread-local handle issue, reinitializing in worker thread...")
                    _tls.reader = None
                    r = _get_reader()
                    tried_reinit = True
                    continue
                raise

        # keys from calibration (order preserved), else snapshot keys
        keys = list(overlay.rois.keys()) or list(roi.keys())

        # Recognize all present keys
        for k in keys:
            img = roi.get(k)
            if img is None:
                debug[k] = last_caption.get(k, "—")
                continue
            _recognize_slot(k, img)

            # Thumbnail
            try:
                p = (thumbs_dir/f"{k}.png").as_posix()
                small = cv2.resize(img, (min(200, img.shape[1]), min(150, img.shape[0])), interpolation=cv2.INTER_AREA)
                label = last_name.get(k) or '—'
                cv2.putText(small, label, (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.imwrite(p, small)
                thumbs[k] = p
            except Exception:
                pass

            debug[k] = last_caption.get(k, "—")

        # Choose suggestion always among bottom slots: fallback to last known or mid slot
        best_key = None; best_sc = -1.0
        for k in BOTTOM_KEYS:
            sc = last_score.get(k, 0.0); nm = last_name.get(k)
            if nm and sc > best_sc:
                best_sc = sc; best_key = k
        if best_key is None:
            # fallback to last non-None name or center slot
            for k in BOTTOM_KEYS:
                if last_name.get(k): best_key = k; break
        if best_key is None:
            best_key = "bottom_slot_2"

        debug.update({"ROI_KEYS": ",".join(keys), "suggest": best_key})
        return {"ok": True, "debug": debug, "thumbs": thumbs, "suggest": best_key}

    pool = ThreadPoolExecutor(max_workers=1); pending = {"fut": None}

    def tick():
        if not scanning_active: return
        if pending["fut"] and not pending["fut"].done(): return
        try:
            fut = pool.submit(worker_job, int(time.time()*1000) & 0xffff)
            pending["fut"] = fut
            def done(f):
                try:
                    out = f.result()
                except Exception as e:
                    overlay.set_debug_info({"ERROR": "worker", "detail": str(e)})
                    logging.getLogger("advisor").exception("Worker exception")
                    return
                overlay.set_debug_info(out["debug"]); overlay.set_thumbnails(out["thumbs"])
                overlay.set_suggestion(out["suggest"])
            fut.add_done_callback(done)
        except Exception as e:
            overlay.set_debug_info({"ERROR":"submit", "detail":str(e)})
            logging.getLogger("advisor").exception("Submit exception")

    QTimer.singleShot(0, tick); timer = QTimer(); timer.timeout.connect(tick); timer.start(110)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
