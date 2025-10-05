from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import os, json, hashlib
import cv2
import numpy as np

BRISCOLA_DEBUG = os.getenv("BRISCOLA_DEBUG", "0") == "1"
BRISCOLA_FAST   = os.getenv("BRISCOLA_FAST", "1") == "1"

def _resize(img: np.ndarray, wh=(120,180)) -> np.ndarray:
    return cv2.resize(img, wh, interpolation=cv2.INTER_AREA)

def _to_gray(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(_resize(img), cv2.COLOR_BGR2GRAY)

def _canny(gray: np.ndarray) -> np.ndarray:
    med = np.median(gray)
    lo = int(max(0, 0.66 * med)); hi = int(min(255, 1.33 * med))
    return cv2.Canny(gray, lo, hi, L2gradient=True)

def _dhash(gray: np.ndarray) -> int:
    g = cv2.resize(gray, (9,8))
    diff = g[:,1:] > g[:,:-1]
    bits = 0
    for i, v in enumerate(diff.flatten()):
        if v: bits |= (1<<i)
    return bits

def _ensure_big_enough(src: np.ndarray, templ: np.ndarray) -> np.ndarray:
    sh, sw = src.shape[:2]; th, tw = templ.shape[:2]
    pad_y = max(0, th - sh); pad_x = max(0, tw - sw)
    if pad_x or pad_y:
        src = cv2.copyMakeBorder(src, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)
    return src

# ----------------------- Library -----------------------
@dataclass
class FullCardLibrary:
    root: Path
    manifest_path: Optional[Path] = None
    def __post_init__(self):
        mp = self.manifest_path or (self.root.parent / "manifest.json")
        if not mp.exists():
            raise RuntimeError(f"Manifest non trovato: {mp}")
        meta = json.loads(mp.read_text(encoding="utf-8"))
        self.meta: Dict[str, Dict[str,str]] = meta.get("full", {})

        # Base templates (grayscale) at 120x180
        self.gray: Dict[str, np.ndarray] = {}
        for p in sorted(self.root.glob("*.png")):
            g = _to_gray(p)
            if g is None: continue
            self.gray[p.name] = g
        if not self.gray:
            raise RuntimeError(f"Nessun template in {self.root}")

        # Build pyramids (small 3 scales for speed)
        self.scales = (0.96, 1.00, 1.04) if BRISCOLA_FAST else (0.92, 0.96, 1.00, 1.04, 1.08)
        self.gray_scaled: Dict[Tuple[str,float], np.ndarray] = {}
        self.edge_scaled: Dict[Tuple[str,float], np.ndarray] = {}
        for name, g in self.gray.items():
            for s in self.scales:
                t = cv2.resize(g, (max(2,int(g.shape[1]*s)), max(2,int(g.shape[0]*s))), interpolation=cv2.INTER_AREA)
                self.gray_scaled[(name,s)] = t
                self.edge_scaled[(name,s)] = _canny(t)

        # Suit masks (soft, per-pixel variance within same suit), resized per scale
        by_suit: Dict[str, List[np.ndarray]] = {"denari":[], "coppe":[], "bastoni":[], "spade":[]}
        for name, g in self.gray.items():
            s = self.meta.get(name,{}).get("suit")
            if s: by_suit[s].append(g.astype(np.float32))
        self.mask_scaled: Dict[Tuple[str,float], Optional[np.ndarray]] = {}
        for suit, arrs in by_suit.items():
            if len(arrs) < 2:
                for s in self.scales:
                    self.mask_scaled[(suit,s)] = None
                continue
            std = np.stack(arrs,0).std(0)
            std = cv2.GaussianBlur(std, (5,5), 0)
            std = std / (std.max()+1e-6)
            h,w = std.shape
            band = np.zeros_like(std, np.float32)
            y1,y2 = int(0.30*h), int(0.70*h); band[y1:y2,:] = 1.0
            std = std*(0.4 + 0.6*band)
            for s in self.scales:
                self.mask_scaled[(suit,s)] = cv2.resize(std, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def semantics(self, name: str) -> Tuple[Optional[str], Optional[str]]:
        m = self.meta.get(name, {}); return m.get("suit"), m.get("rank")

# ----------------------- Recognizer -----------------------
class Recognizer:
    def __init__(self, lib: FullCardLibrary, thr: float=0.62, margin: float=0.055, topk:int=4):
        self.lib = lib
        self.thr = thr; self.margin = margin; self.topk = topk
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        self._cacheA: Dict[int, List[Tuple[str,float]]] = {}  # dhash -> shortlist
        self._last_dbg: Optional[dict] = None

    def _prep(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        small = _resize(bgr)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab); l = self._clahe.apply(l)
        small = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        e = _canny(g)
        return g, e, _dhash(g)

    def _score_tm(self, src_g: np.ndarray, templ_g: np.ndarray, mask: Optional[np.ndarray]) -> float:
        src_g = _ensure_big_enough(src_g, templ_g)
        sm = -1.0
        if mask is not None:
            mh = cv2.resize(mask, (templ_g.shape[1], templ_g.shape[0]), interpolation=cv2.INTER_NEAREST)
            if mh.dtype not in (np.uint8, np.float32):
                mh = mh.astype(np.float32)
            res_m = cv2.matchTemplate(src_g, templ_g, cv2.TM_CCORR_NORMED, mask=mh)
            _, sm, _, _ = cv2.minMaxLoc(res_m)
        res_u = cv2.matchTemplate(src_g, templ_g, cv2.TM_CCOEFF_NORMED)
        _, su, _, _ = cv2.minMaxLoc(res_u)
        return 0.6*(sm if mask is not None else su) + 0.4*su

    def _shortlist(self, src_g: np.ndarray, suit_hint: Optional[str], dh: int) -> List[str]:
        if dh in self._cacheA:  # fast reuse if ROI unchanged
            return [n for n,_ in self._cacheA[dh][:self.topk]]
        # use only mask "edge-band" (no central text) + CCOEFF to shortlist
        h,w = src_g.shape
        maskA = np.full((h,w), 255, np.uint8); y1,y2 = int(0.35*h), int(0.65*h); maskA[y1:y2,:] = 0
        cand = [(n,g) for (n,g) in self.lib.gray.items() if (not suit_hint or self.lib.semantics(n)[0]==suit_hint)]
        if len(cand) < 6: cand = list(self.lib.gray.items())
        scores: List[Tuple[str,float]] = []
        for name, templ in cand:
            sc = self._score_tm(src_g, templ, maskA)
            scores.append((name, float(sc)))
        scores.sort(key=lambda x:x[1], reverse=True)
        self._cacheA[dh] = scores
        return [n for n,_ in scores[:self.topk]]

    def best_match(self, roi_bgr: Optional[np.ndarray], suit_hint: Optional[str]=None) -> Tuple[Optional[str], float, str]:
        if roi_bgr is None: return None, -1.0, "noimg"
        g, e, dh = self._prep(roi_bgr)
        names = self._shortlist(g, suit_hint, dh)
        best_name, best_sc, second_sc = None, -1.0, -1.0
        best_scale, best_tm, best_edge = 1.0, 0.0, 0.0

        for name in names:
            s, _ = self.lib.semantics(name)
            for sc in self.lib.scales:
                tg = self.lib.gray_scaled[(name,sc)]
                tm = self._score_tm(g, tg, self.lib.mask_scaled.get((s,sc)))
                te = self.lib.edge_scaled[(name,sc)]
                gg = _ensure_big_enough(g, tg); ee = _ensure_big_enough(e, te)
                _, se, _, _ = cv2.minMaxLoc(cv2.matchTemplate(ee, te, cv2.TM_CCOEFF_NORMED))
                score = 0.65*tm + 0.35*float(se)
                if score > best_sc:
                    second_sc = best_sc; best_sc = score; best_name = name
                    best_scale, best_tm, best_edge = sc, tm, float(se)
                elif score > second_sc:
                    second_sc = score

        suit, rank = self.lib.semantics(best_name) if best_name else (None,None)
        self._last_dbg = {
            "suit": suit, "rank": rank, "scale": best_scale, "tm": best_tm, "edge": best_edge
        }
        tag = f"{best_name or 'None'} | suit={suit or '?'} rank={rank or '?'} | S={best_sc:.3f} (2nd={second_sc:.3f}) scale={best_scale:.2f}"
        if best_sc >= self.thr and (best_sc - second_sc) >= self.margin:
            return best_name, best_sc, tag
        return None, best_sc, tag

    def get_last_debug(self) -> Optional[dict]:
        return self._last_dbg
