from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np

Rank = str
Suit = str
CardLabel = Tuple[Suit, Rank]

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img is None: return None
    if len(img.shape)==3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    return g

def _load_template(path: Path) -> Optional[np.ndarray]:
    if not path.exists(): return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None: return None
    # Se RGBA -> usa alpha come mask; per il matching usiamo il canale visivo
    if img.shape[2] == 4:
        # stacca alpha su bianco
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _best_match(gray: np.ndarray, templ: np.ndarray, method=cv2.TM_CCOEFF_NORMED) -> float:
    if gray is None or templ is None: return -1.0
    if gray.shape[0] < templ.shape[0] or gray.shape[1] < templ.shape[1]:
        return -1.0
    res = cv2.matchTemplate(gray, templ, method)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    return float(maxVal)

@dataclass
class CardRecognizer:
    root: Path
    suit_thr: float = 0.62
    rank_thr: float = 0.55
    # regioni “tipiche” dove cercare seme/rango (percentuali sull’immagine ROI)
    roi_rank = (0.02, 0.02, 0.48, 0.48)  # top-left
    roi_suit = (0.50, 0.02, 0.48, 0.35)  # top-right (tuning se serve)

    def __post_init__(self):
        self.t_suits: Dict[Suit, np.ndarray] = {}
        self.t_ranks: Dict[Rank, np.ndarray] = {}
        suits_dir = self.root / "suits"
        ranks_dir = self.root / "ranks"
        for name in ["denari","coppe","bastoni","spade"]:
            t = _load_template(suits_dir / f"{name}.png")
            if t is not None: self.t_suits[name] = t
        for r in ["A","3","K","C","F","7","6","5","4","2"]:
            t = _load_template(ranks_dir / f"{r}.png")
            if t is not None: self.t_ranks[r] = t
        if not self.t_suits or not self.t_ranks:
            raise RuntimeError(f"Template mancanti. Metti PNG in {suits_dir} e {ranks_dir}")

    def _crop_rel(self, img: np.ndarray, rel: Tuple[float,float,float,float]) -> np.ndarray:
        H, W = img.shape[:2]
        x = max(0, int(rel[0]*W)); y = max(0, int(rel[1]*H))
        w = max(1, int(rel[2]*W)); h = max(1, int(rel[3]*H))
        x2 = min(W, x+w); y2 = min(H, y+h)
        return img[y:y2, x:x2].copy()

    def recognize(self, roi_bgr: np.ndarray) -> Optional[CardLabel]:
        """
        Ritorna (suit, rank) oppure None se non sicuro.
        """
        if roi_bgr is None: return None
        gray = _to_gray(roi_bgr)
        # equalizzazione leggera
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # focus su sotto-aree
        g_rank = self._crop_rel(gray, self.roi_rank)
        g_suit = self._crop_rel(gray, self.roi_suit)

        # match rank
        best_r, best_r_score = None, -1.0
        for r, templ in self.t_ranks.items():
            s = _best_match(g_rank, templ)
            if s > best_r_score:
                best_r, best_r_score = r, s

        # match suit
        best_s, best_s_score = None, -1.0
        for sname, templ in self.t_suits.items():
            s = _best_match(g_suit, templ)
            if s > best_s_score:
                best_s, best_s_score = sname, s

        if best_r_score >= self.rank_thr and best_s_score >= self.suit_thr:
            return (best_s, best_r)
        # fallback: prova matching su full ROI se non ha passato le soglie
        if best_r is None or best_r_score < self.rank_thr:
            for r, templ in self.t_ranks.items():
                s = _best_match(gray, templ)
                if s > best_r_score:
                    best_r, best_r_score = r, s
        if best_s is None or best_s_score < self.suit_thr:
            for sname, templ in self.t_suits.items():
                s = _best_match(gray, templ)
                if s > best_s_score:
                    best_s, best_s_score = sname, s
        if best_r_score >= self.rank_thr and best_s_score >= self.suit_thr:
            return (best_s, best_r)
        return None
