from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import cv2, json, hashlib
import numpy as np

def _resize(img: np.ndarray, wh=(120,180)) -> np.ndarray:
    return cv2.resize(img, wh, interpolation=cv2.INTER_AREA)

def _to_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img

def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()

@dataclass
class FullCardLibrary:
    root: Path
    manifest_path: Optional[Path] = None
    def __post_init__(self):
        # carica manifest (obbligatorio per semantica chiara)
        mp = self.manifest_path or (self.root.parent / "manifest.json")
        if not mp.exists():
            raise RuntimeError(f"Manifest non trovato: {mp}")
        data = json.loads(mp.read_text(encoding="utf-8"))
        self.meta: Dict[str, Dict[str,str]] = data.get("full", {})

        self.templates: List[Tuple[str, np.ndarray]] = []
        self.gray_resized: Dict[str, np.ndarray] = {}
        for p in sorted(self.root.glob("*.png")):
            bgr = _to_bgr(p)
            if bgr is None: continue
            name = p.name
            g = cv2.cvtColor(_resize(bgr), cv2.COLOR_BGR2GRAY)
            self.templates.append((name, g))
            self.gray_resized[name] = g
        if not self.templates:
            raise RuntimeError(f"Nessun template in {self.root}")

    def semantics(self, name: str) -> Tuple[Optional[str], Optional[str]]:
        m = self.meta.get(name, {})
        s = m.get("suit") or None
        r = m.get("rank") or None
        return s, r

class Recognizer:
    def __init__(self, lib: FullCardLibrary, thr: float = 0.58):
        self.lib = lib
        self.thr = thr
        self._cache: Dict[str, Tuple[str, float, str]] = {}
        # cache: roi_md5 -> (best_name, score, human_readable)

    def best_match(self, roi_bgr: Optional[np.ndarray]) -> Tuple[Optional[str], float, str]:
        if roi_bgr is None:
            return None, -1.0, "noimg"
        roi_small = _resize(roi_bgr)
        g = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
        key = _md5(g)
        if key in self._cache:
            name, sc, tag = self._cache[key]
            return name, sc, tag

        best_name, best_sc = None, -1.0
        for name, templ in self.lib.templates:
            res = cv2.matchTemplate(g, templ, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, _ = cv2.minMaxLoc(res)
            if maxVal > best_sc:
                best_name, best_sc = name, float(maxVal)

        s, r = self.lib.semantics(best_name) if best_name else (None, None)
        tag = f"{best_name or 'None'} | suit={s or '?'} rank={r or '?'} | s={best_sc:.2f}"
        self._cache[key] = (best_name, best_sc, tag)
        if best_sc < self.thr:
            return None, best_sc, tag
        return best_name, best_sc, tag
