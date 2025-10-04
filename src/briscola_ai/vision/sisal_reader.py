from __future__ import annotations
import json
from typing import Dict, Tuple, Optional, Any
import mss
import cv2
import numpy as np
from pathlib import Path

class SisalReader:
    """
    Cattura schermo su ROIs calibrate (json). Fornisce ritagli e utility base.
    """
    def __init__(self, calibration_path: str):
        self.cal_path = calibration_path
        self.cal = self._load_calibration(calibration_path)
        self.sct = mss.mss()

    def _load_calibration(self, path: str) -> Dict[str, Tuple[int,int,int,int]]:
        p = Path(path)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        rois = {k: tuple(v) for k, v in data.get("rois", {}).items()}
        return rois

    def grab_roi(self, key: str) -> Optional[np.ndarray]:
        if key not in self.cal: return None
        x,y,w,h = self.cal[key]
        mon = {"left": x, "top": y, "width": w, "height": h}
        img = self.sct.grab(mon)
        frame = np.asarray(img)[:,:,:3].copy()  # BGRA->BGR
        return frame

    def read_snapshot(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Ritorna un dizionario con i ritagli correnti delle aree standard.
        Keys suggerite in calibrazione:
          bottom_slot_1/2/3, trump, lead, follow
        """
        keys = ["bottom_slot_1","bottom_slot_2","bottom_slot_3","trump","lead","follow"]
        out = {}
        for k in keys:
            out[k] = self.grab_roi(k)
        return out
