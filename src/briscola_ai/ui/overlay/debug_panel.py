from __future__ import annotations
import os, time
from typing import Dict, Any, Optional
import numpy as np, cv2

class DebugPanel:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _to_u8(self, img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    def save_recognizer_debug(self, key: str, roi_bgr: np.ndarray, dbg: Optional[Dict[str,Any]]) -> Optional[str]:
        if not dbg: return None
        roi_small = cv2.resize(roi_bgr, (240,160), interpolation=cv2.INTER_AREA)
        panel = roi_small.copy()
        txt = f"{key} suit={dbg.get('suit')} rank={dbg.get('rank')} scale={dbg.get('scale')} tm={dbg.get('tm'):.3f} edge={dbg.get('edge'):.3f}"
        cv2.putText(panel, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        ts = time.strftime('%Y%m%d_%H%M%S')
        out = os.path.join(self.out_dir, f"{ts}_{key}_dbg.png")
        cv2.imwrite(out, panel)
        return out
