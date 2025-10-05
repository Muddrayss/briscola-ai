from __future__ import annotations
from typing import Optional, Tuple, List
import cv2, numpy as np

class ORBRecognizer:
    def __init__(self, lib, thr_matches:int=18, ratio:float=0.75):
        self.lib = lib
        self.thr_matches = thr_matches
        self.ratio = ratio
        self.orb = cv2.ORB_create(nfeatures=800, fastThreshold=7)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # precompute kp/desc per template (gray 120x180)
        self.tpl_kpdes: List[Tuple[str, List[cv2.KeyPoint], np.ndarray]] = []
        for name, templ in self.lib.templates:
            kp, des = self.orb.detectAndCompute(templ, None)
            self.tpl_kpdes.append((name, kp, des))

    def best_match(self, roi_bgr) -> Tuple[Optional[str], float, str]:
        if roi_bgr is None: return (None, -1.0, "noimg")
        small = cv2.resize(roi_bgr, (120,180), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(g, None)
        if des is None or len(kp)==0: return (None, -1.0, "no-kp")

        best_name, best_good = None, -1
        for name, tkp, tdes in self.tpl_kpdes:
            if tdes is None: continue
            matches = self.bf.knnMatch(des, tdes, k=2)
            good = 0
            for m,n in matches:
                if m.distance < self.ratio * n.distance:
                    good += 1
            if good > best_good:
                best_name, best_good = name, good

        s, r = self.lib.semantics(best_name) if best_name else (None, None)
        tag = f"{best_name or 'None'} | suit={s or '?'} rank={r or '?'} | good={best_good}"
        if best_good >= self.thr_matches:
            return best_name, float(best_good), tag
        return (None, float(best_good), tag)
