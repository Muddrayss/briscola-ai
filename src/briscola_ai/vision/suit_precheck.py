from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple

SUITS = ("denari","coppe","bastoni","spade")

def _mask_range(hsv: np.ndarray, low: Tuple[int,int,int], high: Tuple[int,int,int]) -> np.ndarray:
    low = np.array(low, dtype=np.uint8); high = np.array(high, dtype=np.uint8)
    return cv2.inRange(hsv, low, high)

def estimate_suit(bgr: np.ndarray, erode_iters: int = 1, blur_ksize: int = 3) -> Optional[str]:
    """Heuristic HSV pre-check for italian suits. Returns one of {denari,coppe,bastoni,spade} or None."""
    if bgr is None:
        return None
    small = cv2.resize(bgr, (120,180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    if blur_ksize > 1:
        hsv = cv2.GaussianBlur(hsv, (blur_ksize, blur_ksize), 0)

    # coppe (ROSSO) - due range
    red1 = _mask_range(hsv, (0, 80, 70), (8, 255, 255))
    red2 = _mask_range(hsv, (172, 80, 70), (179, 255, 255))
    m_coppe = cv2.bitwise_or(red1, red2)

    # denari (GIALLO/ORO)
    m_denari = _mask_range(hsv, (18, 70, 80), (34, 255, 255))

    # bastoni (VERDE/MARRONE → verde dominante)
    m_bastoni = _mask_range(hsv, (35, 50, 40), (85, 255, 255))

    # spade (BLU/CIANO)
    m_spade = _mask_range(hsv, (86, 40, 40), (128, 255, 255))

    kernel = np.ones((3,3), np.uint8)
    for m in (m_coppe, m_denari, m_bastoni, m_spade):
        cv2.erode(m, kernel, m, iterations=erode_iters)
        cv2.dilate(m, kernel, m, iterations=erode_iters)

    counts = [int(cv2.countNonZero(m)) for m in (m_denari, m_coppe, m_bastoni, m_spade)]
    order = np.argsort(counts)[::-1]
    top = order[0]
    if counts[top] < 120:  # minima evidenza colore
        return None
    return ("denari","coppe","bastoni","spade")[top]
