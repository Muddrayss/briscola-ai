from __future__ import annotations
import json
from typing import Dict, Tuple, Optional, List
from PySide6.QtCore import Qt, QRect, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QGuiApplication, QFont, QFontMetrics, QPixmap
from PySide6.QtWidgets import QWidget

HUD_KEYS_PRIORITY = ["status","BLOCK","ERROR","WARN","suggest","SUGGEST","ROI_MAP","ROI_KEYS"]

class AdvisorOverlay(QWidget):
    """
    Overlay trasparente sempre-on-top:
      - disegna le ROI + (eventuale) highlight suggerito
      - mostra debug locale dentro ogni ROI (se presente)
      - mostra un HUD globale con le chiavi di debug non-ROI (status, BLOCK, WARN, ecc.)
    """
    def __init__(self, calibration_path: str):
        super().__init__()
        self.setWindowTitle("Briscola Advisor Overlay")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.rois: Dict[str, Tuple[int,int,int,int]] = {}
        self._load_calibration(calibration_path)

        self.suggest_key: Optional[str] = None
        self.debug_texts: Dict[str, str] = {}
        self.thumb_paths: Dict[str, str] = {}
        self._pix_cache: Dict[str, QPixmap] = {}

        # impostazioni HUD
        self._hud_anchor: str = "tr"   # "tl" "tr" "bl" "br"
        self._hud_scale: float = 1.0   # 1.0 = font 12, 1.2 = font ~14.4, ...
        self._hud_margin: int = 16

        # schermo intero
        geo = QGuiApplication.primaryScreen().availableGeometry()
        self.setGeometry(geo)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(33)

    # -------- API pubbliche ---------------------------------------------------
    def set_suggestion(self, key: Optional[str]):
        self.suggest_key = key

    def set_debug_info(self, texts: Dict[str, str]):
        # Accetta anche None / {} e le pulisce
        self.debug_texts = dict(texts) if texts else {}

    def set_thumbnails(self, paths: Dict[str, str]):
        # keys come "bottom_slot_1/2/3", "trump", "lead" ...
        self.thumb_paths = dict(paths) if paths else {}

    def set_hud_anchor(self, anchor: str):
        # "tl" "tr" "bl" "br"
        if anchor in ("tl","tr","bl","br"):
            self._hud_anchor = anchor

    def set_hud_scale(self, scale: float):
        self._hud_scale = max(0.7, min(2.0, scale))

    # -------- internals -------------------------------------------------------
    def _load_calibration(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.rois = {k: tuple(v) for k,v in data.get("rois", {}).items()}
        except Exception:
            self.rois = {}

    def _thumb(self, path: str) -> Optional[QPixmap]:
        if not path:
            return None
        pm = self._pix_cache.get(path)
        if pm is None:
            pm = QPixmap(path)
            self._pix_cache[path] = pm
        return pm

    def _build_hud_lines(self) -> List[str]:
        """
        Raccoglie le chiavi non-ROI del debug in un elenco di righe,
        con priorità per status/BLOCK/WARN/suggest/ROI_MAP/ROI_KEYS.
        """
        lines: List[str] = []

        # chiavi ROI (per evitare di duplicarle nel pannello globale)
        roi_keys = set(self.rois.keys())

        # prioritarie
        for k in HUD_KEYS_PRIORITY:
            if k in self.debug_texts and k not in roi_keys:
                lines.append(f"{k}: {self.debug_texts[k]}")

        # le altre, ordinate alfabeticamente, escludendo le ROI
        for k in sorted(self.debug_texts.keys()):
            if k in roi_keys or k in HUD_KEYS_PRIORITY:
                continue
            v = self.debug_texts[k]
            # evita righe vuote
            if v is None or (isinstance(v, str) and v.strip() == ""):
                continue
            lines.append(f"{k}: {v}")

        # se nessuna riga, metti stato base
        if not lines:
            lines.append("status: (no debug)")

        return lines

    # -------- paint -----------------------------------------------------------
    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)

        # --- ROI + label + debug locale + thumbnails
        qp.setPen(QPen(QColor(0,160,255,180), 2))
        roi_font = QFont("Segoe UI", 11)
        qp.setFont(roi_font)

        for name, r in self.rois.items():
            rect = QRect(*r)
            qp.drawRect(rect)
            # etichetta ROI
            qp.drawText(rect.adjusted(2,2,-2,-2), Qt.AlignTop|Qt.AlignLeft, name)
            # debug locale (sotto l'etichetta)
            if name in self.debug_texts:
                qp.drawText(rect.adjusted(2,20,-2,-2), Qt.AlignTop|Qt.AlignLeft, self.debug_texts[name])

            # thumbnail in basso-sinistra della ROI (se disponibile)
            if name in self.thumb_paths and self.thumb_paths[name]:
                pm = self._thumb(self.thumb_paths[name])
                if pm and not pm.isNull():
                    tw, th = int(rect.width()*0.28), int(rect.height()*0.35)
                    qp.drawPixmap(rect.x()+4, rect.bottom()-th-4, tw, th, pm)

        # --- highlight suggerito
        if self.suggest_key and self.suggest_key in self.rois:
            rect = QRect(*self.rois[self.suggest_key])
            qp.setPen(QPen(QColor(255,215,0,220), 5))
            qp.drawRect(rect)

        # --- HUD globale (pannello fisso in un angolo)
        lines = self._build_hud_lines()
        base_font_pt = int(12 * self._hud_scale)
        hud_font = QFont("Consolas", base_font_pt)  # monospace leggibile
        qp.setFont(hud_font)
        fm = QFontMetrics(hud_font)

        # calcola larghezza/altezza del pannello in base alle righe
        maxw = 0
        for ln in lines:
            w = fm.horizontalAdvance(ln)
            if w > maxw: maxw = w
        line_h = fm.height()
        pad = int(10 * self._hud_scale)
        box_w = maxw + pad*2
        box_h = line_h*len(lines) + pad*2

        # posizione in base all'anchor
        screen_rect = self.geometry()
        x = self._hud_margin
        y = self._hud_margin
        if self._hud_anchor[1] == "r":  # right
            x = screen_rect.width() - box_w - self._hud_margin
        if self._hud_anchor[0] == "b":  # bottom
            y = screen_rect.height() - box_h - self._hud_margin

        # sfondo semitrasparente
        bg = QColor(0,0,0,170)
        qp.fillRect(x, y, box_w, box_h, bg)

        # bordo leggero
        qp.setPen(QPen(QColor(255,255,255,180), 1))
        qp.drawRect(x, y, box_w, box_h)

        # testo (bianco)
        qp.setPen(QColor(255,255,255,220))
        tx = x + pad
        ty = y + pad + fm.ascent()
        for ln in lines:
            qp.drawText(QPoint(tx, ty), ln)
            ty += line_h

        qp.end()
