from __future__ import annotations
import json, os
from typing import Dict, Tuple, Optional, List
from PySide6.QtCore import Qt, QRect, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QGuiApplication, QFont, QFontMetrics, QPixmap, QTextOption
from PySide6.QtWidgets import QWidget

HUD_KEYS_PRIORITY = ["status","BLOCK","ERROR","WARN","suggest","SUGGEST","ROI_MAP","ROI_KEYS"]

class AdvisorOverlay(QWidget):
    """ Transparent overlay HUD that draws inside screen ROIs. """
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
        self._show_debug_panel: bool = os.getenv("BRISCOLA_DEBUG_PANEL","0") == "1"  # default OFF
        self._debug_panel_path: Optional[str] = None

        self._hud_anchor: str = "tr"
        self._hud_scale: float = 1.0
        self._hud_margin: int = 16

        geo = QGuiApplication.primaryScreen().availableGeometry()
        self.setGeometry(geo)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

    # public API
    def set_hud_anchor(self, where: str): self._hud_anchor = where
    def set_hud_scale(self, s: float): self._hud_scale = float(s)
    def set_suggestion(self, key: Optional[str]): self.suggest_key = key
    def set_debug_info(self, d: Dict[str, str]): self.debug_texts = dict(d or {})
    def set_thumbnails(self, paths: Dict[str, str]): self.thumb_paths = dict(paths or {})
    def set_debug_panel(self, path: Optional[str]): self._debug_panel_path = path

    def _load_calibration(self, path: str):
        try:
            data = json.loads(open(path,"r",encoding="utf-8").read())
        except Exception:
            data = {}
        self.rois = {k: tuple(map(int,v)) for k,v in data.get("rois", {}).items()}

    def _thumb(self, path: str) -> Optional[QPixmap]:
        if not path or not os.path.exists(path): return None
        pm = self._pix_cache.get(path)
        if pm is None:
            pm = QPixmap(path)
            self._pix_cache[path] = pm
        return pm

    def _build_hud_lines(self) -> List[str]:
        lines: List[str] = []
        globals_ = {k:v for k,v in (self.debug_texts or {}).items() if k not in self.rois}
        for k in HUD_KEYS_PRIORITY:
            if k in globals_:
                lines.append(f"{k}: {globals_[k]}"); globals_.pop(k, None)
        for k,v in sorted(globals_.items()):
            lines.append(f"{k}: {v}")
        return lines[:18]

    def _choose_font_to_fit(self, rect: QRect, text: str) -> QFont:
        """Pick a font size that fits text inside rect (using wrap)."""
        max_h = max(20, rect.height() * 0.26)  # text band height
        max_w = rect.width()
        size = max(8, int(max_h * 0.42))  # initial guess
        opt = QTextOption(); opt.setWrapMode(QTextOption.WordWrap); opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        while size > 8:
            f = QFont("Consolas", int(size))
            fm = QFontMetrics(f)
            bounds = fm.boundingRect(0, 0, int(max_w - 12), 10_000, Qt.TextWordWrap, text)
            if bounds.height() <= max_h:
                return f
            size -= 1
        return QFont("Consolas", 9)

    def _draw_wrapped_label_top(self, qp: QPainter, rect: QRect, text: str):
        # Top band inside ROI
        pad = 6
        bh = int(rect.height()*0.26)
        br = QRect(rect.x()+pad, rect.y()+pad, rect.width()-2*pad, max(20, bh))
        # Semi-transparent dark band for contrast
        qp.fillRect(br, QColor(0,0,0,165))

        # Auto fit font
        font = self._choose_font_to_fit(br, text)
        qp.setFont(font)
        opt = QTextOption(); opt.setWrapMode(QTextOption.WordWrap); opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Outline then foreground for legibility
        qp.setPen(QPen(QColor(0,0,0,230), 3))
        qp.drawText(br, text, opt)
        qp.setPen(QPen(QColor(255,255,255,235), 1))
        qp.drawText(br, text, opt)

    def paintEvent(self, e):
        qp = QPainter(self); qp.setRenderHint(QPainter.Antialiasing, True)
        # Draw ROIs and captions
        for k, (x,y,w,h) in self.rois.items():
            rect = QRect(x,y,w,h)
            # ROI border
            qp.setPen(QPen(QColor(0,255,150,200), 2))
            qp.drawRect(rect)

            # Per-ROI thumbnail in bottom-left INSIDE the ROI, keep aspect ratio
            pth = (self.thumb_paths or {}).get(k)
            pm = self._thumb(pth) if pth else None
            if pm and not pm.isNull():
                dw = int(w*0.33); dh = int(h*0.33)
                pm2 = pm.scaled(dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                qp.drawPixmap(x+4, y+h-pm2.height()-4, pm2)

            # Per-ROI text (wrapped and constrained at TOP)
            t = (self.debug_texts or {}).get(k)
            if t:
                self._draw_wrapped_label_top(qp, rect, t)

        # Suggest highlight
        if self.suggest_key and self.suggest_key in self.rois:
            x,y,w,h = self.rois[self.suggest_key]
            qp.setPen(QPen(QColor(255,230,0,220), 3))
            for off in (0,3):
                qp.drawRect(QRect(x-off,y-off,w+2*off,h+2*off))

        # Optional small debug panel (bottom-left of screen), OFF by default
        if self._show_debug_panel and self._debug_panel_path:
            pm = self._thumb(self._debug_panel_path)
            if pm and not pm.isNull():
                dw, dh = int(self.width()*0.22), int(self.height()*0.16)
                qp.drawPixmap(12, self.height()-dh-12, dw, dh, pm)

        # Global HUD box (top-right by default)
        lines = self._build_hud_lines()
        if lines:
            scale = self._hud_scale
            font = QFont("Consolas", int(12*scale)); qp.setFont(font)
            fm = QFontMetrics(font)
            box_w = max((fm.horizontalAdvance(s) for s in lines), default=0) + 24
            box_h = int(len(lines)*fm.height()*1.10)+16
            x = self.width()-box_w-self._hud_margin if self._hud_anchor in ("tr","br") else self._hud_margin
            y = self.height()-box_h-self._hud_margin if self._hud_anchor in ("br","bl") else self._hud_margin
            qp.fillRect(QRect(x,y,box_w,box_h), QColor(10,10,10,170))
            qp.setPen(QPen(QColor(255,255,255,235), 1))
            for i,s in enumerate(lines):
                qp.drawText(QPoint(x+10, y+12 + int((i+1)*fm.height()*1.0)), s)

        qp.end()
