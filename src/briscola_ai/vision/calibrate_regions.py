from __future__ import annotations
import sys, json, argparse
from typing import Dict, Tuple, Optional
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QGuiApplication, QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget, QFileDialog

# --- overlay settings ---
EDGE = QPen(QColor(0, 210, 255), 2)
FILL = QColor(0, 200, 255, 70)
DIM  = QColor(0, 0, 0, 90)    # per opzione --dim
INVISIBLE_BG = QColor(0, 0, 0, 1)  # quasi invisibile ma cattura i click

HELP_TEXT = (
    "Calibrazione ROIs (trasparente, full-screen)\n"
    "Mouse: trascina per creare/aggiornare il rettangolo corrente\n"
    "Invio: aggiungi/aggiorna ROI con il nome corrente\n"
    "CTRL+S: salva JSON    |    ESC: esci\n"
    "Tasti rapidi nome: 1,2,3,  T=trump,  L=lead,  F=follow\n"
    "Frecce: sposta di 1px    |    Shift+Frecce: ridimensiona (w/h) di 1px\n"
)

class TransparentCalibrator(QWidget):
    def __init__(self, screen_index: int = 0, dim_bg: bool = False):
        super().__init__()
        screens = QGuiApplication.screens()
        if not screens:
            raise RuntimeError("Nessuno schermo trovato.")
        if not (0 <= screen_index < len(screens)):
            raise RuntimeError(f"Indice schermo non valido: {screen_index}. Disponibili: 0..{len(screens)-1}")

        self.scr = screens[screen_index]
        geo = self.scr.geometry()  # coordinate assolute nel desktop virtuale
        self.setGeometry(geo)

        self.setWindowTitle(f"Calibrazione ROIs - Monitor {screen_index}")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        # assicura focus per le scorciatoie
        self.setFocusPolicy(Qt.StrongFocus)

        self.dim_bg = dim_bg

        self.rois: Dict[str, QRect] = {}
        self.current: Optional[QRect] = None
        self.dragging = False
        self.name_buffer: str = ""   # nome ROI corrente (digitato o scelto con hotkey)

    # ---------- input ----------
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.dragging = True
            p = ev.globalPosition().toPoint()  # assoluto
            self.current = QRect(p, p)
            self.update()

    def mouseMoveEvent(self, ev):
        if self.dragging and self.current:
            p = ev.globalPosition().toPoint()
            self.current.setBottomRight(p)
            self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.dragging = False
            self.update()

    def keyPressEvent(self, ev: QKeyEvent):
        key = ev.key()
        mod = ev.modifiers()

        # --- SCORCIATOIE PRIMA DI TUTTO ---
        # ESC = esci
        if key == Qt.Key_Escape:
            self.close()
            return
        # CTRL+S = salva
        if (mod & Qt.ControlModifier) and key == Qt.Key_S:
            self._save()
            return
        # ENTER = commit ROI corrente
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._commit_current()
            return

        # --- hotkeys nome veloci ---
        if key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_T, Qt.Key_L, Qt.Key_F):
            mapping = {
                Qt.Key_1:"bottom_slot_1",
                Qt.Key_2:"bottom_slot_2",
                Qt.Key_3:"bottom_slot_3",
                Qt.Key_T:"trump",
                Qt.Key_L:"lead",
                Qt.Key_F:"follow"
            }
            self.name_buffer = mapping[key]
            self.update()
            return

        # --- testo libero per il nome ---
        if (Qt.Key_A <= key <= Qt.Key_Z) or (Qt.Key_0 <= key <= Qt.Key_9) or key == Qt.Key_Underscore:
            ch = ev.text()
            if ch:
                self.name_buffer += ch
                self.update()
                return
        if key == Qt.Key_Backspace:
            self.name_buffer = self.name_buffer[:-1] if self.name_buffer else ""
            self.update()
            return

        # --- frecce: muovi / ridimensiona ---
        if self.current:
            step = 1
            if key == Qt.Key_Left:
                if mod & Qt.ShiftModifier:
                    self.current.setWidth(max(1, self.current.width()-step))
                else:
                    self.current.translate(-step, 0)
            elif key == Qt.Key_Right:
                if mod & Qt.ShiftModifier:
                    self.current.setWidth(self.current.width()+step)
                else:
                    self.current.translate(step, 0)
            elif key == Qt.Key_Up:
                if mod & Qt.ShiftModifier:
                    self.current.setHeight(max(1, self.current.height()-step))
                else:
                    self.current.translate(0, -step)
            elif key == Qt.Key_Down:
                if mod & Qt.ShiftModifier:
                    self.current.setHeight(self.current.height()+step)
                else:
                    self.current.translate(0, step)
            self.update()

    # ---------- actions ----------
    def _commit_current(self):
        if not self.current:
            return
        rect = self.current.normalized()
        if rect.width() < 4 or rect.height() < 4:
            return
        name = self.name_buffer.strip() or f"roi_{len(self.rois)+1}"
        self.rois[name] = QRect(rect)
        # lasciamo il rettangolo selezionato per micro-nudge
        self.update()
        print(f"[calib] ROI '{name}' = {rect.x()},{rect.y()},{rect.width()},{rect.height()}")

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Salva calibrazione", "calibration.json", "JSON (*.json)")
        if not path:
            return
        data = {
            "screen_index": QGuiApplication.screens().index(self.scr),
            "screen_geometry": [self.geometry().x(), self.geometry().y(), self.geometry().width(), self.geometry().height()],
            "rois": {k: [r.x(), r.y(), r.width(), r.height()] for k, r in self.rois.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[calib] Salvato: {path}")

    # ---------- render ----------
    def showEvent(self, ev):
        super().showEvent(ev)
        # assicura focus/finestra in primo piano
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)

        # Importante: anche senza --dim, dipingi un alpha=1 invisibile per catturare i click
        qp.fillRect(self.rect(), (DIM if self.dim_bg else INVISIBLE_BG))

        # esistenti
        qp.setPen(EDGE)
        for name, r in self.rois.items():
            qp.drawRect(r)
            qp.drawText(r.adjusted(2,2,-2,-2), Qt.AlignTop|Qt.AlignLeft, name)

        # corrente
        if self.current:
            qp.setPen(EDGE)
            qp.fillRect(self.current.normalized(), FILL)
            qp.drawRect(self.current.normalized())

        # HUD
        qp.setPen(QPen(QColor(255,255,255), 1))
        qp.drawText(self.rect().adjusted(10,10,-10,-10), Qt.AlignTop|Qt.AlignLeft,
                    HELP_TEXT + f"\nNome corrente: {self.name_buffer or '(vuoto)'}")

def run(screen_index: int = 0, dim: bool = False):
    w = TransparentCalibrator(screen_index=screen_index, dim_bg=dim)
    w.showFullScreen()
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=int, default=0, help="Indice monitor (0=primario)")
    ap.add_argument("--dim", action="store_true", help="Oscura leggermente lo sfondo (utile per contrasto)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    w = run(screen_index=args.screen, dim=args.dim)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
