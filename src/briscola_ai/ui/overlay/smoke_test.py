from __future__ import annotations
import sys, json, itertools, time
from PySide6.QtWidgets import QApplication
from .advisor_overlay import AdvisorOverlay

def main():
    if len(sys.argv)<2:
        print("Uso: python -m briscola_ai.ui.overlay.smoke_test <calibration.json>")
        sys.exit(1)
    cal = sys.argv[1]
    app = QApplication(sys.argv)
    ov = AdvisorOverlay(cal); ov.show()

    # leggi le chiavi ROI effettive dal file (per debug chiaro sullo schermo)
    with open(cal, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = list((data.get("rois") or {}).keys())

    if not keys:
        ov.set_debug_info({"ERROR":"No ROIs in calibration file"})
        sys.exit(app.exec())

    # se hai i nomi “canonici”, usali. Altrimenti cicla sulle chiavi reali:
    ov.set_debug_info({"ROI_KEYS": ", ".join(keys)})
    cycle = itertools.cycle(keys)

    # lampeggia tra le chiavi per vedere l’highlight
    def tick():
        ov.set_suggestion(next(cycle))
    from PySide6.QtCore import QTimer
    t = QTimer(); t.timeout.connect(tick); t.start(600)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
