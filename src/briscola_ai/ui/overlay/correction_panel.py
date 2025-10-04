from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QPushButton
)
from PySide6.QtCore import Signal

SUITS = ["(auto)","denari","coppe","bastoni","spade"]
RANKS = ["(auto)","A","3","K","C","F","7","6","5","4","2"]

class CorrectionPanel(QWidget):
    # segnali “dati”
    changed_bottom = Signal(int, str)                 # (slot_idx, template or "")
    changed_trump_card = Signal(str, str)             # (suit, rank) either can be "(auto)"
    changed_lead_card  = Signal(bool, str, str)       # (enabled, suit, rank)
    changed_last_lead  = Signal(str, str)             # (suit, rank)
    changed_last_follow= Signal(str, str)             # (suit, rank)

    # segnali “controllo”
    start_clicked = Signal()
    pause_clicked = Signal()
    reset_clicked = Signal()
    new_hand_clicked = Signal()                       # nessuna lead sul tavolo
    new_trick_clicked = Signal()                      # c'è lead sul tavolo
    lock_bottom_toggled = Signal(bool)

    def __init__(self, template_names):
        super().__init__()
        self.setWindowTitle("Briscola Advisor - Controller")
        self.setFixedWidth(400)

        root = QVBoxLayout(self)

        # ---- controllo partita ----
        rowc = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_reset = QPushButton("Reset")
        rowc.addWidget(self.btn_start); rowc.addWidget(self.btn_pause); rowc.addWidget(self.btn_reset)
        root.addLayout(rowc)

        rowm = QHBoxLayout()
        self.btn_hand = QPushButton("New Hand")
        self.btn_trick = QPushButton("New Trick")
        self.chk_lock = QCheckBox("Lock bottom (freeze)")
        rowm.addWidget(self.btn_hand); rowm.addWidget(self.btn_trick); rowm.addWidget(self.chk_lock)
        root.addLayout(rowm)

        # ---- correzioni bottom ----
        root.addWidget(QLabel("Bottom cards override (facoltativo):"))
        for i in range(3):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Bottom {i+1}:"))
            cb = QComboBox()
            cb.addItem("(auto)")
            for nm in template_names:
                cb.addItem(nm)
            cb.currentIndexChanged.connect(lambda _, s=i, w=cb: self._on_bottom(s, w))
            row.addWidget(cb)
            root.addLayout(row)

        # ---- briscola (seme + rango) ----
        root.addWidget(QLabel("Trump (briscola)"))
        rowt1 = QHBoxLayout()
        rowt1.addWidget(QLabel("Suit:"))
        self.cb_trump_s = QComboBox(); [self.cb_trump_s.addItem(x) for x in SUITS]
        rowt1.addWidget(self.cb_trump_s)
        rowt1.addWidget(QLabel("Rank:"))
        self.cb_trump_r = QComboBox(); [self.cb_trump_r.addItem(x) for x in RANKS]
        rowt1.addWidget(self.cb_trump_r)
        self.cb_trump_s.currentIndexChanged.connect(self._emit_trump)
        self.cb_trump_r.currentIndexChanged.connect(self._emit_trump)
        root.addLayout(rowt1)

        # ---- lead corrente (sul tavolo ora) ----
        root.addWidget(QLabel("Lead (trick corrente)"))
        rowl1 = QHBoxLayout()
        self.chk_lead = QCheckBox("Lead presente"); self.chk_lead.setChecked(True)
        rowl1.addWidget(self.chk_lead)
        rowl2 = QHBoxLayout()
        rowl2.addWidget(QLabel("Suit:"))
        self.cb_lead_s = QComboBox(); [self.cb_lead_s.addItem(x) for x in SUITS]
        rowl2.addWidget(self.cb_lead_s)
        rowl2.addWidget(QLabel("Rank:"))
        self.cb_lead_r = QComboBox(); [self.cb_lead_r.addItem(x) for x in RANKS]
        rowl2.addWidget(self.cb_lead_r)
        self.chk_lead.stateChanged.connect(self._emit_lead)
        self.cb_lead_s.currentIndexChanged.connect(self._emit_lead)
        self.cb_lead_r.currentIndexChanged.connect(self._emit_lead)
        root.addLayout(rowl1); root.addLayout(rowl2)

        # ---- ultimo trick (già chiuso) ----
        root.addWidget(QLabel("Last trick (già giocato)"))
        rowp1 = QHBoxLayout()
        rowp1.addWidget(QLabel("Last LEAD:"))
        self.cb_lastlead_s = QComboBox(); [self.cb_lastlead_s.addItem(x) for x in SUITS]
        self.cb_lastlead_r = QComboBox(); [self.cb_lastlead_r.addItem(x) for x in RANKS]
        rowp1.addWidget(self.cb_lastlead_s); rowp1.addWidget(self.cb_lastlead_r)
        rowp2 = QHBoxLayout()
        rowp2.addWidget(QLabel("Last FOLLOW:"))
        self.cb_lastfollow_s = QComboBox(); [self.cb_lastfollow_s.addItem(x) for x in SUITS]
        self.cb_lastfollow_r = QComboBox(); [self.cb_lastfollow_r.addItem(x) for x in RANKS]
        rowp2.addWidget(self.cb_lastfollow_s); rowp2.addWidget(self.cb_lastfollow_r)
        self.cb_lastlead_s.currentIndexChanged.connect(self._emit_last)
        self.cb_lastlead_r.currentIndexChanged.connect(self._emit_last)
        self.cb_lastfollow_s.currentIndexChanged.connect(self._emit_last)
        self.cb_lastfollow_r.currentIndexChanged.connect(self._emit_last)
        root.addLayout(rowp1); root.addLayout(rowp2)

        # wiring bottoni
        self.btn_start.clicked.connect(self.start_clicked.emit)
        self.btn_pause.clicked.connect(self.pause_clicked.emit)
        self.btn_reset.clicked.connect(self.reset_clicked.emit)
        self.btn_hand.clicked.connect(self.new_hand_clicked.emit)
        self.btn_trick.clicked.connect(self.new_trick_clicked.emit)
        self.chk_lock.stateChanged.connect(lambda st: self.lock_bottom_toggled.emit(bool(st)))

    # ---- emitters ----
    def _on_bottom(self, slot_idx, combo):
        txt = combo.currentText()
        self.changed_bottom.emit(slot_idx, "" if txt=="(auto)" else txt)

    def _emit_trump(self):
        self.changed_trump_card.emit(self.cb_trump_s.currentText(), self.cb_trump_r.currentText())

    def _emit_lead(self):
        self.changed_lead_card.emit(self.chk_lead.isChecked(), self.cb_lead_s.currentText(), self.cb_lead_r.currentText())

    def _emit_last(self):
        self.changed_last_lead.emit(self.cb_lastlead_s.currentText(), self.cb_lastlead_r.currentText())
        self.changed_last_follow.emit(self.cb_lastfollow_s.currentText(), self.cb_lastfollow_r.currentText())
