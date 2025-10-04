from __future__ import annotations
import sys, random
from dataclasses import dataclass
from typing import Optional, List, Tuple

from PySide6.QtCore import Qt, QTimer, QRect, QPoint
from PySide6.QtGui import QPainter, QFont, QPen, QColor
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QSlider, QCheckBox, QMessageBox, QComboBox

from ...core.env import BriscolaEnv, Observation
from ...core.cards import Card, Suit
from ...agents.rule_based import BestChoiceFirstAgent, HighestFirstAgent, RandomAgent
from ...agents.mcts import ISMCTSAgent

# ---- UI params ----
TABLE_BG = QColor(22, 80, 45)
CARD_W, CARD_H = 92, 138
RADIUS = 10
MARGIN = 28
CENTER_Y = 280

@dataclass
class AgentSpec:
    name: str
    make: callable

AGENTS: List[AgentSpec] = [
    AgentSpec("MCTS(800,c=1.0,disc=0.7)", lambda: ISMCTSAgent(iterations=800, c=1.0, discount=0.7, seed=42)),
    AgentSpec("MCTS(400,c=1.0,disc=0.7)", lambda: ISMCTSAgent(iterations=400, c=1.0, discount=0.7, seed=42)),
    AgentSpec("BestChoice", lambda: BestChoiceFirstAgent()),
    AgentSpec("HighestFirst", lambda: HighestFirstAgent()),
    AgentSpec("Random", lambda: RandomAgent()),
]

def suit_symbol(s: Suit) -> str:
    # simboli "approssimati" (non napoletani disegnati); ok per debug
    return {"denari":"◆","coppe":"♥","bastoni":"♣","spade":"♠"}[s.value]

class GameWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Briscola - Sisal-like (debug)")
        self.setMinimumSize(1000, 720)

        # env & agents
        self.env = BriscolaEnv(seed=42)
        self.agent_top = AGENTS[0].make()    # P1 (top)
        self.agent_bottom = AGENTS[2].make() # P0 (bottom) default BestChoice

        # control state
        self.autoplay = False
        self.delay_ms = 450
        self.human_bottom = True   # puoi disabilitarlo per AI vs AI
        self._pending_human_click: Optional[int] = None  # indice carta cliccata

        # UI controls
        self.btn_new = QPushButton("Nuova Partita")
        self.btn_step = QPushButton("Step (1 mossa)")
        self.btn_auto = QPushButton("Autoplay")
        self.btn_swap = QPushButton("Swap Agenti")
        self.chk_human = QCheckBox("Bottom: umano")
        self.chk_human.setChecked(True)

        self.lbl_speed = QLabel("Velocità")
        self.slider = QSlider(Qt.Horizontal); self.slider.setMinimum(80); self.slider.setMaximum(1200)
        self.slider.setValue(self.delay_ms)
        self.slider.valueChanged.connect(lambda v: self._set_delay(v))

        self.sel_top = QComboBox(); self.sel_bottom = QComboBox()
        for a in AGENTS:
            self.sel_top.addItem(a.name)
            self.sel_bottom.addItem(a.name)
        self.sel_top.setCurrentIndex(0)     # MCTS
        self.sel_bottom.setCurrentIndex(2)  # BestChoice

        row1 = QHBoxLayout()
        row1.addWidget(self.btn_new); row1.addWidget(self.btn_step); row1.addWidget(self.btn_auto); row1.addWidget(self.btn_swap)
        row1.addStretch(1)
        row1.addWidget(QLabel("TOP")); row1.addWidget(self.sel_top)
        row1.addWidget(QLabel("BOTTOM")); row1.addWidget(self.sel_bottom)
        row1.addWidget(self.chk_human)
        row1.addWidget(self.lbl_speed); row1.addWidget(self.slider)

        root = QVBoxLayout(self)
        root.addLayout(row1)

        # timer loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # signals
        self.btn_new.clicked.connect(self._new_game)
        self.btn_auto.clicked.connect(self._toggle_auto)
        self.btn_step.clicked.connect(self._once)
        self.btn_swap.clicked.connect(self._swap_agents)
        self.sel_top.currentIndexChanged.connect(self._rebuild_agents)
        self.sel_bottom.currentIndexChanged.connect(self._rebuild_agents)
        self.chk_human.stateChanged.connect(lambda _ : self._toggle_human())

        self._new_game()

    # ----- control handlers -----
    def _rebuild_agents(self):
        self.agent_top = AGENTS[self.sel_top.currentIndex()].make()
        self.agent_bottom = AGENTS[self.sel_bottom.currentIndex()].make()

    def _toggle_human(self):
        self.human_bottom = self.chk_human.isChecked()

    def _set_delay(self, v: int):
        self.delay_ms = int(v)
        if self.autoplay:
            self.timer.setInterval(self.delay_ms)

    def _new_game(self):
        self.env.reset()
        self._pending_human_click = None
        self.update()

    def _toggle_auto(self):
        self.autoplay = not self.autoplay
        if self.autoplay:
            self.timer.start(self.delay_ms)
            self.btn_auto.setText("Stop")
        else:
            self.timer.stop()
            self.btn_auto.setText("Autoplay")

    def _once(self):
        self._do_one_move()
        self.update()

    def _swap_agents(self):
        top_i = self.sel_top.currentIndex()
        bot_i = self.sel_bottom.currentIndex()
        self.sel_top.setCurrentIndex(bot_i)
        self.sel_bottom.setCurrentIndex(top_i)
        self._rebuild_agents()

    # ----- game logic -----
    def _do_one_move(self):
        if self.env.done:
            return
        p = self.env.turn_player
        obs = self.env.observe(p)

        if p == 0:
            if self.human_bottom:
                if self._pending_human_click is None:
                    # attendi clic
                    return
                idx = self._pending_human_click
                self._pending_human_click = None
                self.env.step(p, idx)
                return
            else:
                idx = self.agent_bottom.act(self.env, obs)
                self.env.step(p, idx)
                return
        else:
            idx = self.agent_top.act(self.env, obs)
            self.env.step(p, idx)

    def _tick(self):
        if not self.env.done:
            self._do_one_move()
            self.update()
        else:
            self.timer.stop()
            self.autoplay = False
            self.btn_auto.setText("Autoplay")
            self.update()
            # popup risultato
            p0, p1 = self.env.points
            if p0 > p1: msg = f"Vince BOTTOM ({p0}–{p1})"
            elif p1 > p0: msg = f"Vince TOP ({p1}–{p0})"
            else: msg = f"Pareggio ({p0}–{p1})"
            QMessageBox.information(self, "Partita finita", msg)

    # ----- rendering -----
    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.fillRect(self.rect(), TABLE_BG)

        # board rect
        W, H = self.width(), self.height()
        cx = W//2

        # draw deck & briscola
        self._draw_deck_and_trump(qp, QRect(W-180, 40, CARD_W, CARD_H))

        # trick area (center)
        self._draw_trick(qp, QPoint(cx- (CARD_W+16), CENTER_Y- CARD_H//2))

        # hands
        self._draw_top_hand(qp, QRect(cx- (3*CARD_W + 2*16)//2, 80, 3*CARD_W + 2*16, CARD_H))
        self._draw_bottom_hand(qp, QRect(cx- (3*CARD_W + 2*16)//2, H- CARD_H - 60, 3*CARD_W + 2*16, CARD_H))

        # score
        self._draw_score(qp, QRect(40, 40, 220, 100))

        qp.end()

    def _draw_card(self, qp: QPainter, r: QRect, card: Optional[Card], face_up: bool, highlight: bool=False):
        # background
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(245,245,245) if face_up else QColor(180,180,180))
        qp.drawRoundedRect(r, RADIUS, RADIUS)
        # edge
        pen = QPen(QColor(30,30,30), 2)
        qp.setPen(pen); qp.setBrush(Qt.NoBrush)
        qp.drawRoundedRect(r, RADIUS, RADIUS)
        if not face_up or card is None:
            return
        # rank/suit
        qp.setPen(QPen(QColor(25,25,25), 2))
        font = QFont("Segoe UI", 22); qp.setFont(font)
        qp.drawText(r.adjusted(10, 8, -10, -r.height()//2), Qt.AlignLeft|Qt.AlignTop, card.rank)
        qp.drawText(r.adjusted(0, 8, -10, -r.height()//2), Qt.AlignRight|Qt.AlignTop, suit_symbol(card.suit))
        font2 = QFont("Segoe UI", 18); qp.setFont(font2)
        qp.drawText(r.adjusted(0, 0, 0, -8), Qt.AlignCenter, f"{card.rank}{suit_symbol(card.suit)}")
        if highlight:
            hl = QPen(QColor(255,215,0), 5); qp.setPen(hl)
            qp.drawRoundedRect(r.adjusted(2,2,-2,-2), RADIUS, RADIUS)

    def _draw_deck_and_trump(self, qp: QPainter, r: QRect):
        # deck count & trump face-up a lato
        # deck
        drect = QRect(r.x(), r.y(), CARD_W, CARD_H)
        self._draw_card(qp, drect, None, face_up=False)
        deck_n = self.env.observe(self.env.turn_player).deck_count
        qp.setPen(QPen(QColor(255,255,255)))
        qp.drawText(drect.adjusted(0,0,0,-CARD_H//4), Qt.AlignCenter, f"{deck_n}")
        # trump
        tr = QRect(r.x()+CARD_W+16, r.y(), CARD_W, CARD_H)
        trump = self.env.trump_card if self.env.trump_card is not None else Card(self.env.briscola, " ")
        self._draw_card(qp, tr, trump, face_up=True)
        qp.setPen(QPen(QColor(230,230,230))); qp.drawText(tr.adjusted(0,-22,0,0), Qt.AlignCenter, "Briscola")

    def _draw_trick(self, qp: QPainter, top_left: QPoint):
        lead = self.env.lead_card; follow = self.env.follow_card
        r1 = QRect(top_left.x(), top_left.y(), CARD_W, CARD_H)
        r2 = QRect(top_left.x()+CARD_W+16, top_left.y(), CARD_W, CARD_H)
        # chi ha iniziato il trick è leader
        leader = self.env.leader
        # highlight la carta del giocatore di turno quando è sul trick?
        self._draw_card(qp, r1, lead, face_up=True)
        self._draw_card(qp, r2, follow, face_up=True)

    def _draw_top_hand(self, qp: QPainter, area: QRect):
        # back-of-card for opponent
        start_x = area.x()
        for i in range(len(self.env.hands[1])):
            r = QRect(start_x + i*(CARD_W+16), area.y(), CARD_W, CARD_H)
            self._draw_card(qp, r, None, face_up=False)

    def _draw_bottom_hand(self, qp: QPainter, area: QRect):
        # show your hand (face up). Highlight if is your turn
        p = 0
        obs = self.env.observe(p)
        start_x = area.x()
        for i, c in enumerate(self.env.hands[0]):
            r = QRect(start_x + i*(CARD_W+16), area.y(), CARD_W, CARD_H)
            is_my_turn = (self.env.turn_player == 0)
            highlight = is_my_turn and (self._pending_human_click == i)
            self._draw_card(qp, r, c, face_up=True, highlight=highlight)

    def _draw_score(self, qp: QPainter, area: QRect):
        qp.setPen(QPen(QColor(240,240,240)))
        font = QFont("Segoe UI", 15); qp.setFont(font)
        qp.drawText(area, Qt.AlignLeft|Qt.AlignTop, f"TOP   : {self.env.points[1]}")
        qp.drawText(area.adjusted(0,30,0,0), Qt.AlignLeft|Qt.AlignTop, f"BOTTOM: {self.env.points[0]}")
        # turn hint
        who = "BOTTOM" if self.env.turn_player==0 else "TOP"
        qp.drawText(area.adjusted(0,60,0,0), Qt.AlignLeft|Qt.AlignTop, f"Turno: {who}")

    # mouse -> click su carte bottom
    def mousePressEvent(self, ev):
        if not self.human_bottom: return
        if self.env.done: return
        if self.env.turn_player != 0: return
        # mappa click agli slot delle 3 carte
        W = self.width()
        area = QRect(W//2 - (3*CARD_W+2*16)//2, self.height()- CARD_H - 60, 3*CARD_W + 2*16, CARD_H)
        start_x = area.x()
        for i in range(len(self.env.hands[0])):
            r = QRect(start_x + i*(CARD_W+16), area.y(), CARD_W, CARD_H)
            if r.contains(ev.position().toPoint()):
                self._pending_human_click = i
                # se non è in autoplay, esegui subito lo step
                if not self.autoplay:
                    self._once()
                else:
                    self.update()
                break

def run():
    app = QApplication(sys.argv)
    w = GameWidget()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()
