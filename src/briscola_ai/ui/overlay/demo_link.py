from __future__ import annotations
import sys, time
from PySide6.QtWidgets import QApplication
from .advisor_overlay import AdvisorOverlay
from ...core.env import BriscolaEnv
from ...agents.mcts import ISMCTSAgent

def main():
    app = QApplication(sys.argv)
    cal = sys.argv[1] if len(sys.argv)>1 else "vision/calibrations/sisal_1080p.json"
    ov = AdvisorOverlay(cal); ov.show()

    # Simula: prendi un obs reale (mano bottom) e lascia che MCTS scelga una carta
    env = BriscolaEnv(seed=123)
    # avanza fino al turno di bottom per sicurezza
    if env.turn_player != 0:
        obs = env.observe(0)
        # forza un passo minimo (fa giocare top random)
        from ...agents.rule_based import RandomAgent
        ridx = RandomAgent().act(env, env.observe(1))
        env.step(1, ridx)

    obs = env.observe(0)
    mcts = ISMCTSAgent(iterations=400, c=1.0, discount=0.7, seed=7)
    idx = mcts.act(env, obs)

    # mappa idx -> ROI
    key = f"bottom_slot_{idx+1}"
    ov.set_suggestion(key)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
