from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .base import Agent
from .rule_based import BestChoiceFirstAgent
from ..core.cards import Card
from ..core.env import BriscolaEnv, Observation
from ..core.determinization import determinize_for_player
from ..core.heuristics import move_priors_for_obs, card_id as cid

CardId = Tuple[str,str]

@dataclass
class NodeStats:
    N: Dict[CardId, int] = field(default_factory=dict)     # visite per-azione
    W: Dict[CardId, float] = field(default_factory=dict)   # somma ritorni per-azione
    P: Dict[CardId, float] = field(default_factory=dict)   # prior euristiche per-azione (somma=1, opzionale)
    actions: List[CardId] = field(default_factory=list)

class ISMCTSAgent(Agent):
    """
    ISMCTS con determinizzazioni + PUCT (prior euristiche):
      UCB = Q + c_puct * P[a] * sqrt(Ntot)/(1+N[a])
    """
    def __init__(self, iterations: int = 800, c: float = 1.0, discount: float = 0.7, seed: int = 0, prior_temp: float = 1.0):
        self.iterations = iterations
        self.c = c
        self.discount = discount
        self.rng = random.Random(seed)
        self.rollout = BestChoiceFirstAgent()
        self.tree: Dict[Tuple[int, Tuple], NodeStats] = {}
        self.prior_temp = prior_temp
        self.seed = seed

    # ----------------- helpers -----------------
    def _obs_key(self, env: BriscolaEnv, player: int) -> Tuple:
        obs = env.observe(player)
        hand_ids = tuple(sorted(cid(c) for c in obs.hand))
        lead = cid(obs.lead_card) if obs.lead_card else None
        return (player, env.briscola.value, lead, len(obs.played), obs.deck_count, hand_ids)

    def _legal_action_ids(self, env: BriscolaEnv, player: int) -> List[CardId]:
        obs = env.observe(player)
        # Se la mano è vuota, restituiamo [] (verrà gestito a monte)
        return [cid(c) for c in obs.hand]

    def _choose_puct(self, node: NodeStats):
        if not node.actions:
            return None
        total_N = sum(node.N.values()) if node.N else 0
        best_a, best_val = None, -1e9
        for a in node.actions:
            n = node.N.get(a, 0)
            w = node.W.get(a, 0.0)
            q = (w / n) if n > 0 else 0.0
            p = node.P.get(a, 1.0 / max(1, len(node.actions)))
            u = self.c * p * math.sqrt(total_N + 1) / (1 + n)
            val = q + u
            if val > best_val:
                best_val, best_a = val, a
        return best_a

    def _select_expand(self, env: BriscolaEnv, path: List[Tuple[Tuple,int,CardId]]):
        while not env.done:
            p = env.turn_player
            key = self._obs_key(env, p)

            if key not in self.tree:
                actions = self._legal_action_ids(env, p)
                if not actions:                    # <-- guardia fondamentale
                    return env
                node = NodeStats(actions=actions)
                node.P = move_priors_for_obs(env.observe(p), temp=self.prior_temp)
                self.tree[key] = node

                a = max(actions, key=lambda x: node.P.get(x, 0.0))  # ora è safe (actions non vuoto)
                obs = env.observe(p)
                try:
                    idx = next(i for i, c in enumerate(obs.hand) if cid(c) == a)
                except StopIteration:
                    return env
                env.step(p, idx)
                path.append((key, p, a))
                return env

            node = self.tree[key]
            a = self._choose_puct(node)
            if a is None:                           # <-- niente azioni da scegliere
                return env
            obs = env.observe(p)
            try:
                idx = next(i for i, c in enumerate(obs.hand) if cid(c) == a)
            except StopIteration:
                return env
            env.step(p, idx)
            path.append((key, p, a))
        return env


    def _rollout(self, env: BriscolaEnv):
        while not env.done:
            p = env.turn_player
            obs = env.observe(p)
            if not obs.hand:            # <-- mano vuota = stop
                break
            move = self.rollout.act(env, obs)
            if not isinstance(move, int) or move < 0 or move >= len(obs.hand):
                move = 0                # <-- clamp
            env.step(p, move)

            
    def _result(self, env: BriscolaEnv, root_player: int) -> float:
        me, opp = root_player, 1 - root_player
        diff = env.points[me] - env.points[opp]
        # normalizza tra -1 e 1 (120 = punti totali di Briscola)
        return max(-1.0, min(1.0, diff / 120.0))
    
    def _backprop(self, path: List[Tuple[Tuple, int, CardId]], root_player: int, value: float) -> None:
        """
        Back-propagate rollout value along the visited path.

        - `value` è dal punto di vista di `root_player` ([-1, 1]).
        - Se il giocatore al nodo `p` != `root_player`, invertiamo il segno.
        - Applichiamo discount per profondità: self.discount ** depth.
        - Aggiornamenti:
            N[a] += 1
            W[a] += discounted_signed_value
        """
        if not path:
            return

        for depth, (key, p, a) in enumerate(path):
            node = self.tree.get(key)
            if node is None:
                continue
            signed_v = value if p == root_player else -value
            discounted = signed_v * (self.discount ** depth)
            node.N[a] = node.N.get(a, 0) + 1
            node.W[a] = node.W.get(a, 0.0) + discounted

    def act(self, env: BriscolaEnv, obs: Observation) -> int:
        self.tree.clear()
        root_player = obs.player

        for _ in range(self.iterations):
            sim = determinize_for_player(env, obs, self.rng)
            path: List[Tuple[Tuple,int,CardId]] = []
            sim = self._select_expand(sim, path)
            self._rollout(sim)
            value = self._result(sim, root_player)
            self._backprop(path, root_player, value)

        root_key = self._obs_key(env, root_player)
        node = self.tree.get(root_key)
        if not node or not node.actions:                      # <-- fallback robusto
            return BestChoiceFirstAgent().act(env, obs)

        best_a = max(node.actions, key=lambda a: node.N.get(a, 0))
        idx = next(i for i, c in enumerate(obs.hand) if cid(c) == best_a)
        return idx