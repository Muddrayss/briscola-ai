from __future__ import annotations
from math import exp
from typing import List, Dict, Tuple
from .cards import Card, Suit
from .env import Observation
from .cards import trick_winner

CardId = Tuple[str, str]

def card_id(c: Card) -> CardId:
    return (c.suit.value, c.rank)

def _softmax(xs: List[float], temp: float = 1.0) -> List[float]:
    if temp <= 0: temp = 1e-6
    m = max(xs)
    exps = [exp((x - m) / temp) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def _rank_value(rank: str) -> int:
    # utilità grezza per conservare A/3 e figure
    order = {"A": 9, "3": 8, "K": 7, "C": 6, "F": 5, "7": 4, "6": 3, "5": 2, "4": 1, "2": 0}
    return order[rank]

def move_priors_for_obs(obs: Observation, temp: float = 1.0) -> Dict[CardId, float]:
    """
    Heuristiche leggere:
      - Se SEGUO: grande boost alle mosse che vincono il trick, scalato dai punti del trick.
        Penalizza sprecare briscola se il trick è 'leggero'.
      - Se APRO: preferisci scartare basso non briscola; evita di bruciare A/3 inutilmente.
    Restituisce una distribuzione (somma=1) su card_id in mano.
    """
    priors: Dict[CardId, float] = {}
    br = obs.briscola

    # punti nel trick se esiste una lead
    trick_pts = obs.trick_points_so_far

    if obs.lead_card is not None:
        lead = obs.lead_card
        for c in obs.hand:
            cid = card_id(c)
            wins = (trick_winner(lead, c, br) == 1)
            s = 0.0
            if wins:
                # più punti in palio -> più incentivo a vincere
                s += 2.0 + 0.15 * (trick_pts + c.points())
                # se vinci SENZA briscola, ancora meglio (risparmi trump)
                if c.suit != br:
                    s += 0.8
                else:
                    # se è briscola ma il trick vale poco, punisci leggermente
                    if (trick_pts + c.points()) < 6:
                        s -= 0.4
            else:
                # non vinci: preferisci buttare la più “inutile”
                s += 0.2
                # se butti briscola e NON vinci, penalità forte
                if c.suit == br:
                    s -= 1.0
            # proteggi A/3: piccola penalità allo spreco se non vinci
            if not wins and _rank_value(c.rank) >= 8:
                s -= 0.5
            priors[cid] = s
    else:
        # Apro il trick
        for c in obs.hand:
            cid = card_id(c)
            s = 0.0
            # non briscola bassa: buono per scarico sicuro
            if c.suit != br and c.points() == 0 and _rank_value(c.rank) <= 4:
                s += 1.2
            # non briscola alta con punti: rischiosa se molte briscole in giro
            if c.suit != br and c.points() > 0:
                s -= 0.3
            # briscola: evita di aprire a briscola a meno di necessità
            if c.suit == br:
                s -= 0.6
                # se è una briscola “bassa”, la penalità è minore
                if _rank_value(c.rank) <= 4:
                    s += 0.2
            # conserva A o 3 in apertura se possibile
            if _rank_value(c.rank) >= 8:
                s -= 0.4
            priors[cid] = s

    # normalizza
    scores = list(priors.values())
    probs = _softmax(scores, temp=temp)
    for (k, p) in zip(list(priors.keys()), probs):
        priors[k] = p
    return priors
