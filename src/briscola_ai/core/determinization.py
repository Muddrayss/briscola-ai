from __future__ import annotations
import random
from typing import List, Set
from .cards import Card, Suit, all_deck
from .env import BriscolaEnv, Observation

def _card_set(cards: List[Card]) -> Set[Card]:
    return set(cards)

def determinize_for_player(env: BriscolaEnv, obs: Observation, rng: random.Random) -> BriscolaEnv:
    """
    Crea un clone dell'ambiente in cui:
      - la tua mano è quella di obs
      - le carte avversarie e l'ordine del mazzo rimanente sono campionati
        dalle carte non viste (played + tavolo + briscola scoperta escluse)
    Non usa le carte nascoste reali: solo info pubbliche.
    """
    sim = env.clone()

    me = obs.player
    opp = 1 - me

    # Rimuovi tutte le info note dall'insieme delle 40
    known = set(obs.hand)
    known.update(obs.played)
    if obs.lead_card is not None:
        known.add(obs.lead_card)
    if sim.follow_card is not None:
        known.add(sim.follow_card)
    if sim.trump_card is not None:
        known.add(sim.trump_card)  # carta briscola è visibile
    unseen = [c for c in all_deck() if c not in known]

    # Dimensioni note
    opp_needed = obs.opponent_count
    deck_needed = obs.deck_count

    rng.shuffle(unseen)
    opp_hand = unseen[:opp_needed]
    deck_rest = unseen[opp_needed:opp_needed + deck_needed]

    # Imposta mani e mazzo nel clone
    sim.hands[me] = list(obs.hand)
    sim.hands[opp] = list(opp_hand)
    sim.deck = list(deck_rest)

    # Nota: leader/turn_player/points/lead_card/follow_card restano invariati dal clone
    return sim
