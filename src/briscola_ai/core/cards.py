from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import random

class Suit(str, Enum):
    DENARI = "denari"
    COPPE = "coppe"
    BASTONI = "bastoni"
    SPADE = "spade"

# Ordine di presa (stesso seme): A > 3 > K > Cav > Fan > 7 > 6 > 5 > 4 > 2
RANK_ORDER = ["A", "3", "K", "C", "F", "7", "6", "5", "4", "2"]
RANK_STRENGTH = {r: i for i, r in enumerate(reversed(RANK_ORDER))}  # più alto = più forte? no: usiamo confronto dedicato
POINTS = {"A": 11, "3": 10, "K": 4, "C": 3, "F": 2, "7": 0, "6": 0, "5": 0, "4": 0, "2": 0}

@dataclass(frozen=True, order=False)
class Card:
    suit: Suit
    rank: str  # one of RANK_ORDER (A,3,K,C,F,7,6,5,4,2)

    def points(self) -> int:
        return POINTS[self.rank]

    def __str__(self) -> str:
        symbols = {Suit.DENARI:"◆", Suit.COPPE:"♥", Suit.BASTONI:"♣", Suit.SPADE:"♠"}
        return f"{self.rank}{symbols[self.suit]}"

def all_deck() -> List[Card]:
    return [Card(s, r) for s in Suit for r in RANK_ORDER]

def shuffled_deck(rng: random.Random) -> List[Card]:
    deck = all_deck()
    rng.shuffle(deck)
    return deck

def stronger_in_suit(a: Card, b: Card) -> Card:
    """Ritorna la più forte tra a e b ASSUMENDO stesso seme."""
    if a.suit != b.suit:
        raise ValueError("Confronto per seme: i semi devono coincidere.")
    # ordine A > 3 > K > C > F > 7 > 6 > 5 > 4 > 2
    order_index = {r: idx for idx, r in enumerate(RANK_ORDER)}
    return a if order_index[a.rank] < order_index[b.rank] else b

def trick_winner(lead: Card, follow: Card, briscola: Suit) -> int:
    """
    Ritorna 0 se vince la carta di chi ha aperto (lead), 1 se vince chi ha risposto (follow).
    Regole standard: briscola batte tutto; altrimenti vince la più alta del seme di uscita.
    """
    lead_is_trump = (lead.suit == briscola)
    foll_is_trump = (follow.suit == briscola)

    if lead_is_trump and foll_is_trump:
        return 0 if stronger_in_suit(lead, follow) is lead else 1
    if foll_is_trump and not lead_is_trump:
        return 1
    if (follow.suit != lead.suit) and not foll_is_trump:
        return 0
    # stesso seme dell'uscita (non briscola) -> più alta dell'uscita
    return 0 if stronger_in_suit(lead, follow) is lead else 1

def trick_points(cards: Tuple[Card, Card]) -> int:
    return cards[0].points() + cards[1].points()
