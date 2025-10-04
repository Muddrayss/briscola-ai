from __future__ import annotations
from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, env, obs) -> int:
        """Ritorna l'indice della carta da giocare nella mano corrente."""
        ...
