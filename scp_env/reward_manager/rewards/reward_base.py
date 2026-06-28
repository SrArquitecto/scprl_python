from abc import ABC, abstractmethod

class RewardBase(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reinicia variables de estado internas de esta fase al inicio de cada episodio."""
        pass

    @abstractmethod
    def calcular_reward(self, s: dict, action: int, info_entorno: dict) -> float:
        """
        Calcula la recompensa basándose en el estado JSON 's', la acción
        y cualquier otra variable de control del entorno.
        """
        pass