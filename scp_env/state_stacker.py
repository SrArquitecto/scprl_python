import numpy as np

class StateStacker:
    def __init__(self, history_len: int, vector_dim : int):
        self.history_len = history_len
        self.vector_dim = vector_dim
        self.stack = np.zeros((history_len, vector_dim), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset al primer step: todos los frames son el frame actual."""
        self.stack = np.zeros((self.history_len, self.vector_dim), dtype=np.float32)
        return self.stack.flatten()

    def update(self, new_frame: np.ndarray) -> np.ndarray:
        """Desplaza los frames hacia atrás y añade el nuevo al final."""
        self.stack = np.roll(self.stack, -1, axis=0)
        self.stack[-1] = new_frame
        return self.stack.flatten()

    def seed(self, frame: np.ndarray) -> np.ndarray:
        """En el reset, llena el stack con el mismo frame N veces."""
        self.stack = np.zeros((self.history_len, self.vector_dim), dtype=np.float32)
        for _ in range(self.history_len):
            self.stack = np.roll(self.stack, -1, axis=0)
            self.stack[-1] = frame
        return self.stack.flatten()