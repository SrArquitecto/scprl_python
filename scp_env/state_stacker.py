import numpy as np

class StateStacker:
    def __init__(self, state_dim, history_len=3):
        self.history_len = history_len
        # Pre-asignamos memoria (mucho más rápido que ir haciendo append)
        self.stack = np.zeros((history_len, state_dim), dtype=np.float32)

    def update(self, new_state):
        # Desplazamos los frames hacia atrás (el más viejo sale)
        self.stack = np.roll(self.stack, -1, axis=0)
        # Añadimos el nuevo al final
        self.stack[-1] = new_state
        return self.stack.flatten() # Devuelve el vector plano para la red