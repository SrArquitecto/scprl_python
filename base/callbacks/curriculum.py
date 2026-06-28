import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_manager, check_freq=50_000):
        super().__init__()
        self.curriculum = curriculum_manager
        self.check_freq = check_freq
 
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.model.ep_info_buffer is not None and len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                fase_actual = self.curriculum.get_config().get("Fase", 1)
 
                # Subir de fase si el rendimiento es suficientemente bueno
                if fase_actual == 1 and mean_reward > 500:
                    self.curriculum.fase_2_keycards()
                elif fase_actual == 2 and mean_reward > 800:
                    self.curriculum.fase_3_enemigos()
                elif fase_actual == 3 and mean_reward > 1200:
                    self.curriculum.fase_4_scps()
 
                print(f"[Curriculum] Fase={fase_actual} mean_reward={mean_reward:.1f}")
        return True