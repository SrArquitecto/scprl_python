import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    Gestiona el curriculum automáticamente:
    - Sube de fase cuando el rendimiento supera el umbral
    - Aplica transfer learning al entrar en fase de combate
    - Congela/descongela el trunk en transiciones de fase
    """
    # Umbrales de reward para subir de fase
    UMBRAL_FASE_1 = 500    # navegación básica dominada
    UMBRAL_FASE_2 = 800    # keycards dominadas
    UMBRAL_FASE_3 = 1200   # combate básico dominado
 
    # Steps con trunk congelado al inicio de una nueva fase
    # Permite que la cabeza nueva aprenda sin distorsionar el trunk
    FREEZE_STEPS = 50_000
 
    def __init__(self, curriculum_manager, check_freq=50_000):
        super().__init__()
        self.curriculum  = curriculum_manager
        self.check_freq  = check_freq
        self._fase_actual       = 1
        self._freeze_hasta_step = 0   # step en que se descongela el trunk
 
    def _extractor(self):
        """Acceso rápido al feature extractor del modelo."""
        return self.model.policy.features_extractor
 
    def _on_step(self) -> bool:
        # ── Descongelar trunk si toca ──────────────────────────────────────
        if self._freeze_hasta_step > 0 and self.num_timesteps >= self._freeze_hasta_step:
            self._extractor().freeze_trunk(False)
            self._freeze_hasta_step = 0
 
        # ── Comprobar si subir de fase ─────────────────────────────────────
        if self.n_calls % self.check_freq != 0:
            return True
        if len(self.model.ep_info_buffer) == 0:
            return True
 
        mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        print(f"[Curriculum] Fase={self._fase_actual} mean_reward={mean_reward:.1f}")
 
        if self._fase_actual == 1 and mean_reward > self.UMBRAL_FASE_1:
            self.curriculum.fase_2_keycards()
            self._fase_actual = 2
 
        elif self._fase_actual == 2 and mean_reward > self.UMBRAL_FASE_2:
            self.curriculum.fase_3_enemigos()
            self._fase_actual = 3
 
        elif self._fase_actual == 3 and mean_reward > self.UMBRAL_FASE_3:
            # ── Transición a combate: transfer learning ────────────────────
            # head_combat hereda los pesos de head_surv para no partir de cero
            self._extractor().transfer_combat_from_surv()
            # Congelar trunk FREEZE_STEPS para que head_combat se estabilice
            self._extractor().freeze_trunk(True)
            self._freeze_hasta_step = self.num_timesteps + self.FREEZE_STEPS
            self.curriculum.fase_4_scps()
            self._fase_actual = 4
            print(f"🎯 Fase 4 (combate): trunk congelado hasta step {self._freeze_hasta_step:,}")
 
        return True
