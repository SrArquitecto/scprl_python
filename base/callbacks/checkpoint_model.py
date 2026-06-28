import os
from stable_baselines3.common.callbacks import BaseCallback

class CheckpointCallback(BaseCallback):
    """
    Reemplaza al CheckpointCallback nativo. 
    Guarda tanto el modelo (.zip) como sus estadísticas de normalización (.pkl)
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "classd", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Definir nombres con el número de pasos exacto
            model_filename = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps")
            stats_filename = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_stats.pkl")
            
            # 1. Guardar el modelo PPO
            self.model.save(model_filename)
            
            # 2. Guardar las estadísticas de normalización sincronizadas
            vec_normalize_env = self.model.get_vec_normalize_env()
            if vec_normalize_env is not None:
                vec_normalize_env.save(stats_filename)
                
            if self.verbose > 0:
                print(f"\n📦 [CHECKPOINT] Guardado periódico en paso {self.n_calls}:")
                print(f"   -> {model_filename}.zip")
                print(f"   -> {stats_filename}")
                
        return True