import os
from stable_baselines3.common.callbacks import BaseCallback

class BestModelCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = float('-inf')
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 🌟 CORRECCIÓN: Comprobamos que exista el buffer antes de pasarlo a len()
            if self.model.ep_info_buffer is not None and len(self.model.ep_info_buffer) > 0:
                recompensas = [info['r'] for info in self.model.ep_info_buffer]
                mean_reward = sum(recompensas) / len(recompensas)
                
                if self.verbose > 0:
                    print(self.n_calls, f"-> Revisando rendimiento. Media actual: {mean_reward:.2f} | Mejor histórica: {self.best_mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    
                    # Guardar el modelo PPO (.zip)
                    self.model.save(os.path.join(self.save_path, "best_model"))
                    
                    # 🛠️ Guardar estadísticas de normalización del mejor modelo (.pkl)
                    vec_normalize_env = self.model.get_vec_normalize_env()
                    if vec_normalize_env is not None:
                        vec_normalize_env.save(os.path.join(self.save_path, "best_vec_normalize.pkl"))
                        
                    print(f"🔥 ¡Nuevo récord! Modelo y Stats guardados en {self.save_path} con {mean_reward:.2f} de recompensa media.")
                        
        return True