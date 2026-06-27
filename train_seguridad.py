import os
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# 🛠️ AÑADIDO: Importamos VecNormalize junto a los otros wrappers de vectores
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize
from scp_env import SCPClassDEnv

os.makedirs("models/best", exist_ok=True)
os.makedirs("logs/tb",     exist_ok=True)

# ── Extractor CNN + MLP combinado ─────────────────────────────────────────

class SCPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Procesa imagen con CNN (NatureCNN simplificada)
    y vector numérico con MLP.
    Los concatena en un vector de features final.
    """
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256 + 128)

        img_shape = observation_space['image'].shape  
        vec_dim   = observation_space['vector'].shape[0]

        n_channels = img_shape[0]
        height = img_shape[1]
        width = img_shape[2]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcular tamaño de salida CNN automáticamente
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, n_channels, height, width)
            print("img_shape de entrada a la CNN =", img_shape)
            cnn_out = self.cnn(dummy_tensor).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_out, 256),
            nn.ReLU()
        )

        # MLP para vector numérico
        self.mlp = nn.Sequential(
            nn.Linear(vec_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

    def forward(self, obs):
        img = obs['image'].float() / 255.0
        img_feat = self.cnn_linear(self.cnn(img))

        # Vector
        vec_feat = self.mlp(obs['vector'].float())

        return torch.cat([img_feat, vec_feat], dim=1)

# ── Callbacks ─────────────────────────────────────────────────────────────

class CheckpointCompletoCallback(BaseCallback):
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


class GuardarMejorModeloEntrenamiento(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = float('-inf')
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                recompensas = [info['r'] for info in self.model.ep_info_buffer]
                mean_reward = sum(recompensas) / len(recompensas)
                
                if self.verbose > 0:
                    print(self.n_calls, f"-> Revisando rendimiento. Media actual: {mean_reward:.2f} | Mejor histórica: {self.best_mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    
                    # Guardar el modelo PPO (.zip)
                    self.model.save(os.path.join(self.save_path, "best_model"))
                    
                    # 🛠️ AÑADIDO: Guardar estadísticas de normalización del mejor modelo (.pkl)
                    vec_normalize_env = self.model.get_vec_normalize_env()
                    if vec_normalize_env is not None:
                        vec_normalize_env.save(os.path.join(self.save_path, "best_vec_normalize.pkl"))
                        
                    print(f"🔥 ¡Nuevo récord! Modelo y Stats guardados en {self.save_path} con {mean_reward:.2f} de recompensa media.")
                        
        return True

class CurriculumCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.phase = 1

    def _on_step(self):
        n = self.num_timesteps
        if n > 700_000 and self.phase < 3:
            self.phase = 3
            print("\n🎓 Fase 3 — Usar ascensores y escapar")
        elif n > 300_000 and self.phase < 2:
            self.phase = 2
            print("\n🎓 Fase 2 — Buscar y recoger keycards")
        return True

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                self.ep_rewards.append(r)
                avg = sum(self.ep_rewards[-20:]) / min(len(self.ep_rewards), 20)
                print(f"   Ep {len(self.ep_rewards):4d} | "
                      f"pasos: {l:5d} | "
                      f"reward: {r:+8.1f} | "
                      f"media×20: {avg:+8.1f}")
        return True

# ── Entornos Uniformes e Inyección de Normalización ───────────────────────

print("Conectando entornos...")

# Preparar entorno de entrenamiento
raw_train = Monitor(SCPClassDEnv(), filename="logs/train")
env = DummyVecEnv([lambda: raw_train])

"""
ruta_stats = "models/best_scp/best_v_3.pkl" 
env = VecNormalize.load(ruta_stats, env)
"""


# ⚠️ REGLA DE ORO PARA CONTINUAR ENTRENANDO:
# A diferencia del modo juego/render, aquí DEBEN estar en True.
# Si no lo haces, el agente dejará de actualizar la escala de los nuevos premios.
env.training = True     
env.norm_reward = True

# 🛠️ AÑADIDO: Envolvemos el entorno con VecNormalize
# norm_obs=False porque tu extractor ya escala la imagen de forma fija dividiendo por 255.0
# norm_reward=True se encarga de estabilizar matemáticamente tus hitos de +50.0 y penalizaciones


env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=10.0)


env = VecTransposeImage(env) 

# ── Modelo PPO con extractor personalizado ─────────────────────────────────

policy_kwargs = dict(
    features_extractor_class  = SCPFeaturesExtractor,
    features_extractor_kwargs = {},
    net_arch                  = [256, 128],   
)

model = PPO(
    "MultiInputPolicy", 
    env, 
    learning_rate   = 3e-4,
    n_steps         = 4096,       
    batch_size      = 128,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.05,
    policy_kwargs   = policy_kwargs,
    device          = 'cuda',
    verbose         = 1,
    tensorboard_log = "logs/tb/"
)

# Parámetros modificados listos por si necesitas un .load futuro
parametros_modificados = {
    "learning_rate": 3e-4,
    "n_steps": 2048,          # 🧠 Reducido: Actualiza 4 veces por episodio completo. Aprendizaje más rápido.
    "batch_size": 128,        # Múltiplo limpio de n_steps (1024 / 128 = 8 batches por época)
    "n_epochs": 10,
    "gamma": 0.997,           # 🔭 Aumentado: Amplía el horizonte a ~330 pasos. Clave para conectar la celda con la salida lejana.
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,         # 🎯 Reducido: Conserva su habilidad de "apuntar a puertas" sin volverse loco, pero permitiendo explorar el pasillo.
    "device": "cuda",
    "verbose": 1,
    "tensorboard_log": "logs/tb/"
}

"""
ruta_modelo = "models/best_scp/best_c_3"

model = PPO.load(
    ruta_modelo, 
    env=env, 
    custom_objects=parametros_modificados  # 👈 Aquí se inyectan los cambios externos
)
"""
"""
model.policy.optimizer = torch.optim.Adam(
    model.policy.parameters(), lr=3e-4
)
"""
print(f"Dispositivo: {model.device}")
print(f"Parámetros del modelo: {sum(p.numel() for p in model.policy.parameters()):,}")
print("\n🚀 Iniciando entrenamiento con VecNormalize activo...")
print("   TensorBoard: tensorboard --logdir logs/tb/\n")

model.learn(
    total_timesteps = 1_000_000,
    callback = [
        #CurriculumCallback(),
        EpisodeLoggerCallback(),
        CheckpointCompletoCallback(
            save_freq   = 8_192,
            save_path   = "models/",
            name_prefix = "classd"
        ),
        GuardarMejorModeloEntrenamiento(
            check_freq = 32_768,          
            save_path  = "models/best_scp/", 
            verbose    = 1
        )
    ],
    progress_bar = True
)



# Guardar modelo final y sus estadísticas correspondientes
model.save("models/classd_final")
# 🛠️ AÑADIDO: Extracción y guardado seguro del archivo de normalización final
if model.get_vec_normalize_env() is not None:
    model.get_vec_normalize_env().save("models/classd_final_vec_normalize.pkl")

print("\n✅ Entrenamiento completado. Modelo y estadísticas de recompensa exportados con éxito.")
