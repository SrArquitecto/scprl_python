import os
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import SubprocVecEnv
import socket
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# 🛠️ AÑADIDO: Importamos VecNormalize junto a los otros wrappers de vectores
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize


os.makedirs("models/best", exist_ok=True)
os.makedirs("logs/tb",     exist_ok=True)

from scp_env.scp_env import Scp_env
from base.callbacks.best_model import BestModelCallback
from base.callbacks.checkpoint_model import CheckpointCallback
from base.callbacks.curriculum import CurriculumCallback
from base.callbacks.map_regen import MapRegenCallback
from base.callbacks.episode_logger import EpisodeLoggerCallback

from base.extractor.features_extractor import SCPFeaturesExtractor

from base.actions.mask_actions import (ACTION_MASKS, get_action_mask)

from base.manager.curriculum_manager import CurriculumManager

# ── Extractor CNN + MLP combinado ─────────────────────────────────────────
import socket
 

# ── Entornos Uniformes e Inyección de Normalización ───────────────────────


def negociar_num_agentes(n, port=7900):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    
    # Enviar
    s.sendall(f"HELLO_{n}\n".encode('utf-8'))
    
    # Leer hasta que el servidor nos cierre la conexión
    resp = b""
    while True:
        data = s.recv(1024)
        if not data: # El servidor cerró la conexión
            break
        resp += data
        
    s.close()
    
    decoded_resp = resp.decode('utf-8-sig').strip()
    print(f"DEBUG: Respuesta final recibida: '{decoded_resp}'")
    
    if "CONNECTED" in decoded_resp:
        return
    else:
        raise RuntimeError(f"El servidor respondió '{decoded_resp}'")

def make_env(i):
    """Función factoría para crear cada entorno de forma aislada."""
    def _init():
        def mask_fn(env) -> np.ndarray:
            role = getattr(env, "current_role", "ClassD")
            return get_action_mask(role)
        
        # IMPORTANTE: La conexión al socket debe ocurrir AQUÍ, dentro del proceso hijo
        env = Scp_env(host='localhost', port=7900, agent_id=i, role="classd")
        
        env = ActionMasker(env, mask_fn)
        # Cada proceso registra sus propias métricas para TensorBoard
        env = Monitor(env, filename=f"logs/train_proc_{i}")
        return env
    return _init

if __name__ == "__main__":
    n = 4
    curriculum = CurriculumManager(host="localhost", port=7900)
    curriculum.fase_1_basico()  # empezar en fase 1
    callback = CurriculumCallback(curriculum, check_freq=50_000)

    # 🌟 CRÍTICO: Detectar si soy el proceso principal
    # SubprocVecEnv suele dejar una variable de entorno o simplemente 
    # podemos usar el hecho de que solo el padre debe hacer esto.
    # Si detectas que esto sigue fallando, usa una variable de control:
    
    if os.environ.get("IS_WORKER") is None:
        print("Soy el proceso principal: negociando...")
        negociar_num_agentes(n)
        
    # Crear los entornos
    env_fns = [make_env(i) for i in range(n)]
    env = SubprocVecEnv(env_fns)
    # IMPORTANTE: Al lanzar los workers, pásales una variable de entorno
    # para que ellos sepan que no deben volver a negociar.
    os.environ["IS_WORKER"] = "1"


    # 3. Lanzamos SubprocVecEnv (Esto crea 4 procesos independientes)


    """
    ruta_stats = "models/classd_57344_stats.pkl" 
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


    #env = VecTransposeImage(env) 

    # ── Modelo PPO con extractor personalizado ─────────────────────────────────

    policy_kwargs = dict(
        features_extractor_class=SCPFeaturesExtractor,
        features_extractor_kwargs=dict(),
        # net_arch separado: actor (pi) y crítico (vf) con features independientes.
        # SB3 divide features_dim=512 en dos mitades de 256 automáticamente
        # cuando pi y vf tienen net_arch distintos.
        net_arch=dict(
            pi=[256, 256],  # actor: 2 capas ocultas de 256
            vf=[256, 256],  # crítico: 2 capas ocultas de 256 (con contexto social)
        ),
    )
    
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        learning_rate   = 2e-4,       # Mantener estable al inicio
        n_steps         = 1024,       # Reducido de 2048 -> Total Buffer = 4 * 1024 = 4096 muestras
        batch_size      = 256,        # Incrementado de 128 -> Estimación de gradiente más estable
        n_epochs        = 4,          # Reducido de 10 -> Evita destruir la política en pasos tempranos
        gamma           = 0.99,       
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.02,       # Reducido un poco de 0.05 para que no sea caos puro, pero mantenga exploración
        policy_kwargs   = policy_kwargs,
        device          = 'cuda',
        verbose         = 1,
        tensorboard_log = "logs/tb/"
    )
    
    # Parámetros modificados listos por si necesitas un .load futuro
    parametros_modificados_c2 = {
        "learning_rate": 2e-4,     # ⚠️ Reducido: De 3e-4 a 1e-4. Menos agresivo.
        "n_steps": 1024,          # 🧠 Reducido: Menos pasos por iteración, el agente aprende más frecuente.
        "batch_size": 256,         # ⚠️ Reducido: Mantiene el ratio, pero hace el gradiente más robusto.
        "n_epochs": 4,            # 📉 CRÍTICO: Baja de 10 a 4. Esto bajará el approx_kl inmediatamente.
        "gamma": 0.99,            # 🔭 Reducido ligeramente para ganar estabilidad.
        "gae_lambda": 0.95,
        "clip_range": 0.2,        # 🛡️ Más conservador para evitar cambios bruscos.
        "ent_coef": 0.02,        # Sutilmente más bajo para evitar dispersión.
        "device": "cuda",
        "verbose": 1,
        "tensorboard_log": "logs/tb/"
    }

    """
    ruta_modelo = "models/classd_57344_steps"

    model = PPO.load(
        ruta_modelo, 
        env=env, 
        custom_objects=parametros_modificados_c2  # 👈 Aquí se inyectan los cambios externos
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
            callback,
            MapRegenCallback(regen_every=51_200),
            EpisodeLoggerCallback(),
            CheckpointCallback(
                save_freq   = 8_192,
                save_path   = "models/",
                name_prefix = "classd"
            ),
            BestModelCallback(
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
