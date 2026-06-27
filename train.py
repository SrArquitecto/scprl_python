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
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
import socket, json
import orjson
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# 🛠️ AÑADIDO: Importamos VecNormalize junto a los otros wrappers de vectores
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize
from scp_env import SCPClassDEnv

os.makedirs("models/best", exist_ok=True)
os.makedirs("logs/tb",     exist_ok=True)

# ── Extractor CNN + MLP combinado ─────────────────────────────────────────
import socket
import time
 
class CurriculumManager:
    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port
 
    def _enviar_comando(self, comando: str, retries=5) -> str:
        for intento in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self.host, self.port))
                s.sendall(f"{comando}\n".encode())
                buf = b""
                while b"\n" not in buf:
                    buf += s.recv(4096)
                s.close()
                return buf.split(b"\n")[0].decode().strip()
            except Exception as e:
                print(f"⚠️ CurriculumManager intento {intento+1}: {e}")
                time.sleep(1.0)
        return "ERROR"
 
    def set_flag(self, clave: str, valor: bool) -> bool:
        resp = self._enviar_comando(f"CONFIG:{clave}={'true' if valor else 'false'}")
        return resp == "OK"
 
    def set_param(self, clave: str, valor: float) -> bool:
        resp = self._enviar_comando(f"CONFIG:{clave}={valor}")
        return resp == "OK"
 
    def set_fase(self, fase: int) -> bool:
        return self.set_param("FaseActual", fase)
 
    def get_config(self) -> dict:
        import json
        resp = self._enviar_comando("GET_CONFIG")
        try:
            return json.loads(resp)
        except:
            return {}
 
    # ── Fases predefinidas de curriculum ────────────────────────────────
 
    def fase_1_basico(self):
        """Fase 1: Solo navegación, sin enemigos, sin SCPs, puertas abiertas."""
        self.set_param("FaseActual", 1)
        self.set_param("Dificultad", 0.0)
        self.set_flag("ScpsActivos", False)
        self.set_flag("EnemigosActivos", False)
        self.set_flag("PuertasBloqueadas", False)
        self.set_flag("RespawnInfinito", True)
        self.set_param("NumEnemigos", 0)
        self.set_param("NumScps", 0)
        self.set_param("ProbKeycard", 1.0)
        self.set_param("TiempoMaxEpisodio", 300)
        print("✅ Curriculum Fase 1 activa — navegación básica")
 
    def fase_2_keycards(self):
        """Fase 2: Navegar + recoger keycards, sin enemigos."""
        self.set_param("FaseActual", 2)
        self.set_param("Dificultad", 0.2)
        self.set_flag("PuertasBloqueadas", True)  # ahora necesita keycard
        self.set_param("ProbKeycard", 0.7)         # no garantizada
        self.set_param("TiempoMaxEpisodio", 240)
        print("✅ Curriculum Fase 2 activa — keycards necesarias")
 
    def fase_3_enemigos(self):
        """Fase 3: Navegar + keycards + guardias como amenaza."""
        self.set_param("FaseActual", 3)
        self.set_param("Dificultad", 0.5)
        self.set_flag("EnemigosActivos", True)
        self.set_param("NumEnemigos", 2)
        self.set_flag("RespawnInfinito", False)
        self.set_param("TiempoMaxEpisodio", 180)
        print("✅ Curriculum Fase 3 activa — enemigos presentes")
 
    def fase_4_scps(self):
        """Fase 4: Todo activo, con SCPs."""
        self.set_param("FaseActual", 4)
        self.set_param("Dificultad", 0.8)
        self.set_flag("ScpsActivos", True)
        self.set_param("NumScps", 1)
        self.set_param("NumEnemigos", 2)
        self.set_param("TiempoMaxEpisodio", 120)
        print("✅ Curriculum Fase 4 activa — SCPs presentes")
 


class SCPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extractor CTDE aproximado para SB3 + SubprocVecEnv.

    Arquitectura:
      - Rama ACTOR (pi): procesa la obs del agente → 256 features
        Solo ve el frame actual. Decisiones rápidas y locales.

      - Rama CRÍTICO (vf): procesa todos los frames con atención temporal
        + proyecta NearPlayers desde la obs para emular contexto global → 256 features.
        El crítico tiene más capacidad para estimar el valor a largo plazo.

    SB3 usa features_dim=512: los primeros 256 van al actor (pi),
    los últimos 256 van al crítico (vf). Esto se controla con net_arch
    en policy_kwargs (ver abajo en main).

    Por qué funciona sin compartir obs entre procesos:
      - NearPlayers ya está en la obs de cada agente (posición, health, hostilidad
        de los jugadores cercanos). El crítico aprende a inferir el estado global
        a partir de esa información local — igual que IPPO con obs aumentadas.
      - No es MAPPO puro (el crítico no ve el estado global real), pero es
        significativamente mejor que un crítico sin contexto social.
    """
    def __init__(self, observation_space: spaces.Box):
        total_dim = observation_space.shape[0]
        self.num_frames = 5
        self.frame_dim  = total_dim // self.num_frames

        # features_dim=512: 256 para actor + 256 para crítico
        super().__init__(observation_space, features_dim=512)

        # ── Troncal compartido: embedding temporal ──────────────────────────
        # Procesa los 5 frames → embedding rico de 256 dim
        self.spatial_mlp = nn.Sequential(
            nn.Linear(self.frame_dim, 128),
            nn.ReLU(),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 128, self.num_frames)
            conv_out_dim = self.temporal_conv(dummy).shape[1]  # 64*5 = 320

        self.trunk = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # ── Rama ACTOR (pi): 256 → 256, solo obs actual ────────────────────
        self.actor_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # ── Rama CRÍTICO (vf): trunk + contexto social → 256 ───────────────
        # El crítico recibe el embedding del trunk + proyección de NearPlayers
        # NearPlayers ocupa N_PLAYERS * N_PLAYERS_FEAT floats en la obs.
        # Los extraemos directamente del vector de obs del último frame.
        # Ajusta N_PLAYERS y N_PLAYERS_FEAT a tus constantes reales:
        self.N_PLAYERS      = 5    # máx jugadores cercanos
        self.N_PLAYERS_FEAT = 11   # features por jugador
        players_dim = self.N_PLAYERS * self.N_PLAYERS_FEAT  # 55

        # Offset de NearPlayers dentro de un frame (scp_env.py encoder):
        # base(45) + danio(5) + rooms(60) + whiskers(16) +
        # zones(7) + rooms_oh(67) + items(45) + doors(150) + lifts(39) + lockers(45)
        # = 479 → NearPlayers empieza en 479
        self.players_offset = 479

        self.players_encoder = nn.Sequential(
            nn.Linear(players_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(256 + 64, 256),   # trunk_feat + players_feat
            nn.LayerNorm(256),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]

        # ── Troncal compartido ─────────────────────────────────────────────
        x = observations.view(b, self.num_frames, self.frame_dim)  # (B, 5, frame_dim)
        x = self.spatial_mlp(x)                                    # (B, 5, 128)
        x = x.permute(0, 2, 1)                                     # (B, 128, 5)
        x = self.temporal_conv(x)                                   # (B, 320)
        trunk_feat = self.trunk(x)                                  # (B, 256)

        # ── Rama actor ─────────────────────────────────────────────────────
        actor_feat = self.actor_head(trunk_feat)                    # (B, 256)

        # ── Contexto social para el crítico ────────────────────────────────
        # Extraer NearPlayers del último frame (frame -1)
        last_frame = observations.view(b, self.num_frames, self.frame_dim)[:, -1, :]
        players_raw = last_frame[
            :, self.players_offset : self.players_offset + self.N_PLAYERS * self.N_PLAYERS_FEAT
        ]                                                           # (B, 55)
        players_feat = self.players_encoder(players_raw)            # (B, 64)

        # ── Rama crítico ────────────────────────────────────────────────────
        critic_feat = self.critic_head(
            torch.cat([trunk_feat, players_feat], dim=1)            # (B, 320)
        )                                                           # (B, 256)

        # Concatenar: SB3 usará [:256] para pi y [256:] para vf
        return torch.cat([actor_feat, critic_feat], dim=1)          # (B, 512)
# ── Callbacks ─────────────────────────────────────────────────────────────

class MapRegenCallback(BaseCallback):
    REGEN_FLAG = "/tmp/scp_regen_flag"

    def __init__(self, regen_every=1_000, control_port=7900, verbose=0):
        super().__init__(verbose)
        self.regen_every = regen_every
        self.control_port = control_port
        self._last_regen = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_regen >= self.regen_every:
            self._last_regen = self.num_timesteps
            print(f"\n🗺️  Regenerando mapa (timestep {self.num_timesteps})...")

            try:
                s = socket.socket()
                s.settimeout(10.0)
                s.connect(("localhost", self.control_port))
                s.sendall(b"RESTART\n")
                buf = b""
                while b"\n" not in buf:
                    buf += s.recv(4096)
                s.close()
                print("✅ Mapa regenerado.")
            except Exception as e:
                print(f"❌ Error regenerando mapa: {e}")
                return True

            # Bandera para que los workers de SubprocVecEnv trunquen el episodio
            with open(self.REGEN_FLAG, "w") as f:
                f.write(str(self.num_timesteps))

            time.sleep(1.0)

        return True


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
# Callback de SB3 para cambiar fase automáticamente según el rendimiento
class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_manager, check_freq=50_000):
        super().__init__()
        self.curriculum = curriculum_manager
        self.check_freq = check_freq
 
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
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
        
            

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                #print(info)
                self.ep_rewards.append(r)
                avg = sum(self.ep_rewards[-20:]) / min(len(self.ep_rewards), 20)
                print(f"   Ep {len(self.ep_rewards):4d} | "
                      f"pasos: {l:5d} | "
                      f"reward: {r:+8.1f} | "
                      f"media×20: {avg:+8.1f}")
        return True

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
        # IMPORTANTE: La conexión al socket debe ocurrir AQUÍ, dentro del proceso hijo
        env = SCPClassDEnv(tcp_host='localhost', tcp_port=7900, agent_id=i)
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
    
    model = PPO(
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
            MapRegenCallback(regen_every=1000000),
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
