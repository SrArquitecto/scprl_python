from turtle import pos
import time
import subprocess
import gymnasium as gym
import numpy as np
import socket
import json
import time
import mss
from Xlib import display
import cv2
from sympy import rad
from gymnasium import spaces
import random
import mss
import cv2
import numpy as np
import subprocess
import re
import orjson
import os

from .comunicacion import Comunicacion
from .observaciones import Observacion
from .state_stacker import StateStacker
from .tipos import (HISTORY_LEN, VEC_DIM, N_ACTIONS, TOTAL_OBS_DIM)
from .reward_manager.reward_manager import RewardManager

class Scp_env(gym.Env):
    
    def __init__(self, host = "localhost", port = 7900, agent_id = 0, role = "classd"):
        self.agent_id = agent_id
        self.host = host
        self.port = port
        self.role = role
        
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(TOTAL_OBS_DIM,), dtype=np.float32
        )
        
        self.comunicacion = Comunicacion(host, port, agent_id, role)
        self.observacion = Observacion()
        self.stacker = StateStacker(HISTORY_LEN, VEC_DIM)
        self.reward_manager = RewardManager()
        
        self._max_steps = 10000
        self._max_steps_ep = 512
        self._curriculum_level = 1
        self._max_steps_spawn = 450
        self.mix_probability = 0.1
        
        self.current_role = "ClassD"
        
        self._ep_steps = 0
        self._ep_reward = 0
        
        self.comunicacion.conexion()
        
    def step(self, action):

        # ─── 🛑 FRENO DE MANO TEMPORAL ───
        #action = 12 # Forzamos a que la acción sea 0 para que no se mueva solo
        # ──────────────────────────────────

        if not hasattr(self, '_gc_counter'):
            self._gc_counter = 0
        self._gc_counter += 1
        if self._gc_counter >= 512:
            self._gc_counter = 0
            import gc
            gc.collect()

        try:
            s = self.comunicacion.enviar_accion(action)
        except Exception as e:
            print(f"\n❌ Agente {self.agent_id} error en step, intentando reconectar...: {e}")
            if self.comunicacion.reconectar(max_retries=5, retry_delay=1.0):
                try:
                    s = self.comunicacion.enviar_accion(action)
                except Exception as e2:
                    print(f"❌ Agente {self.agent_id} sigue sin respuesta tras reconexión: {e2}")
                    return self.observacion.empty_obs(), 0.0, True, False, {}
            else:
                print(f"❌ Agente {self.agent_id} no pudo reconectar. Devolviendo obs vacía.")
                return self.observacion.empty_obs(), 0.0, True, False, {}
        
        self.current_role = s.get("Role", "ClassD")
        
        # ── DETECTOR DE MUERTE ───────────────────────────────────────────────
        # Si el bot está muerto (Health=0) continuamos el episodio sin interrumpirlo:
        # devolvemos obs vacía (shape correcta) y reward=0 hasta que C# lo respawnee.
        # El resto de agentes siguen entrenando sin pausa.
        if float(s.get("Health", 1)) <= 0:
            return self.observacion.empty_obs(), 0.0, False, False, {}

        pos = np.array([float(s["PosX"]), float(s["PosY"]), float(s["PosZ"])])
        print(s)
        obs = self._get_obs(s)
        
        info_entorno = {"curriculum_level": self._curriculum_level} 
        
        terminated_obs = bool(s.get("Done", False))
        
        # 3. 🌟 LLAMADA DINÁMICA AL REWARD MANAGER
        reward, terminated_ep = self.reward_manager.calcular_reward(s, action, info_entorno, self._ep_steps)
        
        #if self._curriculum_level == 1:
        #    res_reward = self._reward(s, action) 
        #else:
        #    res_reward = self._reward2(s, action)
        
        #reward = 0 #float(res_reward) #+ time_penalty
        terminated = terminated_obs or terminated_ep
        #print(self._ep_reward)
        self._ep_reward += reward
        self._ep_steps  += 1

        #print(self._ep_reward)
        
        """
        if self._left_spawn_room:
            print("🚪 ¡Objetivo alcanzado! Reseteando entorno de entrenamiento.")
            self.comunicacion.respawn()
            terminated = True

        if self._ep_steps >= self._max_steps_ep:
            self.comunicacion.respawn()
             # Bajado de 5 a 2 segundos
            terminated = True

        if self._fin_episodio:
            self.comunicacion.respawn()
            time.sleep(10) # Bajado de 5 a 2 segundos
            terminated = True
        """
        
        # Bandera escrita por MapRegenCallback cuando regenera el mapa
        if os.path.exists("/tmp/scp_regen_flag"):
            terminated = True
            try:
                os.remove("/tmp/scp_regen_flag")
            except: pass
            print(f"\n🗺️  Mapa regenerado por callback (step {self._ep_steps})")
            # Reconectar socket (el C# lo cerró durante el reinicio)
            try:
                self.comunicacion.cerrar()
            except: pass
            self._sock = None
            time.sleep(2.0)
            self.comunicacion.reconectar()
            
            
        return obs, reward, terminated, False, {}    
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.reward_manager.reset_fase_actual()
        
        if self._ep_steps > 0:
            print(f"  Ep: {self._ep_steps} pasos | reward: {self._ep_reward:+.1f}")

        self.current_role = "ClassD" 
        
        self._ep_reward = 0.0
        self._ep_steps  = 0

        #if self._curriculum_level != 1:
        #    if random.random() < self.mix_probability:
        #        self._curriculum_level = 1
        #        self._max_steps_ep = 512
        #    else:
        #        self._curriculum_level = 2
        #        self._max_steps_ep = 2048

        vector_plano = self.stacker.reset()
        
        print("🔄 Reiniciando entorno...", end="", flush=True)

        max_reset_attempts = 30
        for reset_attempt in range(max_reset_attempts):
            try:
                s = self.comunicacion.solicitar_estado()
                # Validar que el estado sea un diccionario válido y listo
                if isinstance(s, dict) and s.get("Health", 0) > 0 and s.get("Zone") != "Unknown":
                    print("✅")
                    break # Salimos del bucle cuando el estado es válido
            except Exception:
                pass
            time.sleep(1.0)
        else:
            # Si llegamos aquí, el bot no respawneó después de 30 segundos
            print(f"⚠️ Timeout: bot no respawneó después de {max_reset_attempts}s, continuando...")
            # Retornar observación vacía con la shape CORRECTA (TOTAL_OBS_DIM = VEC_DIM * history_len)
            # VEC_DIM solo es un frame; SubprocVecEnv necesita el buffer completo o np.stack() falla
            return np.zeros(TOTAL_OBS_DIM, dtype=np.float32), {}
        
        
        
        # 2. Codificamos el vector una sola vez
        vec = self.observacion.encode_vector(s)
        if vec.shape[0] != VEC_DIM:
            raise ValueError(f"CRÍTICO: El vector generado tiene tamaño {vec.shape[0]}, pero el buffer espera {VEC_DIM}. "
                             f"Revisa la suma en tu cálculo de VEC_DIM.")
        # 3. Limpieza y precarga del buffer (3 frames idénticos al inicio)
        vector_plano = self.stacker.seed(vec)
        
        # 4. Retornamos la observación usando el buffer ya cargado
        # Nota: aquí NO llamamos a _get_obs(s) porque ya preparamos el buffer
        
        return vector_plano, {}

    # En _get_obs
    def _get_obs(self, s):
        vec = self.observacion.encode_vector(s)
        vector_plano = self.stacker.update(vec)
        if vector_plano.shape[0] != self.observation_space.shape[0]:
            print(f"❌ ERROR CRÍTICO: Dimensión esperada {self.observation_space.shape[0]}, recibida {self.stacker.stack.shape[0]}")
        return vector_plano  
        
    def close(self):
        try: 
            self.comunicacion.cerrar()
        except: 
            pass
