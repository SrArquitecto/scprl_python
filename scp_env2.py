from __future__ import annotations
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
from input_controller import InputController
from gymnasium import spaces
import random
import mss
import cv2
import numpy as np
import subprocess
import re
import orjson
import os

N_ACTIONS  = 13  # resolución reducida para CNN (más rápido)

class SCPClassDEnv(gym.Env):
    """
    Observación combinada:
      - image:  (120, 160, 3) captura de pantalla reducida
      - vector: (68,) datos del servidor

    Acciones:
      0  W             adelante
      1  S             atrás
      2  A             izquierda
      3  D             derecha
      4  Shift+W       sprint
      5  E (tap)       abrir puerta / usar ascensor
      6  E (hold)      recoger objeto
      7  Ctrl          equipar keycard
      8  ratón ←       girar izquierda
      9  ratón →       girar derecha
      10 ratón ↑       mirar arriba
      11 ratón ↓       mirar abajo
      12 noop
    """
    N_WHISKERS = 8
    N_DANIO = 8
    N_ZONAS = 7
    N_ROOMS = 5
    N_HABITACIONES = 67
    N_LOCKERS  = 5
    N_ITEMS    = 5
    N_DOORS    = 15
    N_LIFTS    = 3
    N_PLAYERS  = 5
    N_GRAPH_NODES = 16
    N_GRAPH_FEAT  = 12
    N_GRAPH_ADJ   = 16 * 16
    N_GRAPH_MASK  = 16

    N_BASE         = 45
    N_ROOMS_FEAT   = 12
    N_ITEMS_FEAT   = 12
    N_DOORS_FEAT   = 13
    N_LIFTS_FEAT   = 16
    N_LOCKERS_FEAT = 12
    N_PLAYERS_FEAT = 11
    N_WHISKERS_FEAT = 2

    VEC_DIM2 = (
        N_BASE         +
        N_DANIO        +
        (N_ROOMS * N_ROOMS_FEAT) +
        (N_WHISKERS * N_WHISKERS_FEAT) +
        N_ZONAS        +
        N_HABITACIONES +
        (N_ITEMS * N_ITEMS_FEAT) +
        (N_DOORS * N_DOORS_FEAT) +
        (N_LIFTS * N_LIFTS_FEAT) +
        (N_LOCKERS * N_LOCKERS_FEAT) +
        (N_PLAYERS * N_PLAYERS_FEAT) +
        (N_GRAPH_NODES * N_GRAPH_FEAT) +
        N_GRAPH_ADJ    +
        N_GRAPH_MASK
    )
    VEC_DIM = ((VEC_DIM2 + 4) // 5) * 5
    TOTAL_OBS_DIM = VEC_DIM * 5

    def __init__(self, tcp_host='localhost', tcp_port=7900, agent_id=0):
        super().__init__()
        self._sock = None
        self.host = tcp_host
        self.port = tcp_port
        self.agent_id = agent_id
        
        # Hiperparámetros
        self._max_steps_ep = 512
        self._curriculum_level = 1
        self.mix_probability = 0.1

        # Estado del episodio
        self._prev_pos       = np.zeros(3)
        self._stuck          = 0
        self._ep_reward      = 0.0
        self._ep_steps       = 0
        self._spawn_pos      = None
        self._left_spawn_room = False
        self.last_action = -1
        self._has_seen_door = False

        # ── Reward state machine (celda → afuera) — constantes y estado ──────
        self.CE_R_LOCALIZE_FIRST = 15.0
        self.CE_R_FACING_DOOR    = 0.05
        self.CE_R_APPROACH_RATE  = 1.0
        self.CE_R_AT_DOOR        = 0.3
        self.CE_R_OPEN_DOOR      = 50.0
        self.CE_R_CROSS_DOOR     = 200.0
        self.CE_R_TIME           = -0.05
        self.CE_R_ANTI_STUCK     = -0.4
        self.CE_R_LOOK_AWAY      = -0.02
        self.CE_R_SEARCH_TURN    = 0.02
        self.CE_R_REGRESSION     = -1.0
        self.CE_R_STAY_OUT       = 0.1

        self.CE_DOOR_FOV         = 0.4    # FOV normalizado (1=180°). 0.4 = ~72° total
        self.CE_AT_DOOR_DIST     = 1.5
        self.CE_CROSS_MIN_DIST   = 2.0
        self.CE_STUCK_WINDOW     = 15
        self.CE_AT_DOOR_CAP      = 30    # máx steps cobrando R_AT_DOOR

        self._ce_state              = "INIT"  # INIT, LOCALIZED, AT_DOOR, OPENED, CROSSED
        self._ce_door_pos           = None
        self._ce_prev_door_dist     = None
        self._ce_door_was_open      = False
        self._ce_was_facing_door    = False
        self._ce_at_door_counter    = 0
        self._ce_pos_history        = []
        self._ce_localize_step      = None
        self._ce_open_step          = None
        self._ce_cross_step         = None
        self._ce_initial_spawn_pos  = None
        self._ce_door_opened        = False
        self._ce_last_breakdown     = {}
        # Espacios
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.TOTAL_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._prev_yaw         = 0.0
        self._left_own_cell    = False
        self._prev_pos = None
        self._stuck = 0
        self._has_seen_door = False
        self._pos_history = []
        # Dispositivos
        self._mss   = mss.mss()
        #self._sock = None  # Guardaremos aquí el socket persistente
        


        self.history_len = 5
        # Pre-asignamos la memoria: (3 frames) x (VEC_DIM datos por frame)
        self.vec_buffer = np.zeros((self.history_len, self.VEC_DIM), dtype=np.float32)


        self._left_own_cell   = False
        self._saw_exit_door   = False
        self._pos_history     = []
        self._prev_yaw        = 0.0
        self.vec_buffer = np.zeros((self.history_len, self.VEC_DIM), dtype=np.float32)



        
    
    
    def _reward(self, s, a):
        """State machine reward: INIT → LOCALIZED → AT_DOOR → OPENED → CROSSED.

        - INIT:     no ha visto la puerta todavía. Bonus por girar (buscar).
        - LOCALIZED: vio la puerta. Shaping por acercarse + bonus por mirarla.
        - AT_DOOR:  está a <1.5m. Bonus per-step + detectar apertura.
        - OPENED:   la puerta se abrió. Empuja hacia el otro lado.
        - CROSSED:  ya salió. Bonus por mantenerse fuera (anti-regresión).

        Yaw/pitch asumidos normalizados en rango -1/1 (1=180°).
        """
        r, breakdown = self._compute_cell_escape_reward(s, a)
        self._ce_last_breakdown = breakdown
        # Telemetría periódica del breakdown
        if self._ep_steps % 100 == 0 and breakdown:
            bd_str = " | ".join(f"{k}:{v:+.2f}" for k, v in breakdown.items() if abs(v) > 0.001)
            print(f"   [CE-Reward] {bd_str}")
        return r

    def _compute_cell_escape_reward(self, s, a):
        r = 0.0
        bd = {}

        # ── Estado del agente ────────────────────────────────────────────
        pos   = np.array([float(s["PosX"]), float(s["PosY"]), float(s["PosZ"])])
        yaw   = float(s["Yaw"])    # -1..1 (0=forward, 0.5=right, ±1=back)
        pitch = float(s["Pitch"])  # -1..1

        # Guardar spawn pos la primera vez (para detectar "haber salido")
        if self._ce_initial_spawn_pos is None:
            self._ce_initial_spawn_pos = pos.copy()

        # ── Detectar puerta en FOV ───────────────────────────────────────
        door_info = self._ce_find_door_in_fov(s, pos, yaw)

        # ── Transición INIT → LOCALIZED (primera vez que ve la puerta) ──
        if door_info is not None and self._ce_state == "INIT":
            self._ce_state = "LOCALIZED"
            self._ce_door_pos = door_info["pos"]
            self._ce_localize_step = self._ep_steps
            r += self.CE_R_LOCALIZE_FIRST
            bd["event_localize"] = self.CE_R_LOCALIZE_FIRST
        elif door_info is not None and self._ce_state == "LOCALIZED":
            # Actualizar pos puerta (puede haber jitter leve)
            self._ce_door_pos = door_info["pos"]

        # Distancia actual a la puerta (puede ser 99.0 si no se ha visto)
        if self._ce_door_pos is not None:
            current_door_dist = float(np.linalg.norm(pos - self._ce_door_pos))
        else:
            current_door_dist = 99.0

        # ── Reward por sub-estado ───────────────────────────────────────
        if self._ce_state == "INIT":
            # Pequeño bonus por girar (busca activa)
            yaw_rate = abs(float(s.get("AngVelYaw", 0.0)))
            if yaw_rate > 0.1:
                r += self.CE_R_SEARCH_TURN
                bd["search_turn"] = self.CE_R_SEARCH_TURN

        elif self._ce_state == "LOCALIZED":
            # Shaping de aproximación: se acerca = +, se aleja = -
            if self._ce_prev_door_dist is not None and self._ce_prev_door_dist < 99.0:
                delta = self._ce_prev_door_dist - current_door_dist  # + se acerca
                r_approach = max(-2.0, min(2.0, delta * self.CE_R_APPROACH_RATE))
                r += r_approach
                bd["approach"] = r_approach

            # Bonus por seguir mirando la puerta
            facing = self._ce_is_facing_door(pos, yaw)
            if facing:
                r += self.CE_R_FACING_DOOR
                bd["facing"] = self.CE_R_FACING_DOOR
            elif self._ce_was_facing_door:
                # Miraba y dejó de hacerlo: leve penalización
                r += self.CE_R_LOOK_AWAY
                bd["look_away"] = self.CE_R_LOOK_AWAY

        elif self._ce_state == "AT_DOOR":
            # Mantener posición cerca de la puerta
            if current_door_dist < self.CE_AT_DOOR_DIST:
                if self._ce_at_door_counter < self.CE_AT_DOOR_CAP:
                    r += self.CE_R_AT_DOOR
                    bd["at_door"] = self.CE_R_AT_DOOR
                    self._ce_at_door_counter += 1
            # Si se aleja sin abrir: regresión a LOCALIZED
            if current_door_dist > 3.0:
                self._ce_state = "LOCALIZED"
                self._ce_at_door_counter = 0
                r += self.CE_R_REGRESSION
                bd["regression"] = self.CE_R_REGRESSION

        elif self._ce_state == "OPENED":
            # Empuja hacia el otro lado: si dist-from-spawn > 1.5m, ya salió
            if self._ce_initial_spawn_pos is not None:
                dist_from_spawn = float(np.linalg.norm(pos - self._ce_initial_spawn_pos))
                if dist_from_spawn > 1.5:
                    r += self.CE_R_CROSS_DOOR
                    bd["event_cross"] = self.CE_R_CROSS_DOOR
                    self._ce_cross_step = self._ep_steps
                    self._ce_state = "CROSSED"

        elif self._ce_state == "CROSSED":
            # Mantenerse fuera (anti-regresión)
            if self._ce_door_pos is not None:
                d = float(np.linalg.norm(pos - self._ce_door_pos))
                if d > self.CE_CROSS_MIN_DIST:
                    r += self.CE_R_STAY_OUT
                    bd["stay_out"] = self.CE_R_STAY_OUT

        # ── Detectar evento de apertura de puerta ────────────────────────
        door_is_open = self._ce_check_door_open(s)
        if door_is_open and not self._ce_door_was_open and self._ce_state in ("AT_DOOR", "LOCALIZED"):
            # Si estaba lejos, forzar paso por AT_DOOR primero
            if self._ce_state != "AT_DOOR":
                self._ce_state = "AT_DOOR"
                self._ce_at_door_counter = 0
            # Bonus de apertura + transición a OPENED
            r += self.CE_R_OPEN_DOOR
            bd["event_open"] = self.CE_R_OPEN_DOOR
            self._ce_open_step = self._ep_steps
            self._ce_door_opened = True
            self._ce_state = "OPENED"
        self._ce_door_was_open = door_is_open

        # ── Transición LOCALIZED → AT_DOOR por proximidad ────────────────
        if self._ce_state == "LOCALIZED" and current_door_dist < self.CE_AT_DOOR_DIST:
            self._ce_state = "AT_DOOR"
            self._ce_at_door_counter = 0
            bd["enter_at_door"] = 0.0

        # ── Penalización temporal universal ──────────────────────────────
        r += self.CE_R_TIME
        bd["time"] = self.CE_R_TIME

        # ── Anti-stuck (ventana móvil de 15 steps) ───────────────────────
        r_stuck, bd_stuck = self._ce_anti_stuck(s, pos)
        r += r_stuck
        bd.update(bd_stuck)

        # ── Update prev ──────────────────────────────────────────────────
        self._ce_prev_door_dist = current_door_dist
        self._ce_was_facing_door = (self._ce_state in ("LOCALIZED", "AT_DOOR"))

        return float(r), bd

    # ─────────────────────────────────────────────────────────────────────
    # Helpers del cell-escape reward
    # ─────────────────────────────────────────────────────────────────────

    def _ce_find_door_in_fov(self, s, pos, yaw):
        """Detecta la puerta más cercana dentro del FOV. Retorna {pos, is_open} o None."""
        near_doors = s.get("NearDoors", [])
        if not near_doors:
            return None
        candidates = []
        for d in near_doors:
            dist = float(d.get("Distance", 99.0))
            if dist > 6.0:
                continue
            door_pos = np.array([
                pos[0] + float(d.get("RealRelX", 0.0)),
                pos[1] + float(d.get("RealRelY", 0.0)),
                pos[2] + float(d.get("RealRelZ", 0.0))
            ])
            if self._ce_is_in_fov(pos, yaw, door_pos, self.CE_DOOR_FOV):
                candidates.append((dist, door_pos, bool(d.get("IsOpen", False))))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return {"pos": candidates[0][1], "is_open": candidates[0][2]}

    def _ce_is_in_fov(self, pos, yaw, target_pos, fov_normalized):
        """¿Target dentro del FOV? yaw y fov en -1..1 (1 = 180°)."""
        dx = target_pos[0] - pos[0]
        dz = target_pos[2] - pos[2]
        if abs(dx) < 0.01 and abs(dz) < 0.01:
            return True
        # Ángulo al target normalizado: arctan2(dx, dz) / pi
        target_angle = np.arctan2(dx, dz) / np.pi  # -1..1
        diff = target_angle - yaw
        # Wrap-around (yaw 0.9 vs target -0.9 → diff debería ser -0.2, no 1.8)
        if diff > 1.0:
            diff -= 2.0
        elif diff < -1.0:
            diff += 2.0
        return abs(diff) < fov_normalized / 2

    def _ce_is_facing_door(self, pos, yaw):
        """¿El agente está mirando a la puerta?"""
        if self._ce_door_pos is None:
            return False
        return self._ce_is_in_fov(pos, yaw, self._ce_door_pos, self.CE_DOOR_FOV)

    def _ce_check_door_open(self, s):
        """¿Alguna puerta en NearDoors está abierta? Usa el flag agregado del C#."""
        return bool(s.get("DoorIsOpen", False))

    def _ce_anti_stuck(self, s, pos):
        """Penaliza si lleva STUCK_WINDOW steps sin moverse significativamente."""
        r = 0.0
        bd = {}
        self._ce_pos_history.append(pos.copy())
        if len(self._ce_pos_history) > self.CE_STUCK_WINDOW:
            self._ce_pos_history.pop(0)
        if len(self._ce_pos_history) == self.CE_STUCK_WINDOW and self._ep_steps > 20:
            dist = float(np.linalg.norm(self._ce_pos_history[-1] - self._ce_pos_history[0]))
            if dist < 0.3:
                r = self.CE_R_ANTI_STUCK
                bd["stuck"] = r
        return r, bd

    def _reward2(self, s, action=None):
        """
        Curriculum Fase 3: Salir del recinto de celdas.
        El agente ya sabe salir de su celda — ahora debe explorar
        el pasillo y encontrar la puerta de salida del recinto.
        """
        r = float(s.get("Reward", 0.0))

        # ── 1. Extracción de datos del Estado ─────────────────────────────────
        # Coordenadas absolutas del agente
        pos        = np.array([float(s["PosX"]), float(s["PosY"]), float(s["PosZ"])])
        yaw        = float(s["Yaw"])
        pitch      = float(s["Pitch"])
        aim_target = s.get("AimTarget", "None").strip()
        aim_dist   = float(s.get("AimDistance", 0.0))
        aim_room   = s.get("AimRoom", "Unknown")
        hit_name   = s.get("HitName", "None").strip().lower()  
        aim_door   = s.get("AimDoorName", "Unknow")
        room       = s.get("Room", "Unknown")
        ep_steps   = self._ep_steps
        
        # Inicialización de variables de control
        dist_actual_a_salida = float('inf')
        progreso = 0.0
        alineacion = None
        keywords = ["plainside", "door_right_lod1", "door_left_lod1", "door_right_lod0", "door_left_lod0"]
        
        near_doors        = s.get("NearDoors", [])
        closest_door      = min(near_doors, key=lambda d: float(d["Distance"])) if near_doors else None
        current_door_dist = float(closest_door["Distance"]) if closest_door else 99.0
        is_open           = bool(closest_door.get("IsOpen", False)) if closest_door else False

        puerta_mirada_y_abierta = False

        # Cache de puertas cercanas 
        current_doors = s.get("NearDoors", [])
        if id(current_doors) != self._last_near_doors_id:
            self.cached_doors_dict = {d.get("ColliderName"): d for d in current_doors if "ColliderName" in d}
            self._last_near_doors_id = id(current_doors)

        puerta = self.cached_doors_dict.get(hit_name)
        if puerta:
            if puerta.get("ColliderName") != "PlainSide":
                puerta_mirada_y_abierta = puerta.get("IsOpen", False)
        
        # Localización automática de la salida real
        if not self._salida_localizada:
            for p in current_doors:
                nombre_c = p.get("ColliderName", "None")
                if nombre_c != "PlainSide":
                    self.puerta_salida_actual = p
                    self._salida_localizada = True
                    break

        #print(self.puerta_salida_actual)           
        # 🌟 ESTRATEGIA REWARD1: Anclamos la posición absoluta de la salida una sola vez
        if self.puerta_salida_actual and getattr(self, '_exit_door_pos', None) is None:
            if p.get("ColliderName") != "PlainSide" and float(p.get("RealRelX", 0)) != 0:
                self._exit_door_pos = np.array([
                pos[0] + float(self.puerta_salida_actual.get("RealRelX")), 
                pos[1] + float(self.puerta_salida_actual.get("RealRelY")), 
                pos[2] + float(self.puerta_salida_actual.get("RealRelZ"))
                ])
        
        # 🌟 Calculamos la distancia usando la norma entre la posición absoluta actual y el anclaje fijo
        if getattr(self, '_exit_door_pos', None) is not None:
            dist_actual_a_salida = float(np.linalg.norm(pos - self._exit_door_pos))
            #print(f"🚀 AGENTE {self.agent_id} ANCLAJE FIJADO: {self._exit_door_pos}")
        else:
            dist_actual_a_salida = 99.0

    
        # ── 2. Hito: Salida exitosa del recinto completo ──────────────────────
        #if room != "LczClassDSpawn" and room != "Unknown":
        #if not self._left_spawn_room and ep_steps > 10 and dist_actual_a_salida < 1.5:
        if self._spawn_pos is None or self._ep_steps < 5:
            self._spawn_pos = pos
            
        dist_desde_spawn = np.linalg.norm(pos - self._spawn_pos)

        llegada_a_meta = (
            room != "LczClassDSpawn" and 
            room != "Unknown" and 
            dist_desde_spawn > 4.0 and   # debe haberse alejado razonablemente del spawn
            dist_desde_spawn < 40.0      # pero no un teletransporte absurdo (glitch)
        )

        if llegada_a_meta and not self._left_spawn_room:
            r += 600.0
            self._left_spawn_room = True
            print(f"🏃 [Paso {ep_steps}] ¡SALIÓ DEL RECINTO! → {room} (dist={dist_desde_spawn:.1f}m) (+1000)")
            return float(r)

        # Penalización leve por falta de movimiento instantáneo (Menos de 2 cm)
        if self._prev_pos is not None:
            if np.linalg.norm(pos - self._prev_pos) < 0.02:  
                r -= 0.1

        # ── 3. Sistema de Exploración por Rejilla Absoluta ────────────────────
        coord_actual = (int(pos[0] * 2), int(pos[2] * 2))

        if coord_actual not in self._visitadas:
            r += 0.05  
            self._visitadas.add(coord_actual)
        else:
            r -= 0.01  

        # ── 4. Control de Mirada Inteligente ──────────────────────────────────
        if aim_target == 0.3:
            r -= 0.40
        elif aim_target == 0.2:
            r -= 0.25
        elif aim_target == 0.4:
            if any(word in hit_name for word in keywords):
                r += 0.05 if not self._left_own_cell else -0.10
            else:
                r += 0.15  
        elif aim_target == 0.1:
            if "tunnels" in hit_name:
                r += 0.03  
            elif "prisonchamber" in hit_name or "antijumpercells" in hit_name:
                r += 0.01 if not self._left_own_cell else -0.15
            else:
                r += 0.01 if aim_dist > 0.8 else -0.10

        if abs(pitch) > 0.3:
            r -= 0.10

        # Premio por fijar la vista en el objetivo final
        looking_at_exit = (
            #aim_target == "Door" and
            aim_room not in ("LczClassDSpawn", "Unknown", "") and
            not any(word in hit_name for word in keywords)
        )

        if looking_at_exit:
            r += 0.30  
            if not getattr(self, '_saw_exit_door', False):
                r += 50.0  
                self._saw_exit_door = True
                print(f"🚪 AGENTE: {self.agent_id} [Paso {ep_steps}] ¡Puerta de salida real localizada en {hit_name}! (+50)")
        
        # ── 5. Detector de Inercia y Atasco ───────────────────────────────────
        self._pos_history.append(pos.copy())
        if len(self._pos_history) > 15:
            self._pos_history.pop(0)

        if len(self._pos_history) == 15 and ep_steps > 10:
            dist_neta = np.linalg.norm(pos - self._pos_history[0])
            is_static = dist_neta < 0.15  
        else:
            is_static = False

        if is_static:
            if aim_target == 0.4 and not any(word in hit_name for word in keywords) and not is_open and current_door_dist < 1.5:
                r += 0.05  
                self._stuck = 0  
            else:
                self._stuck += 1
                r -= 0.05 * min(self._stuck, 20)
                if self._stuck == 15:
                    print(f"⚠️ AGENTE: {self.agent_id} [Paso {ep_steps}] Atasco detectado ({aim_target} | {hit_name})")
        else:
            self._stuck = 0

        # ── 6. Validación de Spawn Absoluto e Hitos de Celda ──────────────────
        if self._spawn_pos is None:
            self._spawn_pos = pos
            
        dist_desde_spawn = np.linalg.norm(pos - self._spawn_pos)

        if not self._left_own_cell and dist_desde_spawn > 3.5 and ep_steps > 10:
            r += 300.0
            self._left_own_cell = True
            print(f"🚪 AGENTE: {self.agent_id} [Paso {ep_steps}] Salió de su celda al pasillo (+300)")

        if self._left_own_cell:
            if "prisonchamber" in hit_name or "antijumpercells" in hit_name:
                r -= 0.20  

        # ── 7. Lógica de Progreso y Recompensa por Acercamiento ────────────────
        if getattr(self, '_exit_door_pos', None) is not None:
            # 🌟 CORRECCIÓN 1: Inicialización blindada
            # Si el récord está vacío, en 0 o heredó el 99.0 de cuando la puerta no era visible,
            # lo fijamos inmediatamente a la distancia real actual del agente.
            if getattr(self, '_min_dist_to_exit', None) is None or self._min_dist_to_exit in (0, 99.0):
                self._min_dist_to_exit = dist_actual_a_salida
                
            progreso = 0.0
            
            # 🌟 CORRECCIÓN 2: Umbral de precisión milimétrica (0.005 = medio centímetro)
            # Cualquier progreso real hacia adelante, por pequeño que sea, se registra.
            if dist_actual_a_salida < self._min_dist_to_exit - 0.005:
                progreso = self._min_dist_to_exit - dist_actual_a_salida
                
                # 🌟 CORRECCIÓN 3: Escalado agresivo (* 40.0)
                # Al pasar por el filtro del final (/ 10.0), avanzar 1 metro neto hacia la puerta
                # significará un acumulado sólido de +4.0 puntos netos. ¡Ahora la IA sí lo sentirá!
                r += progreso * 40.0  
                self._min_dist_to_exit = dist_actual_a_salida

            # Penalización si se aleja notablemente de su mejor marca (Margen de medio metro para giros)
            if dist_actual_a_salida > self._min_dist_to_exit + 0.5:
                r -= 0.2

            if self._left_own_cell and puerta_mirada_y_abierta and dist_actual_a_salida < 1.5:
                if not getattr(self, '_exit_door_opened', False):
                    r += 200.0  
                    self._exit_door_opened = True
                    print(f"🔓 AGENTE: {self.agent_id} [Paso {ep_steps}] ¡EL AGENTE ABRIÓ LA PUERTA DE SALIDA! (+200)")
            
            # ── 8. Recompensa por Alineación de Mirada (Dirección) ────────────────
            # 🌟 CORRECCIÓN: Como ahora la puerta es absoluta, el vector hacia ella es (Destino - Origen)
            vec_hacia_salida = np.array([self._exit_door_pos[0] - pos[0], self._exit_door_pos[2] - pos[2]])
            norm_salida = np.linalg.norm(vec_hacia_salida)
            
            if norm_salida > 0.10:
                vec_hacia_salida /= norm_salida
                vec_mirada = np.array([s.get("ForwardX", 0.0), s.get("ForwardZ", 0.0)])
                norm_mirada = np.linalg.norm(vec_mirada)
                
                if norm_mirada > 0.0001:
                    vec_mirada /= norm_mirada
                
                alineacion = np.dot(vec_mirada, vec_hacia_salida)

                if alineacion > 0.9: 
                    r += 0.5 
                elif alineacion > 0:
                    r += alineacion * 0.2
        else:
            self._prev_dist_to_exit = None
        
        # ── 9. Costes de Tiempo y Penalizaciones por Bucles ───────────────────
        r -= 0.30 if not self._left_own_cell else 0.10  

        if ep_steps >= self._max_steps_ep:
            r -= 50.0
            print(f"⏰ AGENTE: {self.agent_id} [Paso {ep_steps}] Time-out del episodio (-50)")

        mira_interior_celda = ("prisonchamber" in hit_name or
                            "plainside" in hit_name or 
                            "antijumpercells" in hit_name or 
                            ("collider" in hit_name and aim_dist < 1.0))

        if mira_interior_celda:
            self._pasos_encerrado += 1
            if self._pasos_encerrado >= 400:
                print(f"🛑 AGENTE: {self.agent_id} [Paso {ep_steps}] ¡El agente lleva 400 pasos atrapado en la celda (-50p)!")
                r -= 50.0  
                self._pasos_encerrado = 0
        else:
            self._pasos_encerrado = 0

        alin_texto = f"{alineacion:.3f}" if alineacion is not None else "None"

        if ep_steps % 100 == 0:
            print(f"📊 AGENTE: {self.agent_id} [Paso {ep_steps}] "
                f"Enfoque: {aim_target}({aim_room} | {hit_name}) {aim_dist:.1f}m | "
                f"DistSpawn: {dist_desde_spawn:.1f}m | "
                f"DistSalida: {dist_actual_a_salida:.2f}m | Progreso: {progreso:.4f} | Alin: {alin_texto} | "
                f"R: {r:.2f}")

        # Guardar estados anteriores para el siguiente paso
        self._prev_pos = pos.copy()
        self._prev_yaw = yaw
        
        return float(r)


        

    # ── API Gymnasium ──────────────────────────────────────────────────────



    
        
    
        
            
        #Curriculum 1: 
        

         
        #return obs, reward, terminated, False, {}





