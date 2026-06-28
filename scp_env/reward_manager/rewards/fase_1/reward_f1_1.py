from scp_env.reward_manager.rewards.reward_base import RewardBase
import numpy as np

# ── Magnitudes del reward (state machine de escape de celda) ─────────────
# Estado: INIT → LOCALIZED → AT_DOOR → OPENED → CROSSED
CE_R_LOCALIZE_FIRST = 15.0
CE_R_FACING_DOOR    = 0.05
CE_R_APPROACH_RATE  = 1.0
CE_R_AT_DOOR        = 0.3
CE_R_OPEN_DOOR      = 50.0
CE_R_CROSS_DOOR     = 200.0
CE_R_TIME           = -0.05
CE_R_ANTI_STUCK     = -0.4
CE_R_LOOK_AWAY      = -0.02
CE_R_SEARCH_TURN    = 0.02
CE_R_REGRESSION     = -1.0
CE_R_STAY_OUT       = 0.1

CE_DOOR_FOV         = 0.4    # FOV normalizado (1=180°). 0.4 = ~72° total
CE_AT_DOOR_DIST     = 1.5
CE_CROSS_MIN_DIST   = 2.0
CE_STUCK_WINDOW     = 15
CE_AT_DOOR_CAP      = 30     # máx steps cobrando R_AT_DOOR

# Detección de teleport (nueva ronda en medio de episodio)
TELEPORT_THRESHOLD = 5.0   # metros: salto de posición en 1 step

class Fase1SalirHabitacion(RewardBase):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self._min_dist_to_exit = None
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
        self._ep_steps = 0
        
    def calcular_reward(self, s: dict, action: int, info_entorno: dict, _ep_steps) -> float:
        """State machine reward: INIT → LOCALIZED → AT_DOOR → OPENED → CROSSED.

        - INIT:     no ha visto la puerta todavía. Bonus por girar (buscar).
        - LOCALIZED: vio la puerta. Shaping por acercarse + bonus por mirarla.
        - AT_DOOR:  está a <1.5m. Bonus per-step + detectar apertura.
        - OPENED:   la puerta se abrió. Empuja hacia el otro lado.
        - CROSSED:  ya salió. Bonus por mantenerse fuera (anti-regresión).

        Yaw/pitch asumidos normalizados en rango -1/1 (1=180°).
        """
        r, breakdown = self._compute_cell_escape_reward(s, action)
        self._ce_last_breakdown = breakdown
        # Telemetría periódica del breakdown
        self._ep_steps = _ep_steps
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
            r += CE_R_LOCALIZE_FIRST
            bd["event_localize"] = CE_R_LOCALIZE_FIRST
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
                r += CE_R_SEARCH_TURN
                bd["search_turn"] = CE_R_SEARCH_TURN

        elif self._ce_state == "LOCALIZED":
            # Shaping de aproximación: se acerca = +, se aleja = -
            if self._ce_prev_door_dist is not None and self._ce_prev_door_dist < 99.0:
                delta = self._ce_prev_door_dist - current_door_dist  # + se acerca
                r_approach = max(-2.0, min(2.0, delta * CE_R_APPROACH_RATE))
                r += r_approach
                bd["approach"] = r_approach

            # Bonus por seguir mirando la puerta
            facing = self._ce_is_facing_door(pos, yaw)
            if facing:
                r += CE_R_FACING_DOOR
                bd["facing"] = CE_R_FACING_DOOR
            elif self._ce_was_facing_door:
                # Miraba y dejó de hacerlo: leve penalización
                r += CE_R_LOOK_AWAY
                bd["look_away"] = CE_R_LOOK_AWAY

        elif self._ce_state == "AT_DOOR":
            # Mantener posición cerca de la puerta
            if current_door_dist < CE_AT_DOOR_DIST:
                if self._ce_at_door_counter < CE_AT_DOOR_CAP:
                    r += CE_R_AT_DOOR
                    bd["at_door"] = CE_R_AT_DOOR
                    self._ce_at_door_counter += 1
            # Si se aleja sin abrir: regresión a LOCALIZED
            if current_door_dist > 3.0:
                self._ce_state = "LOCALIZED"
                self._ce_at_door_counter = 0
                r += CE_R_REGRESSION
                bd["regression"] = CE_R_REGRESSION

        elif self._ce_state == "OPENED":
            # Empuja hacia el otro lado: si dist-from-spawn > 1.5m, ya salió
            if self._ce_initial_spawn_pos is not None:
                dist_from_spawn = float(np.linalg.norm(pos - self._ce_initial_spawn_pos))
                if dist_from_spawn > 1.5:
                    r += CE_R_CROSS_DOOR
                    bd["event_cross"] = CE_R_CROSS_DOOR
                    self._ce_cross_step = self._ep_steps
                    self._ce_state = "CROSSED"

        elif self._ce_state == "CROSSED":
            # Mantenerse fuera (anti-regresión)
            if self._ce_door_pos is not None:
                d = float(np.linalg.norm(pos - self._ce_door_pos))
                if d > CE_CROSS_MIN_DIST:
                    r += CE_R_STAY_OUT
                    bd["stay_out"] = CE_R_STAY_OUT

        # ── Detectar evento de apertura de puerta ────────────────────────
        door_is_open = self._ce_check_door_open(s)
        if door_is_open and not self._ce_door_was_open and self._ce_state in ("AT_DOOR", "LOCALIZED"):
            # Si estaba lejos, forzar paso por AT_DOOR primero
            if self._ce_state != "AT_DOOR":
                self._ce_state = "AT_DOOR"
                self._ce_at_door_counter = 0
            # Bonus de apertura + transición a OPENED
            r += CE_R_OPEN_DOOR
            bd["event_open"] = CE_R_OPEN_DOOR
            self._ce_open_step = self._ep_steps
            self._ce_door_opened = True
            self._ce_state = "OPENED"
        self._ce_door_was_open = door_is_open

        # ── Transición LOCALIZED → AT_DOOR por proximidad ────────────────
        if self._ce_state == "LOCALIZED" and current_door_dist < CE_AT_DOOR_DIST:
            self._ce_state = "AT_DOOR"
            self._ce_at_door_counter = 0
            bd["enter_at_door"] = 0.0

        # ── Penalización temporal universal ──────────────────────────────
        r += CE_R_TIME
        bd["time"] = CE_R_TIME

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
            if self._ce_is_in_fov(pos, yaw, door_pos, CE_DOOR_FOV):
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
        return self._ce_is_in_fov(pos, yaw, self._ce_door_pos, CE_DOOR_FOV)

    def _ce_check_door_open(self, s):
        """¿Alguna puerta en NearDoors está abierta? Usa el flag agregado del C#."""
        return bool(s.get("DoorIsOpen", False))

    def _ce_anti_stuck(self, s, pos):
        """Penaliza si lleva STUCK_WINDOW steps sin moverse significativamente."""
        r = 0.0
        bd = {}
        self._ce_pos_history.append(pos.copy())
        if len(self._ce_pos_history) > CE_STUCK_WINDOW:
            self._ce_pos_history.pop(0)
        if len(self._ce_pos_history) == CE_STUCK_WINDOW and self._ep_steps > 20:
            dist = float(np.linalg.norm(self._ce_pos_history[-1] - self._ce_pos_history[0]))
            if dist < 0.3:
                r = CE_R_ANTI_STUCK
                bd["stuck"] = r
        return r, bd