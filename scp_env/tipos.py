from __future__ import annotations
import numpy as np
from gymnasium import spaces



# ── Acciones (idénticas a v3) ────────────────────────────────────────────
N_ACTIONS = 13

ACTION_NAMES = {
    0: "W",           # adelante
    1: "S",           # atrás
    2: "A",           # izquierda
    3: "D",           # derecha
    4: "Shift+W",     # sprint
    5: "E_tap",       # abrir puerta / usar ascensor
    6: "E_hold",      # recoger objeto
    7: "Ctrl",        # equipar keycard
    8: "mouse_left",  # girar izquierda
    9: "mouse_right", # girar derecha
    10: "mouse_up",   # mirar arriba
    11: "mouse_down", # mirar abajo
    12: "noop",
}


# ── Dimensiones del vector de observación (5 frames apilados) ────────────
N_DAMAGE        = 8
N_ZONES         = 7
N_ROOMS         = 5
N_HABITACIONES  = 67
N_LOCKERS       = 5
N_ITEMS         = 5
N_DOORS         = 15
N_LIFTS         = 3
N_PLAYERS       = 5
N_GRAPH_NODES   = 16
N_GRAPH_FEAT    = 12
N_GRAPH_ADJ     = 16 * 16
N_GRAPH_MASK    = 16

N_BASE          = 45
N_DANIO         = 8
N_ZONAS         = 7
N_WHISKERS      = 8
N_ROOMS_FEAT    = 12
N_ITEMS_FEAT    = 12
N_DOORS_FEAT    = 13
N_LIFTS_FEAT    = 16
N_LOCKERS_FEAT  = 12
N_PLAYERS_FEAT  = 11
N_WHISKERS_FEAT = 2     # [dist, type] por whisker
N_NAV_FEAT      = 8     # wall×4 + door dist/yaw + area + shape (room nav)

# 5 frames apilados
HISTORY_LEN = 5

N_ACTIONS = 13

VEC_DIM2 = (
    N_BASE
    + N_DAMAGE
    + (N_ROOMS * N_ROOMS_FEAT)
    + (N_WHISKERS * N_WHISKERS_FEAT)
    + N_NAV_FEAT
    + N_ZONES
    + N_HABITACIONES
    + (N_ITEMS * N_ITEMS_FEAT)
    + (N_DOORS * N_DOORS_FEAT)
    + (N_LIFTS * N_LIFTS_FEAT)
    + (N_LOCKERS * N_LOCKERS_FEAT)
    + (N_PLAYERS * N_PLAYERS_FEAT)
    + (N_GRAPH_NODES * N_GRAPH_FEAT)
    + N_GRAPH_ADJ
    + N_GRAPH_MASK
)
VEC_DIM = ((VEC_DIM2 + 4) // 5) * 5

TOTAL_OBS_DIM = VEC_DIM * HISTORY_LEN


# ── Magnitudes del reward (state machine de escape de celda) ─────────────
# Estado: INIT → LOCALIZED → AT_DOOR → OPENED → CROSSED
R_LOCALIZE_FIRST = 15.0   # one-time: primera vez que se ve la puerta (en ronda nueva)
R_FACING_DOOR    = 0.05   # per-step bonus por mirar a la puerta (proporcional)
R_APPROACH_RATE  = 1.0    # por metro de progreso hacia la puerta
R_AT_DOOR        = 0.3    # per-step estando a <1.5m de la puerta
R_OPEN_DOOR      = 50.0   # one-time: evento de apertura
R_CROSS_DOOR     = 200.0  # one-time: cruzar al otro lado
R_TIME           = -0.05  # per-step (penalización universal)
R_ANTI_STUCK     = -0.4   # per-step cuando lleva 15 sin moverse
R_LOOK_AWAY      = -0.02  # per-step cuando estaba alineado y deja de estarlo
R_SEARCH_TURN    = 0.02   # per-step bonus por girar (búsqueda activa en INIT)
R_REGRESSION     = -1.0   # one-time: se aleja de la puerta sin abrir
R_STAY_OUT       = 0.1    # per-step: mantenerse fuera de la celda (CROSSED)
R_SHAPE_BONUS    = 1.0    # one-time: bonus si RoomShape > 1.5 (celda alargada)

DOOR_FOV_NORMALIZED = 0.4   # FOV normalizado (~72°): se considera "mirando" si |yaw_rel| < 0.2
AT_DOOR_DIST         = 1.5   # metros: distancia para entrar en AT_DOOR
CROSS_MIN_DIST       = 2.0   # metros: distancia mínima para contar como "cruzó"
STUCK_WINDOW         = 15    # steps sin moverse = atasco
AT_DOOR_CAP          = 30    # máx steps cobrando R_AT_DOOR

# Detección de teleport (nueva ronda en medio de episodio)
TELEPORT_THRESHOLD = 8.0   # metros: salto de posición en 1 step


# ── Helpers ─────────────────────────────────────────────────────────────
def yaw_to_degrees(yaw_normalized: float) -> float:
    """
    Convierte el yaw normalizado (-1/1) que envía el C# a grados (0-360).
    Convención C#: 0 = +Z (forward), 0.5 = +X (right), ±1 = -Z (back).
    """
    return yaw_normalized * 180.0


def normalize_angle_degrees(angle_deg: float) -> float:
    """
    Normaliza un ángulo en grados a [-180, 180].
    Usado para comparar yaw de agente y yaw a target sin wrap-around.
    """
    a = angle_deg % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def make_observation_space() -> spaces.Box:
    """
    Observation space per-agent: vector apilado de 5 frames.

    Cada frame es VEC_DIM floats. TOTAL_OBS_DIM = VEC_DIM * 5.
    Rango [-inf, +inf] porque hay valores normalizados (0-1) mezclados con
    distancias reales (pueden ser > 1m).
    """
    return spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(TOTAL_OBS_DIM,),
        dtype=np.float32,
    )


def make_action_space() -> spaces.Discrete:
    """Action space per-agent: 13 acciones discretas (idénticas a v3)."""
    return spaces.Discrete(N_ACTIONS)
