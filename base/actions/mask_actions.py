import numpy as np

class Actions:
    # ── Curriculum 1: Humanos ────────────────────────────────────────────────
    NOOP            = 0
    MOVE_FORWARD    = 1
    MOVE_BACKWARD   = 2
    MOVE_RIGHT      = 3
    MOVE_LEFT       = 4
    SPRINT_FORWARD  = 5
    CAM_UP          = 6
    CAM_DOWN        = 7
    CAM_RIGHT       = 8
    CAM_LEFT        = 9
    INTERACT        = 10   # puertas / objetos del mundo
    PICK_ITEM       = 11   # recoger item del suelo
    EQUIP_KEYCARD   = 12   # equipar tarjeta en mano

    # ── Inventario humano (acciones nuevas) ──────────────────────────────────
    # Los IDs numéricos deben coincidir con ActionProcessor.cs (C#)
    DROP_ITEM       = 13   # tirar el item equipado al suelo
    EQUIP_PRIMARY   = 14   # equipar arma principal (rifle/MG/escopeta)
    EQUIP_SECONDARY = 15   # equipar arma secundaria (pistola/SMG)
    EQUIP_MEDICAL   = 16   # equipar medicamento (Medkit/Painkillers/Adrenaline/SCP500)
    EQUIP_GRENADE   = 17   # equipar granada (HE/Flash/SCP018)
    RELOAD          = 18   # recargar arma equipada
    USE_ITEM        = 19   # usar item equipado (disparar si arma, curar si med, lanzar si granada)

    # ── Curriculum 2: SCPs (esqueleto — implementar según el SCP) ───────────
    # Las acciones SCP se mapean a la misma tecla pero C# activa la habilidad
    # correcta según el rol. Así el espacio de acciones es compartido y limpio.
    SCP_ABILITY_PRIMARY   = 20  # click izq / ataque primario
    SCP_ABILITY_SECONDARY = 21  # mantener E / habilidad secundaria
    SCP_ABILITY_TERTIARY  = 22  # reserva (SCP-079 cambiar cámara, etc.)

    TOTAL_C1 = 20   # acciones válidas en curriculum 1 (humanos, incluyendo inventario)
    TOTAL    = 23   # acciones totales incluyendo SCPs


ACTION_MASKS = {
    # Roles humanos — curriculum 1
    "ClassD":       [True]*20 + [False]*3,
    "Scientist":    [True]*20 + [False]*3,
    "FacilityGuard":[True]*20 + [False]*3,
    "NtfPrivate":   [True]*20 + [False]*3,
    "NtfSergeant":  [True]*20 + [False]*3,
    "NtfSpecialist":[True]*20 + [False]*3,
    "NtfCaptain":   [True]*20 + [False]*3,
    "ChaosConscript":[True]*20 + [False]*3,
    "ChaosRifleman":[True]*20 + [False]*3,
    "ChaosRepressor":[True]*20 + [False]*3,
    "ChaosMarauder":[True]*20 + [False]*3,

    # SCPs — curriculum 2 (esqueleto: activar cuando se implementen)
    # Acciones 0-10 comunes, 11-19 desactivadas (no cogen items ni manipulan inventario),
    # 20-22 activas según habilidades del SCP concreto.
    "Scp049":  [True]*5 + [False] + [True]*5 + [False]*9 + [True,  True,  False],  # primaria=paro cardiaco, secundaria=zombificar
    "Scp096":  [True]*5 + [False] + [True]*5 + [False]*9 + [True,  False, False],  # primaria=rage
    "Scp173":  [True]*5 + [False] + [True]*5 + [False]*9 + [True,  False, False],  # primaria=snap
    "Scp106":  [True]*5 + [False] + [True]*5 + [False]*9 + [True,  True,  False],  # primaria=atrapar, secundaria=bolsillo
    "Scp939":  [True]*5 + [False] + [True]*5 + [False]*9 + [True,  False, False],  # primaria=morder
    "Scp079":  [True]*5  + [False]*6 + [False]*9 + [True, True, True],  # solo mover cam + habilidades

    # Fallback para roles desconocidos — solo acciones básicas seguras
    "Unknown": [True]*11 + [False]*12,
}

def get_action_mask(role_str: str) -> np.ndarray:
    """Devuelve la máscara de acciones para un rol dado."""
    mask = ACTION_MASKS.get(role_str, ACTION_MASKS["Unknown"])
    return np.array(mask, dtype=bool)
