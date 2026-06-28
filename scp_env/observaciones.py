import numpy as np

from .tipos import (
    N_BASE, N_DANIO, N_ROOMS, N_ROOMS_FEAT, N_ITEMS, N_ITEMS_FEAT,
    N_DOORS, N_DOORS_FEAT, N_LIFTS, N_LIFTS_FEAT, N_LOCKERS, N_LOCKERS_FEAT,
    N_PLAYERS, N_PLAYERS_FEAT, N_WHISKERS, N_WHISKERS_FEAT, N_NAV_FEAT,
    N_ZONAS, N_HABITACIONES, N_GRAPH_NODES, N_GRAPH_FEAT, N_GRAPH_ADJ,
    N_GRAPH_MASK, VEC_DIM, HISTORY_LEN, TOTAL_OBS_DIM
)
class Observacion:
    
    def __init__(self):
        self.DAMAGE_TYPE_MAP = {
            "None": 0.0, "Firearm": 1.0, "Explosion": 2.0,
            "Scp": 3.0, "Fall": 4.0, "Status": 5.0, "Unknown": 6.0
        }
        
        self.ZONE_MAP = {
            "Unspecified": 0,
            "LightContainment": 1,
            "HeavyContainment": 2,
            "Entrance": 3,
            "Surface": 4,
            "Pocket": 5,        # 🚪 ¡Clave para cuando SCP-106 te atrapa!
            "Other": 6
        }
        

        self.ROOM_MAP = {
            "Unknown": 0,
            # --- Light Containment Zone (LCZ) ---
            "LczArmory": 1, "LczCurve": 2, "LczStraight": 3, "Lcz914": 4, "LczCrossing": 5,
            "LczTCross": 6, "LczCafe": 7, "LczPlants": 8, "LczToilets": 9, "LczAirlock": 10,
            "Lcz173": 11, "LczClassDSpawn": 12, "LczCheckpointB": 13, "LczGlassBox": 14,
            "LczCheckpointA": 15, "Lcz330": 16,
            # --- Heavy Containment Zone (HCZ) ---
            "Hcz079": 17, "HczEzCheckpointA": 18, "HczEzCheckpointB": 19, "HczArmory": 20,
            "Hcz939": 21, "HczHid": 22, "Hcz049": 23, "HczCrossing": 24, "Hcz106": 25,
            "HczNuke": 26, "HczTesla": 27, "HczCurve": 28, "Hcz096": 29, "HczStraight": 30,
            "HczTestRoom": 31, "HczElevatorA": 32, "HczElevatorB": 33, "HczDss08": 34,
            "HczCornerDeep": 35, "HczIntersectionJunk": 36, "HczIntersection": 37,
            "HczStraightC": 38, "HczStraightPipeRoom": 39, "HczStraightVariant": 40,
            "Hcz127": 41, "HczServerRoom": 42, "HczIncineratorWayside": 43, "HczLoadingBay": 44,
            # --- Entrance Zone (EZ) ---
            "EzVent": 45, "EzIntercom": 46, "EzGateA": 47, "EzDownstairsPcs": 48, "EzCurve": 49,
            "EzPcs": 50, "EzCrossing": 51, "EzCollapsedTunnel": 52, "EzConference": 53,
            "EzChef": 54, "EzStraight": 55, "EzStraightColumn": 56, "EzCafeteria": 57,
            "EzUpstairsPcs": 58, "EzGateB": 59, "EzShelter": 60, "EzTCross": 61,
            "EzCheckpointHallwayA": 62, "EzCheckpointHallwayB": 63, "EzSmallrooms": 64,
            # --- Especiales ---
            "Pocket": 65, "Surface": 66
        }
    
    def empty_obs(self):
        return np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
    
    def encode_vector(self, s):

        zone_enc = self.ZONE_MAP.get(s.get("Zone", "Unspecified"), 0)
        
        room_enc = self.ROOM_MAP.get(s.get("Room", "Unknown"), 0) 

        aim_enc = s.get("AimTarget", 0.0)
        
        faction = self.safe_val(s.get("FactionId", 0.0))
        
        aim_dist = self.safe_val(s.get("AimDistance", 0.0)) / 75.0  # Normalización de distancia de mira
        
        yaw_n = self.safe_val(s.get("Yaw", 0.0))
        pitch_n = self.safe_val(s.get("Pitch", 0.0))
        #yaw_n = self.normalize_angle(yaw)
        #pitch_n = self.normalize_angle(pitch)

        lin_vel_raw = self.safe_val(s.get("LinVel", 0.0))
        lat_vel_raw = self.safe_val(s.get("LatVel", 0.0))
        ver_vel_raw = self.safe_val(s.get("VerVel", 0.0))
        lin_vel_n = np.clip(lin_vel_raw, -1.0, 1.0) 
        lat_vel_n = np.clip(lat_vel_raw, -1.0, 1.0)
        ver_vel_n = np.clip(ver_vel_raw, -1.0, 1.0)

        ang_yaw_n = np.clip(self.safe_val(s.get("AngVelYaw", 0.0)), -1.0, 1.0)
        ang_pitch_n = np.clip(self.safe_val(s.get("AngVelPitch", 0.0)), -1.0, 1.0)

        time_last_action_n = self.safe_val(s.get("TimeLastAction", 3.0))
        
        can_interact_n = float(s.get("CanInteract", 0))
        last_action_n = self.safe_val(s.get("LastAction", 12)) / 12.0 

        forward_x = self.safe_val(s.get("ForwardX", 0.0))
        forward_z = self.safe_val(s.get("ForwardZ", 0.0))

        # --- Normalización de nuevas métricas de Estado e Inventario ---
        health_n = self.safe_val(s.get("Health", 100.0))
        inv_slots_n = self.safe_val(s.get("InventorySlots", 0))
        
        # Municiones escaladas basándose en capacidades lógicas máximas (ej: 120 balas)
        ammo_9x19 = np.clip(self.safe_val(s.get("Ammo9x19", 0.0)), 0.0, 1.0)
        ammo_12g = np.clip(self.safe_val(s.get("Ammo12gauge", 0.0)), 0.0, 1.0)
        ammo_556 = np.clip(self.safe_val(s.get("Ammo556x45", 0.0)), 0.0, 1.0)
        ammo_762 = np.clip(self.safe_val(s.get("Ammo762x39", 0.0)), 0.0, 1.0)
        ammo_44 = np.clip(self.safe_val(s.get("Ammo44cal", 0.0)), 0.0, 1.0)

        role_id  = self.safe_val(s.get("RoleId", 0.0))
        am_i_hurt = float(s.get("AmIHurt", False))
        round_time_remaining = np.clip(self.safe_val(s.get("RoundTimeRemaining", 0.0)) / 600.0, 0.0, 1.0)
        time_since_last_damage = np.clip(self.safe_val(s.get("TimeSinceLastDamage", 0.0)) / 60.0, 0.0, 1.0)

        count_keycards  = np.clip(self.safe_val(s.get("CountKeycards",  0.0)), 0.0, 1.0)
        count_firearms  = np.clip(self.safe_val(s.get("CountFirearms",  0.0)), 0.0, 1.0)
        count_medicals  = np.clip(self.safe_val(s.get("CountMedicals",  0.0)), 0.0, 1.0)
        count_armor     = np.clip(self.safe_val(s.get("CountArmor",     0.0)), 0.0, 1.0)
        count_grenades  = np.clip(self.safe_val(s.get("CountGrenades",  0.0)), 0.0, 1.0)
        count_scp_items = np.clip(self.safe_val(s.get("CountScpItems",  0.0)), 0.0, 1.0)
        count_others    = np.clip(self.safe_val(s.get("CountOthers",    0.0)), 0.0, 1.0)

        count_enemies  = np.clip(self.safe_val(s.get("CountEnemies",  0.0)),  0.0, 1.0)
        count_friends  = np.clip(self.safe_val(s.get("CountFriends",  0.0)),  0.0, 1.0)
        count_neutrals = np.clip(self.safe_val(s.get("CountNeutrals", 0.0)),  0.0, 1.0)
        closet_enemy_dist = self.safe_val(s.get("ClosetEnemyDistance", 1.0))

        base = np.array([
            faction,
            role_id,
            self.safe_val(s.get("RelX")), self.safe_val(s.get("RelY")), self.safe_val(s.get("RelZ")),
            self.safe_val(s.get("GPSX")), self.safe_val(s.get("GPSY")), self.safe_val(s.get("GPSZ")),
            yaw_n, pitch_n,
            aim_enc,
            forward_x, forward_z,
            lin_vel_n, lat_vel_n, ver_vel_n,
            ang_yaw_n, ang_pitch_n,
            last_action_n, time_last_action_n, can_interact_n,
            health_n,
            am_i_hurt,
            float(s.get("HasKeycard", False)),
            self.safe_val(s.get("KeycardTier", 0)) / 9.0,
            aim_dist,
            inv_slots_n,
            ammo_9x19, ammo_12g, ammo_556, ammo_762, ammo_44,
            round_time_remaining,
            time_since_last_damage,
            count_keycards, count_firearms, count_medicals, count_armor,
            count_grenades, count_scp_items, count_others,
            count_enemies, count_friends, count_neutrals, closet_enemy_dist
        ], dtype=np.float32)

        graph_features, graph_adj, graph_mask = self._encode_graph(s)

        vec = np.concatenate([
            base,
            self._encode_danio(s),
            self.enc_rooms(s.get("NearRooms", [])),
            self._encode_whiskers(s),
            self.one_hot(zone_enc, 7),
            self.one_hot(room_enc, 67),
            self.enc_items(s.get("NearItems", [])),
            self.enc_doors(s.get("NearDoors", [])),
            self.enc_lifts(s.get("NearLifts", [])),
            self.enc_lockers(s.get("NearLockers", [])),
            self.enc_players(s.get("NearPlayers", [])),
            graph_features,
            graph_adj,
            graph_mask
        ])

        if vec.shape[0] < VEC_DIM:
            vec = np.concatenate([vec, np.zeros(VEC_DIM - vec.shape[0], dtype=np.float32)])
        return vec

    def safe_val(self, val, default=0.0):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)     
       
    def one_hot(self, index, size):
        vec = np.zeros(size, dtype=np.float32)
        if index < size:
            vec[index] = 1.0
        return vec

        # --- Codificadores de Listas con Persistencia de Memoria y Coordenadas Reales ---
    def _encode_danio(self, s):
        return np.array([
            s.get("DamageReceived", 0.0),
            self.DAMAGE_TYPE_MAP.get(s.get("DamageType", "None"), 0.0) / 5.0,
            s.get("DamageDirX", 0.0),
            s.get("DamageDirZ", 0.0),
            float(s.get("AttackerInMemory", False)),
        ], dtype=np.float32)  # shape: (5,)
        
    def enc_rooms(self, lst):
        # 12 características por habitación activa
        v = np.zeros(N_ROOMS * 12, dtype=np.float32)
        for i, o in enumerate(lst[:N_ROOMS]):
            room_id = np.clip(self.safe_val(o.get("Id", 0.0)) / 67.0, 0.0, 1.0)
            norm_x = np.clip(self.safe_val(o.get("NormX")), -1.0, 1.0)
            norm_y = np.clip(self.safe_val(o.get("NormY")), -1.0, 1.0)
            norm_z = np.clip(self.safe_val(o.get("NormZ")), -1.0, 1.0)
            ubi_x = np.clip(self.safe_val(o.get("UbiX")), -1.0, 1.0)
            ubi_y = np.clip(self.safe_val(o.get("UbiY")), -1.0, 1.0)
            ubi_z = np.clip(self.safe_val(o.get("UbiZ")), -1.0, 1.0)
            prioridad = np.clip(self.safe_val(o.get("Prioridad")), 0.0, 1.0)
            dist_n = np.clip(self.safe_val(o.get("Dist")), 0.0, 1.0)
            es_recordado = float(o.get("EsRecordado", False))
            antiguedad = np.clip(self.safe_val(o.get("Antiguedad")) / 45.0, 0.0, 1.0) # Escalado a 15s max
            slot_existe = 1.0

            inicio = i * 12
            v[inicio : inicio + 12] = [
                room_id, norm_x, norm_y, norm_z, ubi_x, ubi_y, ubi_z,
                prioridad, dist_n, es_recordado, antiguedad, slot_existe
            ]
        return v

    def enc_items(self, lst):
        # SUSTITUCIÓN: Procesa 'NearItems' en lugar de llaves aisladas (12 características por slot)
        # Asegúrate de definir self.N_ITEMS en tu inicialización de Python (antiguo N_KEYCARDS)
        v = np.zeros(N_ITEMS * 9, dtype=np.float32)
        for i, o in enumerate(lst[:N_ITEMS]):
            rx = np.clip(self.safe_val(o.get("RelX")), -1.0, 1.0)
            ry = np.clip(self.safe_val(o.get("RelY")), -1.0, 1.0)
            rz = np.clip(self.safe_val(o.get("RelZ")), -1.0, 1.0)
            #real_rx = np.clip(safe_val(o.get("RealRelX")), -1.0, 1.0)
            #real_ry = np.clip(safe_val(o.get("RealRelY")), -1.0, 1.0)
            #real_rz = np.clip(safe_val(o.get("RealRelZ")), -1.0, 1.0)
            dist_n = np.clip(self.safe_val(o.get("Distance")), 0.0, 1.0)
            prio = np.clip(self.safe_val(o.get("Prioridad")), 0.0, 1.0)
            tier = self.safe_val(o.get("Tier")) / 4.0
            es_recordado = float(o.get("EsRecordado", False))
            antiguedad = np.clip(self.safe_val(o.get("Antiguedad")) / 45.0, 0.0, 1.0)
            slot_existe = 1.0

            inicio = i * 9
            v[inicio : inicio + 9] = [
                rx, ry, rz,
                dist_n, prio, tier, es_recordado, antiguedad, slot_existe
            ]
        return v

    def enc_doors(self, lst):
        # 13 características por puerta activa
        v = np.zeros(N_DOORS * 10, dtype=np.float32)
        for i, o in enumerate(lst[:N_DOORS]):
            rx = np.clip(self.safe_val(o.get("RelX")), -1.0, 1.0)
            ry = np.clip(self.safe_val(o.get("RelY")), -1.0, 1.0)
            rz = np.clip(self.safe_val(o.get("RelZ")), -1.0, 1.0)
            #real_rx = np.clip(safe_val(o.get("RealRelX")), -1.0, 1.0)
            #real_ry = np.clip(safe_val(o.get("RealRelY")), -1.0, 1.0)
            #real_rz = np.clip(safe_val(o.get("RealRelZ")), -1.0, 1.0)
            dist_n = np.clip(self.safe_val(o.get("Distance")), 0.0, 1.0)
            tier = self.safe_val(o.get("RequiredTier", 0)) / 9.0
            can_open = float(o.get("CanOpen", False))
            is_open = float(o.get("IsOpen", False))
            es_recordado = float(o.get("EsRecordado", False))
            antiguedad = np.clip(self.safe_val(o.get("Antiguedad")) / 45.0, 0.0, 1.0)
            slot_existe = 1.0

            inicio = i * 10
            v[inicio : inicio + 10] = [
                rx, ry, rz, dist_n,
                tier, can_open, is_open, es_recordado, antiguedad, slot_existe
            ]
        return v

    def enc_lifts(self, lst):
        v = np.zeros(N_LIFTS * 13, dtype=np.float32)
        for i, o in enumerate(lst[:N_LIFTS]):
            rx = np.clip(self.safe_val(o.get("RelX")), -1.0, 1.0)
            ry = np.clip(self.safe_val(o.get("RelY")), -1.0, 1.0)
            rz = np.clip(self.safe_val(o.get("RelZ")), -1.0, 1.0)
            #real_rx = np.clip(safe_val(o.get("RealRelX")), -1.0, 1.0)
            #real_ry = np.clip(safe_val(o.get("RealRelY")), -1.0, 1.0)
            #real_rz = np.clip(safe_val(o.get("RealRelZ")), -1.0, 1.0)
            dist_n = np.clip(self.safe_val(o.get("Distance")), 0.0, 1.0)
            can_use = float(o.get("CanUse", False))
            is_moving = float(o.get("IsMoving", False))
            is_locked = float(o.get("IsLocked", False))
            is_closed = float(o.get("IsClosed", False))
            level = self.safe_val(o.get("CurrentLevel", 0)) / 3.0
            is_in_elevator = float(o.get("IsInElevator", False))
            es_recordado = float(o.get("EsRecordado", False))
            antiguedad = np.clip(self.safe_val(o.get("Antiguedad")) / 45.0, 0.0, 1.0)
            slot_existe = 1.0

            inicio = i * 13
            v[inicio : inicio + 13] = [
                rx, ry, rz, dist_n,
                can_use, is_moving, is_locked, is_closed, level,
                is_in_elevator,
                es_recordado, antiguedad, slot_existe
            ]
        return v

    def enc_lockers(self, lst):
        # 12 características por locker activo
        v = np.zeros(N_LOCKERS * 9, dtype=np.float32)
        for i, o in enumerate(lst[:N_LOCKERS]):
            rx = np.clip(self.safe_val(o.get("RelX")), -1.0, 1.0)
            ry = np.clip(self.safe_val(o.get("RelY")), -1.0, 1.0)
            rz = np.clip(self.safe_val(o.get("RelZ")), -1.0, 1.0)
            #real_rx = np.clip(safe_val(o.get("RealRelX")), -1.0, 1.0)
            #real_ry = np.clip(safe_val(o.get("RealRelY")), -1.0, 1.0)
            #real_rz = np.clip(safe_val(o.get("RealRelZ")), -1.0, 1.0)
            dist_n = np.clip(self.safe_val(o.get("Distance")), 0.0, 1.0)
            has_is_open = float(o.get("HasIsOpen", False))
            is_open = float(o.get("IsOpen", False))
            es_recordado = float(o.get("EsRecordado", False))
            antiguedad = np.clip(self.safe_val(o.get("Antiguedad")) / 45.0, 0.0, 1.0)
            slot_existe = 1.0

            inicio = i * 9
            v[inicio : inicio + 9] = [
                rx, ry, rz, dist_n,
                has_is_open, is_open, es_recordado, antiguedad, slot_existe                
            ]
        return v

    def enc_players(self, lst):
        # EXTRA: Codificador para la lista NearPlayers (11 características por enemigo/aliado)
        v = np.zeros(N_PLAYERS * 11, dtype=np.float32)
        for i, o in enumerate(lst[:N_PLAYERS]):
            rx = np.clip(self.safe_val(o.get("RelX")), -1.0, 1.0)
            ry = np.clip(self.safe_val(o.get("RelY")), -1.0, 1.0)
            rz = np.clip(self.safe_val(o.get("RelZ")), -1.0, 1.0)
            dist_n = np.clip(self.safe_val(o.get("Distance")), 0.0, 1.0)
            f_id = self.safe_val(o.get("FactionId"))
            hostilidad = self.safe_val(o.get("Hostilidad"))
            hp = self.safe_val(o.get("Health", 100.0))
            mirada = self.safe_val(o.get("MiradaHaciaMi"))
            es_recordado = float(o.get("EsRecordado", False))
            antiguedad = np.clip(self.safe_val(o.get("Antiguedad")) / 45.0, 0.0, 1.0)
            slot_existe = 1.0

            inicio = i * 11
            v[inicio : inicio + 11] = [
                rx, ry, rz, dist_n, f_id, hostilidad, hp, mirada,
                es_recordado, antiguedad, slot_existe
            ]
        return v
        
    def _encode_whiskers(self, s):
        dist = s.get("WhiskerDist", [1.0] * 8)
        tipo = s.get("WhiskerType", [0.0] * 8)
        vec = np.zeros(16, dtype=np.float32)
        for i in range(8):
            vec[i*2]   = dist[i] if i < len(dist) else 1.0
            vec[i*2+1] = tipo[i] if i < len(tipo) else 0.0
        return vec

    def _encode_graph(self, s):
        graph_nodes = s.get("GraphNodes", [])
        graph_adj   = s.get("GraphAdjacency", [])
        graph_mask  = s.get("GraphMask", [])

        mask = np.zeros(N_GRAPH_MASK, dtype=np.float32)
        for i in range(min(len(graph_mask), N_GRAPH_MASK)):
            mask[i] = self.safe_val(graph_mask[i])

        features = np.zeros((N_GRAPH_NODES, N_GRAPH_FEAT), dtype=np.float32)
        for i in range(min(len(graph_nodes), N_GRAPH_NODES)):
            n = graph_nodes[i]
            features[i] = [
                #np.clip(safe_val(n.get("Id", 0))         / 1_000_000.0  , 0.0, 1.0),
                np.clip(self.safe_val(n.get("TypeId", 0))     / 100.0 ,        0.0, 1.0),
                np.clip(self.safe_val(n.get("RelX")), -1.0, 1.0),
                np.clip(self.safe_val(n.get("RelY")), -1.0, 1.0),
                np.clip(self.safe_val(n.get("RelZ")), -1.0, 1.0),
                #np.clip(safe_val(n.get("PosX")) / 500.0,   -1.0, 1.0),
                #np.clip(safe_val(n.get("PosY")) / 500.0,   -1.0, 1.0),
                #np.clip(safe_val(n.get("PosZ")) / 500.0,   -1.0, 1.0),
                np.clip(self.safe_val(n.get("Prioridad")),       0.0, 1.0),
                #np.clip(safe_val(n.get("Distancia")) / 500.0, 0.0, 1.0),
                np.clip(self.safe_val(n.get("DistNorm")),        0.0, 1.0),
                np.clip(self.safe_val(n.get("VisitCount")) / 20.0, 0.0, 1.0),
                np.clip(self.safe_val(n.get("Antiguedad")) / 60.0, 0.0, 1.0),
                np.clip(self.safe_val(n.get("EsActual")),        0.0, 1.0),
                np.clip(self.safe_val(n.get("TieneEnemigo")),    0.0, 1.0),
                np.clip(self.safe_val(n.get("TieneLoot")),       0.0, 1.0),
                np.clip(self.safe_val(n.get("PuertaBloq")),      0.0, 1.0),
            ]

        adj_flat = np.zeros(N_GRAPH_ADJ, dtype=np.float32)
        for i in range(min(len(graph_adj), N_GRAPH_ADJ)):
            adj_flat[i] = self.safe_val(graph_adj[i])
        adj = adj_flat.reshape(N_GRAPH_NODES, N_GRAPH_NODES)

        features = features * mask[:, None]
        adj = adj * np.outer(mask, mask)

        return features.flatten(), adj.flatten(), mask