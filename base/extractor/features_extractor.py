import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# 🛠️ AÑADIDO: Importamos VecNormalize junto a los otros wrappers de vectores


class SCPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extractor multi-rol con CTDE aproximado.
 
    Arquitectura:
    ┌─────────────────────────────────────────────────────┐
    │  TRUNK COMPARTIDO (todos los roles)                 │
    │  spatial_mlp → temporal_conv → trunk  →  256 dim   │
    │  Aprende: navegación, percepción, física            │
    └──────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┼──────────────┐
        ▼          ▼              ▼
    head_surv  head_combat    head_scp
    (ClassD,   (NTF, Chaos,   (SCP-049,
    Scientist)  Guard)         096, 173...)
        └──────────┴──────────────┘
                   │  256 dim → actor_feat
                   │
    ┌──────────────▼──────────────────────┐
    │  CRÍTICO CTDE: trunk + NearPlayers  │
    │  players_encoder → 64 dim           │
    │  critic_head(320) → 256 dim         │
    └─────────────────────────────────────┘
                   │  256 dim → critic_feat
 
    Output: [actor_feat | critic_feat] → 512 dim
    SB3 usa [:256] para pi y [256:] para vf.
 
    El rol se lee de obs en cada forward — el cambio de rol
    tras respawn es transparente, activa otra cabeza sin
    reiniciar el modelo.
 
    Curriculum:
      Fase 0-1: solo ClassD → trunk + head_surv aprenden
      Fase 2:   NTF/Chaos → head_combat (transfer desde head_surv)
      Fase 3:   SCPs → head_scp (trunk ya maduro)
    """
 
    # Rangos de RoleId normalizado para identificar el grupo
    # Ajustar si cambia la normalización de RoleId en C# (RoleId/MAX_ROLE)
    _ROLE_SURV_MAX   = 0.10   # ClassD~0.00, Scientist~0.03
    _ROLE_COMBAT_MAX = 0.50   # Guard~0.12, NTF~0.2-0.4, Chaos~0.4-0.5
    # > 0.50 → SCP
 
    # RoleId está en posición 1 del bloque base (base[0]=FactionId, base[1]=RoleId)
    _ROLE_ID_OFFSET = 1
 
    # NearPlayers: base(45)+danio(5)+rooms(60)+whiskers(16)+zones(7)
    #              +rooms_oh(67)+items(45)+doors(150)+lifts(39)+lockers(45) = 479
    _PLAYERS_OFFSET = 479
    _N_PLAYERS      = 5
    _N_PLAYERS_FEAT = 11
 
    def __init__(self, observation_space: spaces.Box):
        total_dim = observation_space.shape[0]
        self.num_frames = 5
        self.frame_dim  = total_dim // self.num_frames
 
        super().__init__(observation_space, features_dim=512)
 
        # ── Trunk compartido ───────────────────────────────────────────────
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
            conv_out_dim = self.temporal_conv(
                torch.zeros(1, 128, self.num_frames)
            ).shape[1]  # 64*5 = 320
 
        self.trunk = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
 
        # ── Cabezas de actor por grupo de rol ─────────────────────────────
        # Supervivencia: ClassD, Scientist
        # Aprende: esconderse, coger keycards, escapar
        self.head_surv = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
 
        # Combate: FacilityGuard, NTF*, Chaos*
        # Aprende: flanquear, disparar, coordinar
        # Se inicializa desde head_surv via transfer_combat_from_surv()
        self.head_combat = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
 
        # SCP: Scp049, Scp096, Scp173, Scp106, Scp939, Scp079
        # Curriculum 2 — presente pero inactivo hasta fase SCP
        self.head_scp = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
 
        # ── Crítico CTDE: trunk + NearPlayers ─────────────────────────────
        players_dim = self._N_PLAYERS * self._N_PLAYERS_FEAT  # 55
        self.players_encoder = nn.Sequential(
            nn.Linear(players_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),          nn.ReLU(),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

    def _role_masks(self, role_ids: torch.Tensor):
        """Devuelve tres máscaras booleanas (B,) según el RoleId normalizado."""
        m_surv   = role_ids <= self._ROLE_SURV_MAX
        m_combat = (role_ids > self._ROLE_SURV_MAX) & (role_ids <= self._ROLE_COMBAT_MAX)
        m_scp    = role_ids > self._ROLE_COMBAT_MAX
        return m_surv, m_combat, m_scp
 
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]
 
        # ── Trunk compartido (todos los frames) ───────────────────────────
        x = observations.view(b, self.num_frames, self.frame_dim)  # (B, 5, F)
        x = self.spatial_mlp(x)                                    # (B, 5, 128)
        x = x.permute(0, 2, 1)                                     # (B, 128, 5)
        x = self.temporal_conv(x)                                   # (B, 320)
        trunk_feat = self.trunk(x)                                  # (B, 256)
 
        # ── Cabeza de actor según rol ──────────────────────────────────────
        last_frame = observations.view(b, self.num_frames, self.frame_dim)[:, -1, :]
        role_ids   = last_frame[:, self._ROLE_ID_OFFSET]            # (B,)
        m_surv, m_combat, m_scp = self._role_masks(role_ids)
 
        actor_feat = torch.zeros(b, 256, device=observations.device)
        if m_surv.any():
            actor_feat[m_surv]   = self.head_surv(trunk_feat[m_surv])
        if m_combat.any():
            actor_feat[m_combat] = self.head_combat(trunk_feat[m_combat])
        if m_scp.any():
            actor_feat[m_scp]    = self.head_scp(trunk_feat[m_scp])
 
        # ── Crítico CTDE ───────────────────────────────────────────────────
        players_raw  = last_frame[
            :, self._PLAYERS_OFFSET : self._PLAYERS_OFFSET + self._N_PLAYERS * self._N_PLAYERS_FEAT
        ]                                                           # (B, 55)
        players_feat = self.players_encoder(players_raw)            # (B, 64)
        critic_feat  = self.critic_head(
            torch.cat([trunk_feat, players_feat], dim=1)
        )                                                           # (B, 256)
 
        return torch.cat([actor_feat, critic_feat], dim=1)          # (B, 512)
 
    def transfer_combat_from_surv(self):
        """
        Transfer learning: inicializa head_combat desde head_surv.
        Llamar al inicio de la fase de combate — NTF/Chaos parten de
        una política de supervivencia ya entrenada en lugar de aleatoria.
 
        Uso:
            model.policy.features_extractor.transfer_combat_from_surv()
        """
        self.head_combat.load_state_dict(self.head_surv.state_dict())
        print("✅ head_combat inicializada desde head_surv (transfer learning)")
 
    def freeze_trunk(self, freeze: bool = True):
        """
        Congela/descongela el trunk compartido.
        Congelar al inicio de una nueva fase para que la cabeza nueva
        aprenda sin distorsionar representaciones ya aprendidas.
        Descongelar tras ~50k steps cuando la cabeza se estabilice.
 
        Uso:
            extractor = model.policy.features_extractor
            extractor.freeze_trunk(True)   # congelar al inicio de fase 2
            # ... 50k steps ...
            extractor.freeze_trunk(False)  # descongelar para refinamiento
        """
        for module in [self.spatial_mlp, self.temporal_conv, self.trunk]:
            for param in module.parameters():
                param.requires_grad = not freeze
        print(f"{'🧊 Trunk congelado' if freeze else '🔥 Trunk descongelado'}")
