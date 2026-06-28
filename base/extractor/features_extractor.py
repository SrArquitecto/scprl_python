import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# 🛠️ AÑADIDO: Importamos VecNormalize junto a los otros wrappers de vectores


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