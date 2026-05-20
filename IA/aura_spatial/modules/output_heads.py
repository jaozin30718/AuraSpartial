"""
modules/output_heads.py
=======================
MÓDULO 5: Cabeçotes de Saída Multitarefa

Três heads independentes alimentadas pelo estado oculto do Mamba:

    1. BSSHead      → Separação de Fontes: [B, L, D] → [B, S, T, F]
                      Máscaras espectrais por fonte (cIRM ou real mask)

    2. DoAHead      → Direção de Chegada: [B, L, D] → [B, S, 2]
                      Azimute e elevação por fonte (em graus)

    3. HeatmapHead  → Mapa Acústico: [B, L, D] → [B, Rx, Ry]
                      Mapa de probabilidade 2D de energia sonora
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from ..config import OutputHeadsConfig, PatchEmbedderConfig


class BSSHead(nn.Module):
    """
    Head de Separação Cega de Fontes (BSS).

    Gera máscaras espectrais complexas (cIRM) ou reais por fonte,
    que serão multiplicadas pelo espectrograma de mistura para obter
    os espectrogramas de cada fonte isolada.

    Estratégia:
        1. Projeta [B, L, D] → [B, L, S*2*F_patch] via Linear
        2. Reorganiza para [B, S, T_patches, F_patches*2]
        3. Deconvolve (PixelShuffle ou ConvTranspose2d) para [B, S, T, F]
        4. Ativação: Sigmoid (máscara real ∈ [0,1]) ou Tanh (cIRM ∈ [-1,1])

    A máscara complexa (cIRM) tem parte real e imaginária:
        [B, S*2, T, F] onde S*2 = {Re_mask₁, Im_mask₁, Re_mask₂, Im_mask₂}

    Input : [B, L, D]
    Output: [B, S, T, F] (real mask) ou [B, S*2, T, F] (cIRM)
    """

    def __init__(
        self,
        config:      OutputHeadsConfig,
        patch_config: PatchEmbedderConfig,
    ) -> None:
        super().__init__()

        self.config       = config
        self.patch_config = patch_config

        D  = config.embed_dim
        S  = config.n_sources
        pt = patch_config.patch_time
        pf = patch_config.patch_freq
        nt = patch_config.n_patches_time
        nf = patch_config.n_patches_freq

        # Para cIRM: 2 canais por fonte (real + imaginário)
        is_cirm = (config.mask_activation == "tanh")
        self.n_mask_channels = S * 2 if is_cirm else S

        # ── Projeção Linear D → S * pt * pf ──────────────────────────────────
        # Projeta cada token para um "patch de máscara" por fonte
        # Output por token: S * pt * pf valores de máscara
        self.proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, self.n_mask_channels * pt * pf, bias=True),
        )

        # ── ConvTranspose2d para upsample fino (opcional refinamento) ──────────
        # Após o reshape inicial para [B, n_mask_channels, T, F],
        # aplica convolução para refinamento espacial sem alterar shape
        self.refine_conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = self.n_mask_channels,
                out_channels = self.n_mask_channels * 4,
                kernel_size  = 3,
                padding      = 1,
                groups       = self.n_mask_channels,   # depthwise
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels  = self.n_mask_channels * 4,
                out_channels = self.n_mask_channels,
                kernel_size  = 1,
            ),
        )

        # Ativação final da máscara
        if config.mask_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif config.mask_activation == "tanh":
            # cIRM Compression (Williamson et al.) em vez de Tanh puro
            self.activation = lambda x: 10 * (1 - torch.exp(-0.1 * x)) / (1 + torch.exp(-0.1 * x))
        else:
            self.activation = nn.Identity()

        # Armazena shapes para uso no forward
        self.nt = nt   # número de patches temporais
        self.nf = nf   # número de patches de frequência
        self.pt = pt   # tamanho do patch temporal
        self.pf = pf   # tamanho do patch de frequência
        self.S  = S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] — estado oculto do Mamba (L = nt * nf)

        Returns:
            masks: [B, S, T, F] — máscaras espectrais por fonte
                   Onde T = nt*pt, F = nf*pf
                   Valores: [0,1] (sigmoid) ou [-1,1] (tanh/cIRM)
        """
        B, L, D = x.shape
        S  = self.S
        C  = self.n_mask_channels  # S ou S*2
        nt = self.nt
        nf = self.nf
        pt = self.pt
        pf = self.pf

        # ── 1. Projeção: [B, L, D] → [B, L, C*pt*pf] ─────────────────────────
        x = self.proj(x)   # [B, L, C*pt*pf]

        # ── 2. Reorganização para grade de patches ────────────────────────────
        # [B, L, C*pt*pf] → [B, nt*nf, C, pt, pf]
        x = rearrange(
            x,
            "b (nt nf) (c pt pf) -> b c (nt pt) (nf pf)",
            nt=nt, nf=nf, c=C, pt=pt, pf=pf,
        )  # [B, C, T, F]

        # ── 3. Refinamento convolucional ──────────────────────────────────────
        x = self.refine_conv(x)   # [B, C, T, F]

        # ── 4. Separação em fontes e ativação ────────────────────────────────
        # [B, C, T, F] onde C = S ou S*2
        masks = self.activation(x)   # [B, S ou S*2, T, F]

        return masks   # [B, S, T, F] ou [B, S*2, T, F]


class DoAHead(nn.Module):
    """
    Head de Direção de Chegada (DoA).

    MLP que colapsa a sequência temporal em coordenadas angulares
    por fonte: (azimute, elevação) em graus.

    Estratégia de pooling:
        - Média ponderada sobre patches temporais (global average pooling)
        - Seguido de MLP profundo com skip connections
        - Saída sem ativação (regressão angular no domínio real)

    Input : [B, L, D]
    Output: [B, S, 2]  — (azimute [0°,360°], elevação [-90°,90°])
    """

    def __init__(self, config: OutputHeadsConfig) -> None:
        super().__init__()

        D = config.embed_dim
        S = config.n_sources
        H = config.doa_hidden_dim

        self.S = S

        # ── Pooling Atencional (aprendível) ───────────────────────────────────
        # Em vez de média simples, aprende quais tokens são mais informativos
        # para DoA. Vetor de atenção: [D] → score escalar por token
        self.attn_pool = nn.Linear(D, 1, bias=False)

        # ── MLP de Regressão ──────────────────────────────────────────────────
        # Input: [B, D] → Output: [B, S*2]
        self.mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, H),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, S * 2),
            nn.Tanh()  # <--- ADICIONAR ESTA LINHA
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] — estado oculto do Mamba

        Returns:
            doa: [B, S, 2] — (azimute, elevação) por fonte em graus
                 Azimute  : valor real sem restrição (pode aplicar módulo 360 externamente)
                 Elevação : valor real sem restrição (pode clipar a [-90, 90])
        """
        B, L, D = x.shape

        # ── 1. Pooling Atencional: [B, L, D] → [B, D] ────────────────────────
        # Calcula scores de atenção: [B, L, 1]
        attn_scores = self.attn_pool(x)           # [B, L, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, L, 1] — softmax sobre L

        # Média ponderada: Σ w_l * x_l → [B, D]
        x_pooled = (attn_weights * x).sum(dim=1)  # [B, D]

        # ── 2. MLP de regressão: [B, D] → [B, S*2] ───────────────────────────
        doa_flat = self.mlp(x_pooled)             # [B, S*2]

        # ── 3. Reshape para [B, S, 2] ─────────────────────────────────────────
        doa = rearrange(doa_flat, "b (s two) -> b s two", s=self.S, two=2)
        
        # Escala física: Azimute[-180, 180], Elevação [-90, 90]
        azimute_scaled = doa[..., 0] * 180.0
        elevacao_scaled = doa[..., 1] * 90.0
        return torch.stack([azimute_scaled, elevacao_scaled], dim=-1)


class HeatmapHead(nn.Module):
    """
    Head de Mapa Térmico Acústico Espacial.

    Transforma os tokens latentes em um mapa 2D de probabilidade de
    presença de energia sonora por região angular.

    Estratégia:
        1. Reduz L tokens → grid compacto via reshape
        2. Aplica upsampling progressivo com ConvTranspose2d
        3. Saída com Sigmoid → probabilidade ∈ [0, 1]

    Input : [B, L, D]
    Output: [B, 1, Rx, Ry] — mapa de calor 2D (1 canal de probabilidade)

    Onde Rx = heatmap_res_x (bins de azimute) e Ry = heatmap_res_y (bins de elevação)
    """

    def __init__(
        self,
        config:       OutputHeadsConfig,
        patch_config: PatchEmbedderConfig,
    ) -> None:
        super().__init__()

        D  = config.embed_dim
        Rx = config.heatmap_res_x
        Ry = config.heatmap_res_y
        nt = patch_config.n_patches_time
        nf = patch_config.n_patches_freq

        self.nt = nt
        self.nf = nf
        self.Rx = Rx
        self.Ry = Ry

        # ── Projeção de compressão de canais ──────────────────────────────────
        # D → C_start (número de canais iniciais para a rede de upsampling)
        C_start = 128
        self.proj_in = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, C_start),
            nn.GELU(),
        )

        # ── Rede de Upsampling Convolucional ──────────────────────────────────
        # Parte de [B, C_start, nt, nf] e progressivamente faz upsample
        # até alcançar [B, 1, Rx, Ry]
        #
        # ConvTranspose2d(in, out, kernel=4, stride=2, padding=1)
        # → duplica resolução espacial a cada bloco
        self.upsample_net = nn.Sequential(
            # Bloco 1: [B, 128, nt, nf] → [B, 64, nt*2, nf*2]
            nn.ConvTranspose2d(C_start, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Bloco 2: [B, 64, nt*2, nf*2] → [B, 32, nt*4, nf*4]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Bloco 3: [B, 32, ...] → [B, 1, ...]
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # Adaptive pooling final para garantir resolução exata [Rx, Ry]
        # independente das dimensões intermediárias
        self.adaptive_pool = nn.AdaptiveAvgPool2d((Rx, Ry))

        # NOTA: Sigmoid REMOVIDO — a head retorna logits crus.
        # BCEWithLogitsLoss aplica sigmoid internamente com maior estabilidade numérica.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] — estado oculto do Mamba (L = nt * nf)

        Returns:
            heatmap: [B, 1, Rx, Ry] — logits crus (pré-sigmoid)
                     Usar BCEWithLogitsLoss na loss function (NÃO aplicar sigmoid aqui).
        """
        B, L, D = x.shape

        # ── 1. Projeção: [B, L, D] → [B, L, C_start] ─────────────────────────
        x = self.proj_in(x)   # [B, L, C_start]

        # ── 2. Reshape para grid 2D: [B, L, C] → [B, C, nt, nf] ─────────────
        x = rearrange(
            x,
            "b (nt nf) c -> b c nt nf",
            nt=self.nt, nf=self.nf,
        )  # [B, C_start, nt, nf]

        # ── 3. Upsampling progressivo ─────────────────────────────────────────
        x = self.upsample_net(x)   # [B, 1, nt*4, nf*4] (aprox.)

        # ── 4. Pooling adaptativo para resolução exata [Rx, Ry] ───────────────
        x = self.adaptive_pool(x)  # [B, 1, Rx, Ry]

        # ── 5. Retorna logits crus (SEM sigmoid) ──────────────────────────────
        return x   # [B, 1, Rx, Ry] — logits para BCEWithLogitsLoss