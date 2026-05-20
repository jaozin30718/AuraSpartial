"""
modules/patch_embedder.py
=========================
MÓDULO 1: Input Embedding e Patching

Converte o tensor de entrada DSP [B, C, T, F] em uma sequência
de embeddings latentes [B, L, D], onde:
  B = batch size
  C = 5 canais DSP (Mag_L, Mag_R, cos_IPD, sin_IPD, GCC-PHAT)
  T = frames temporais
  F = bins de frequência
  L = número total de patches (T//pt * F//pf)
  D = dimensão do embedding

A abordagem usa nn.Conv2d com stride=kernel_size para patches
não sobrepostos (equivalente a ViT-style patch extraction),
tornando a operação altamente otimizada em CUDA via GEMM implícito.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from ..config import PatchEmbedderConfig


class PatchEmbedder(nn.Module):
    """
    Converte entrada DSP espacial 4D em sequência de tokens latentes.

    Fluxo:
        [B, C, T, F]
          → Conv2d(stride=patch_size) →  [B, D, T//pt, F//pf]
          → rearrange                 →  [B, L, D]      (L = T//pt * F//pf)
          → + pos_embed               →  [B, L, D]

    Args:
        config: PatchEmbedderConfig com hiperparâmetros do módulo.
    """

    def __init__(self, config: PatchEmbedderConfig) -> None:
        super().__init__()
        self.config = config

        # ── Extração de Patches via Conv2d ────────────────────────────────────
        # Conv2d com kernel_size = stride → patches NÃO sobrepostos.
        # Equivale a: particionar [T, F] em grades regulares e achatar.
        # Output: [B, embed_dim, n_patches_time, n_patches_freq]
        self.patch_conv = nn.Conv2d(
            in_channels  = config.in_channels,
            out_channels = config.embed_dim,
            kernel_size  = (config.patch_time, config.patch_freq),
            stride       = (config.patch_time, config.patch_freq),
            bias         = True,
        )

        # Alternativa explícita com einops para máxima clareza dimensional:
        # (usada no forward para documentar o reshape)
        # Garante que a dimensão mais interna seja o Tempo (nt), para que 
        # a sequência L = (nf * nt) possua blocos contíguos no tempo para o Mamba.
        self.to_sequence = Rearrange(
            "b d nf nt -> b (nf nt) d",
            # b=batch, d=embed_dim, nt=patches_time, nf=patches_freq
        )

        # ── Layer Norm pós-projeção ───────────────────────────────────────────
        self.norm = nn.LayerNorm(config.embed_dim)

        # ── Positional Embedding 2D Aprendível ───────────────────────────────
        # Dois embeddings separados (tempo e frequência) somados ao token.
        # Isso captura estrutura 2D de forma mais eficiente que um único
        # embedding 1D plano — inspirado em CPVT e Swin Transformer.
        #
        # Shape: (1, n_patches_time, 1, D) + (1, 1, n_patches_freq, D)
        # → broadcast → (1, n_patches_time, n_patches_freq, D)
        # → flatten  → (1, L, D)
        self.pos_embed_time = nn.Parameter(
            torch.zeros(1, config.n_patches_time, 1, config.embed_dim)
        )
        self.pos_embed_freq = nn.Parameter(
            torch.zeros(1, 1, config.n_patches_freq, config.embed_dim)
        )

        # Token de máscara aprendível — substituirá patches mascarados no E_ctx
        # Shape: (D,) → expandido para (1, 1, D) durante forward
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialização de pesos seguindo ViT (truncated normal)."""
        nn.init.trunc_normal_(self.pos_embed_time, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_freq, std=0.02)
        nn.init.trunc_normal_(self.mask_token,     std=0.02)
        nn.init.xavier_uniform_(self.patch_conv.weight)
        if self.patch_conv.bias is not None:
            nn.init.zeros_(self.patch_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x    : [B, C, T, F] — tensor de entrada DSP
            mask : [B, L] bool — True indica patches a mascarar (opcional)

        Returns:
            tokens    : [B, L, D] — sequência de tokens (com máscara aplicada se fornecida)
            pos_embed : [B, L, D] — embeddings posicionais (para uso externo no preditor)

        Notas dimensionais:
            C  = 5 canais DSP
            T  = n_time_frames (ex: 128)
            F  = n_freq_bins   (ex: 256)
            pt = patch_time    (ex: 8)  → nt = T/pt = 16
            pf = patch_freq    (ex: 16) → nf = F/pf = 16
            L  = nt * nf       (ex: 256 patches)
            D  = embed_dim     (ex: 512)
        """
        cfg = self.config
        B   = x.shape[0]

        # ── 1. Extração de patches via Conv2d ─────────────────────────────────
        # [B, C, T, F] → [B, D, nt, nf]
        tokens = self.patch_conv(x)

        # ── 2. Positional Embedding 2D ────────────────────────────────────────
        # pos_time: [1, nt, 1, D] + pos_freq: [1, 1, nf, D]
        # → broadcast: [1, nt, nf, D]
        pos_2d = self.pos_embed_time + self.pos_embed_freq
        # [1, nt, nf, D] → [1, D, nt, nf] para somar com tokens
        pos_2d_conv = rearrange(pos_2d, "1 nt nf d -> 1 d nt nf")
        tokens = tokens + pos_2d_conv

        # ── 3. Flatten para sequência ─────────────────────────────────────────
        # [B, D, nt, nf] → [B, D, nf, nt] → [B, L, D]   (L = nf * nt)
        tokens = rearrange(tokens, "b d nt nf -> b d nf nt")
        tokens = self.to_sequence(tokens)
        tokens = self.norm(tokens)

        # Positional embed plano para retornar ao preditor
        pos_embed_flat = rearrange(
            pos_2d.expand(B, -1, -1, -1),
            "b nt nf d -> b (nf nt) d",
        )  # [B, L, D]

        # ── 4. Aplicar máscara (substituição por mask_token) ──────────────────
        # mask: [B, L] bool — True = mascarar este patch
        if mask is not None:
            # Expande mask_token: [1, 1, D] → [B, L, D]
            mask_tokens = repeat(self.mask_token, "1 1 d -> b l d", b=B, l=tokens.shape[1])

            # Onde mask=True, substitui pelo token de máscara
            # mask[..., None]: [B, L] → [B, L, 1] → broadcast para [B, L, D]
            tokens = torch.where(mask.unsqueeze(-1), mask_tokens, tokens)

        return tokens, pos_embed_flat  # [B, L, D], [B, L, D]

    def generate_random_mask(
        self,
        batch_size: int,
        device: torch.device,
        mask_ratio: float = 0.75,
    ) -> torch.Tensor:
        """
        Gera máscara aleatória de patches.

        Returns:
            mask: [B, L] bool — True = patch mascarado
        """
        L     = self.config.n_patches_total
        n_mask = int(L * mask_ratio)

        # Amostra índices a mascarar para cada amostra do batch
        noise = torch.rand(batch_size, L, device=device)
        ids_sorted = torch.argsort(noise, dim=1)

        mask = torch.zeros(batch_size, L, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_sorted[:, :n_mask], True)

        return mask  # [B, L]