"""
modules/jepa_encoders.py
========================
MÓDULO 2: Codificadores LeJEPA

Implementa E_ctx (codificador de contexto, recebe tokens mascarados)
e E_tgt (codificador alvo, recebe tokens limpos).

Design LeJEPA sem EMA:
  - E_tgt compartilha os MESMOS pesos que E_ctx (mesma instância)
  - Isso economiza ~50% de VRAM comparado a abordagens BYOL-style
  - O gradiente flui através de AMBOS os codificadores simultaneamente
  - A regularização VICReg (L_var + L_cov) previne colapso sem amostras negativas

Arquitetura base: Transformer enxuto com:
  - Pre-LayerNorm (mais estável que post-LN para treino profundo)
  - Atenção multi-cabeça padrão via nn.MultiheadAttention
  - MLP com GELU e dropout
  - Conexões residuais
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..config import EncoderConfig


class TransformerBlock(nn.Module):
    """
    Bloco Transformer com Pre-LayerNorm.

    Input/Output: [B, L, D]
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        D = config.embed_dim

        # ── Self-Attention ────────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(D)
        self.attn  = nn.MultiheadAttention(
            embed_dim    = D,
            num_heads    = config.num_heads,
            dropout      = config.attn_dropout,
            batch_first  = True,    # espera [B, L, D] — consistente com o resto
        )

        # ── Feed-Forward MLP ──────────────────────────────────────────────────
        self.norm2 = nn.LayerNorm(D)
        hidden_dim = int(D * config.mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, D),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            x: [B, L, D]
        """
        # Pre-norm + self-attention + residual
        residual = x
        x_norm   = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_out

        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))

        return x  # [B, L, D]


class JEPAEncoder(nn.Module):
    """
    Codificador base compartilhado para E_ctx e E_tgt.

    No LeJEPA sem EMA:
      - E_ctx e E_tgt são a MESMA instância deste módulo
      - Diferença está apenas no INPUT fornecido:
          * E_ctx recebe tokens COM patches mascarados
          * E_tgt recebe tokens SEM máscara (patches originais)

    Fluxo:
        [B, L, D] → N × TransformerBlock → LayerNorm → [B, L, D]

    Args:
        config: EncoderConfig
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Pilha de blocos Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # LayerNorm final (pós todos os blocos)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, L, D] — sequência de patches (mascarados ou limpos)

        Returns:
            z: [B, L, D] — representações latentes codificadas
        """
        x = tokens

        for block in self.blocks:
            x = block(x)   # [B, L, D] → [B, L, D]

        z = self.norm(x)   # [B, L, D]

        return z           # [B, L, D]

    def forward_context(self, tokens_masked: torch.Tensor) -> torch.Tensor:
        """
        Alias semântico para E_ctx: processa tokens COM máscara.

        Args:
            tokens_masked: [B, L, D] — tokens onde patches mascarados
                           foram substituídos pelo mask_token aprendível

        Returns:
            z_ctx: [B, L, D]
        """
        return self.forward(tokens_masked)

    def forward_target(self, tokens_clean: torch.Tensor) -> torch.Tensor:
        """
        Alias semântico para E_tgt: processa tokens SEM máscara.
        Como compartilha pesos com E_ctx, a distinção é apenas semântica.

        Args:
            tokens_clean: [B, L, D] — tokens originais sem substituição

        Returns:
            z_tgt: [B, L, D]
        """
        return self.forward(tokens_clean)