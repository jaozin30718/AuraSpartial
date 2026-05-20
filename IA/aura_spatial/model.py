"""
model.py
========
AuraSpatialModel: Invólucro principal do sistema híbrido LeJEPA + Mamba.

Integra todos os módulos em um único nn.Module com dois modos de operação:

    Modo 1 — PRETRAINING (auto-supervisionado):
        Input:  [B, C, T, F]
        Output: (z_pred, z_target, mask) para cálculo de LeJEPALoss

    Modo 2 — INFERENCE / FINE-TUNING (multitarefa):
        Input:  [B, C, T, F]
        Output: {
            "masks"   : [B, S, T, F]      — BSS
            "doa"     : [B, S, 2]         — DoA
            "heatmap" : [B, 1, Rx, Ry]   — Mapa Acústico
        }

Fluxo completo (modo inference):
    x [B, C, T, F]
      → PatchEmbedder (clean)  → tokens_clean [B, L, D]
      → JEPAEncoder (E_tgt)    → z_clean [B, L, D]
      → MambaPredictor         → z_pred [B, L, D]
      → BSSHead                → masks [B, S, T, F]
      → DoAHead                → doa [B, S, 2]
      → HeatmapHead            → heatmap [B, 1, Rx, Ry]
"""

from __future__ import annotations

from typing import Optional

import copy
import torch
import torch.nn as nn
from einops import rearrange

from .config import AuraSpatialConfig
from .modules.patch_embedder import PatchEmbedder
from .modules.jepa_encoders import JEPAEncoder
from .modules.mamba_predictor import MambaPredictor
# Usa a versão robusta da loss definida no diretório de treinamento
from aura_training.losses.jepa_loss import LeJEPALoss
from .modules.output_heads import BSSHead, DoAHead, HeatmapHead


class AuraSpatialModel(nn.Module):
    """
    Modelo híbrido completo para áudio espacial: LeJEPA + Mamba + Multitask.

    Parâmetros estimados (D=512, 6L encoder, 4L Mamba):
        PatchEmbedder : ~1.5M params
        JEPAEncoder   : ~25M params
        MambaPredictor: ~4M params
        OutputHeads   : ~5M params
        Total         : ~35M params

    Args:
        config: AuraSpatialConfig — configuração completa do modelo
    """

    def __init__(self, config: AuraSpatialConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedder = PatchEmbedder(config.patch_embedder)
        
        # Context Encoder (Treinável)
        self.encoder = JEPAEncoder(config.encoder)
        
        # Target Encoder (EMA - Sem gradientes)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_decay = 0.996
        
        self.predictor = MambaPredictor(config.predictor)
        self.loss_fn = LeJEPALoss(config.loss)
        self.bss_head = BSSHead(config.heads, config.patch_embedder)
        self.doa_head = DoAHead(config.heads)
        self.heatmap_head = HeatmapHead(config.heads, config.patch_embedder)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Inicialização padrão para camadas Linear e LayerNorm."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_target_encoder(self):
        """Atualiza o EMA do Target Encoder."""
        for param_ctx, param_tgt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_tgt.data = param_tgt.data * self.ema_decay + param_ctx.data * (1. - self.ema_decay)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward: Modo Pré-treino (LeJEPA)
    # ─────────────────────────────────────────────────────────────────────────

    def forward_pretraining(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        
        B = x.shape[0]
        if mask is None:
            mask = self.patch_embedder.generate_random_mask(
                B, x.device, self.config.encoder.mask_ratio
            )

        # 1. Extrai patches e pos_embed UMA única vez
        tokens_clean, pos_embed = self.patch_embedder(x, mask=None)
        
        # 2. E_tgt: Target encoder (limpo)
        with torch.no_grad():
            z_target = self.target_encoder.forward_target(tokens_clean)
            
        # 3. Aplica máscara artificialmente nos tokens latentes para o E_ctx
        # Expande mask token e substitui
        mask_tokens = self.patch_embedder.mask_token.expand(B, tokens_clean.size(1), -1)
        tokens_masked = torch.where(mask.unsqueeze(-1), mask_tokens, tokens_clean)
        
        # 4. E_ctx: Context encoder
        z_ctx = self.encoder.forward_context(tokens_masked)

        # 5. Preditor Mamba: z_ctx → z_pred
        pos_target = pos_embed * mask.float().unsqueeze(-1)
        z_pred = self.predictor(z_ctx, pos_embed_target=pos_target)
        
        # 6. Perda LeJEPA
        loss, metrics = self.loss_fn(z_pred, z_target, mask=mask)
        
        return z_pred, z_target, mask, metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Forward: Modo Inferência / Fine-tuning (Multitarefa)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Modo multitarefa completo para inferência e fine-tuning.

        Args:
            x: [B, C, T, F] — tensor de entrada DSP
               C = 5: [Mag_L, Mag_R, cos_IPD, sin_IPD, GCC-PHAT]
               T = n_time_frames (ex: 128)
               F = n_freq_bins   (ex: 256)

        Returns:
            dict com:
                "masks"   : [B, S, T, F]  — máscaras espectrais BSS
                "doa"     : [B, S, 2]     — ângulos DoA (azimute, elevação)
                "heatmap" : [B, 1, Rx, Ry] — mapa térmico acústico

        Fluxo completo:
            x [B, C, T, F]
              → PatchEmbedder   → tokens [B, L, D]
              → JEPAEncoder     → z [B, L, D]
              → MambaPredictor  → h [B, L, D]
              ├─→ BSSHead       → masks [B, S, T, F]
              ├─→ DoAHead       → doa [B, S, 2]
              └─→ HeatmapHead   → heatmap [B, 1, Rx, Ry]
        """
        # ── 1. Embedding de patches (sem máscara em inferência) ───────────────
        tokens, pos_embed = self.patch_embedder(x, mask=None)   # [B, L, D]

        # ── 2. Codificação (usa E_tgt = encoder limpo para inferência) ────────
        z = self.encoder.forward_target(tokens)    # [B, L, D]

        # ── 3. Processamento SSM temporal via Mamba ───────────────────────────
        h = self.predictor(z)                      # [B, L, D]

        # ── 4. Cabeçotes multitarefa ──────────────────────────────────────────
        masks   = self.bss_head(h)                 # [B, S, T, F]
        doa     = self.doa_head(h)                 # [B, S, 2]
        heatmap = self.heatmap_head(h)             # [B, 1, Rx, Ry]

        return {
            "masks":   masks,     # BSS: máscaras espectrais
            "doa":     doa,       # DoA: coordenadas angulares
            "heatmap": heatmap,   # Mapa acústico 2D
        }

    def count_parameters(self) -> dict[str, int]:
        """Conta parâmetros por módulo para análise de capacidade."""
        modules = {
            "patch_embedder": self.patch_embedder,
            "encoder":        self.encoder,
            "predictor":      self.predictor,
            "bss_head":       self.bss_head,
            "doa_head":       self.doa_head,
            "heatmap_head":   self.heatmap_head,
        }
        counts = {}
        for name, module in modules.items():
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts