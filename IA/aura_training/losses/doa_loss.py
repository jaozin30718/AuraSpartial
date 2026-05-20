"""
losses/doa_loss.py
==================
MSE Circular para regressão angular de DoA.

O problema da MSE angular: a distância entre 359° e 1° é 358° em MSE linear,
mas apenas 2° na geometria circular. A perda circular resolve isso.

Formulação:
    L_circular = E[ 1 - cos(θ_pred - θ_true) ]

Esta formulação:
  1. É diferenciável em todo o domínio
  2. Trata a periodicidade 0°-360° corretamente
  3. Varia de 0 (predição perfeita) a 2 (predição oposta)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math


class CircularMSELoss(nn.Module):
    """
    Perda de regressão angular circular.

    Converte ângulos para representação vetorial (cos, sin) e calcula
    distância no espaço circular, evitando a descontinuidade em 360°/0°.

    Args:
        azimuth_weight   : peso relativo para erro de azimute
        elevation_weight : peso relativo para erro de elevação
        reduction        : "mean" | "sum" | "none"
    """

    def __init__(
        self,
        azimuth_weight:   float = 1.0,
        elevation_weight: float = 1.0,
        reduction:        str   = "mean",
    ) -> None:
        super().__init__()
        self.w_az  = azimuth_weight
        self.w_el  = elevation_weight
        self.reduction = reduction

    def forward(
        self,
        pred:   torch.Tensor,   # [B, N, 2] — (azimute_pred, elevacao_pred) em graus
        target: torch.Tensor,   # [B, N, 2] — (azimute_true, elevacao_true) em graus
        vad:    torch.Tensor | None = None,  # [B, N] — peso por fonte ativa
    ) -> torch.Tensor:
        """
        Args:
            pred   : [B, N, 2] — predições angulares em graus
            target : [B, N, 2] — alvos angulares em graus
            vad    : [B, N] — máscara de atividade vocal (pondera fontes ativas)

        Returns:
            loss: escalar — perda circular média
        """
        # Converte graus → radianos
        deg2rad = math.pi / 180.0
        pred_rad   = pred   * deg2rad    # [B, N, 2]
        target_rad = target * deg2rad    # [B, N, 2]

        # ── Azimute: [B, N] ────────────────────────────────────────────────────
        # Usa representação 2D (cos, sin) para circularidade
        az_pred_cos   = torch.cos(pred_rad[..., 0])
        az_pred_sin   = torch.sin(pred_rad[..., 0])
        az_target_cos = torch.cos(target_rad[..., 0])
        az_target_sin = torch.sin(target_rad[..., 0])

        # Distância circular: 1 - cos(Δθ) = 1 - (cos_p*cos_t + sin_p*sin_t)
        az_loss = 1.0 - (
            az_pred_cos * az_target_cos +
            az_pred_sin * az_target_sin
        )  # [B, N] ∈ [0, 2]

        # ── Elevação: [B, N] ────────────────────────────────────────────────────
        el_pred_cos   = torch.cos(pred_rad[..., 1])
        el_pred_sin   = torch.sin(pred_rad[..., 1])
        el_target_cos = torch.cos(target_rad[..., 1])
        el_target_sin = torch.sin(target_rad[..., 1])

        el_loss = 1.0 - (
            el_pred_cos * el_target_cos +
            el_pred_sin * el_target_sin
        )  # [B, N] ∈ [0, 2]

        # ── Perda combinada ────────────────────────────────────────────────────
        loss_per_source = (
            self.w_az  * az_loss +
            self.w_el  * el_loss
        )  # [B, N]

        # Pondera por VAD se fornecido (apenas fontes ativas contribuem)
        if vad is not None:
            # vad: [B, N] ∈ [0, 1]
            loss_per_source = loss_per_source * vad
            # Normaliza pela soma dos pesos para evitar viés
            denom = vad.sum() + 1e-8
            return loss_per_source.sum() / denom

        if self.reduction == "mean":
            return loss_per_source.mean()
        elif self.reduction == "sum":
            return loss_per_source.sum()
        else:
            return loss_per_source   # [B, N]

    @staticmethod
    def angular_error_degrees(
        pred:   torch.Tensor,   # [B, N, 2]
        target: torch.Tensor,   # [B, N, 2]
    ) -> torch.Tensor:
        """Calcula erro angular absoluto em graus (para métricas de validação)."""
        deg2rad = math.pi / 180.0

        # Para azimute
        delta_az = pred[..., 0] - target[..., 0]
        # Wraps para [-180, 180]
        delta_az = (delta_az + 180) % 360 - 180

        # Para elevação (limitada a [-90, 90])
        delta_el = pred[..., 1] - target[..., 1]

        error = torch.sqrt(delta_az ** 2 + delta_el ** 2)  # [B, N] em graus
        return error

