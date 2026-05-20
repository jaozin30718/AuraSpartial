"""
data/online_augmentations.py
============================
Aumentações ONLINE aplicadas em tempo real durante o treinamento.

Todas as transformações operam em tensores PyTorch já carregados na CPU
(dentro do Collate), portanto devem ser RÁPIDAS e NÃO-BLOQUEANTES.

REGRA DE OURO para áudio espacial:
  - Qualquer augmentação que altere a AMPLITUDE é segura.
  - Qualquer augmentação que altere a FASE ou o TEMPO é PROIBIDA
    (destruiria DoA, TDoA e coerência inter-canal).

Augmentações implementadas:
  1. Inversão de Polaridade     — Multiplica por -1 (preserva fase relativa)
  2. Random Gain                — Escala global calibrada (mix + targets)
  3. AWGN Online                — Ruído térmico leve (apenas no mix)
  4. Channel Swap (L↔R)         — Troca canais + espelha azimute e heatmap
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class OnlineAugConfig:
    """Configuração das augmentações online."""

    # ── Master switch ────────────────────────────────────────────────────────
    enabled: bool = True           # False = desativa tudo de uma vez

    # ── 1. Inversão de Polaridade ────────────────────────────────────────────
    polarity_flip_prob: float = 0.5
    """Probabilidade de inverter a polaridade (×-1) do sinal inteiro.
    Seguro: a fase relativa entre canais L/R é preservada, então DoA/TDoA
    permanecem intactos. Ensina o modelo a ignorar sinal absoluto vs. relativo."""

    # ── 2. Random Gain ───────────────────────────────────────────────────────
    random_gain_prob: float = 0.8
    random_gain_db_min: float = -6.0    # dB
    random_gain_db_max: float = 3.0     # dB
    """Escala aleatória aplicada IGUALMENTE ao mix estéreo e às sources limpas.
    Seguro: proporções de amplitude entre fontes e canais são preservadas.
    Ensina o modelo a ser invariante ao volume absoluto de gravação."""

    # ── 3. AWGN Online ───────────────────────────────────────────────────────
    awgn_prob: float = 0.5
    awgn_snr_db_min: float = 20.0       # SNR mínima (mais ruído)
    awgn_snr_db_max: float = 50.0       # SNR máxima (menos ruído)
    """Ruído branco gaussiano independente por canal, adicionado APENAS ao mix.
    As sources (targets BSS) NÃO são contaminadas — isso é correto porque o
    modelo deve aprender a SEPARAR o sinal do ruído.
    SNR alta (20-50dB) = ruído sutil, apenas para regularização."""

    # ── 4. Channel Swap (L↔R) ────────────────────────────────────────────────
    channel_swap_prob: float = 0.5
    """Troca canais L e R com probabilidade configurável.
    ATENÇÃO: Exige ajuste simultâneo dos rótulos:
      - Azimute: az_novo = (360° - az_original) % 360°
      - Heatmap: flip horizontal no eixo de azimute
    Efeito: dobra virtualmente o dataset, ensinando simetria espacial."""


class OnlineAugmentor:
    """
    Pipeline de augmentações online para áudio espacial.

    Todas as operações são in-place ou com alocação mínima.
    Projetado para executar dentro do Collate Function do DataLoader.

    Usage:
    ------
    >>> aug = OnlineAugmentor(OnlineAugConfig())
    >>> batch = aug(batch)  # dict com stereo, sources, doa, heatmap
    """

    def __init__(self, config: OnlineAugConfig):
        self.cfg = config

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
        is_training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Aplica todas as augmentações online ao batch collated.

        Parameters
        ----------
        batch : dict com chaves:
            "stereo"  : [B, 2, S]  — mix estéreo
            "sources" : [B, N, S]  — fontes isoladas (target BSS)
            "doa"     : [B, N, 2]  — azimute e elevação
            "heatmap" : [B, Az, El] — mapa espacial
        is_training : bool — se False, não aplica nenhuma augmentação

        Returns
        -------
        dict — mesmo batch, possivelmente augmentado in-place
        """
        if not is_training or not self.cfg.enabled:
            return batch

        # Ordem importa: primeiro alterações de sinal, depois geométricas.
        # Channel swap é por último porque mexe nos labels.
        batch = self._apply_polarity_flip(batch)
        batch = self._apply_random_gain(batch)
        batch = self._apply_awgn(batch)
        batch = self._apply_channel_swap(batch)

        return batch

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Inversão de Polaridade
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_polarity_flip(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Multiplica o sinal inteiro por -1 com probabilidade configurável.

        Aplicado POR AMOSTRA no batch (cada amostra tem flip independente).
        Ambos stereo e sources são invertidos para manter consistência BSS.
        """
        if self.cfg.polarity_flip_prob <= 0:
            return batch

        B = batch["stereo"].shape[0]
        # Máscara booleana por amostra: True = flip
        flip_mask = torch.rand(B) < self.cfg.polarity_flip_prob

        if not flip_mask.any():
            return batch

        # Reshape para broadcasting: [B, 1, 1] para stereo [B, 2, S]
        flip_factor = torch.where(flip_mask, -1.0, 1.0).to(
            device=batch["stereo"].device, dtype=batch["stereo"].dtype
        )

        batch["stereo"] = batch["stereo"] * flip_factor[:, None, None]

        if "sources" in batch:
            batch["sources"] = batch["sources"] * flip_factor[:, None, None]

        return batch

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Random Gain
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_random_gain(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Escala global aleatória em dB, aplicada igualmente ao mix e targets.

        O ganho é sorteado POR AMOSTRA — cada item do batch recebe
        um fator de escala diferente, maximizando a diversidade.

        Matemática:
          gain_linear = 10^(gain_dB / 20)
          stereo *= gain_linear
          sources *= gain_linear  (CRÍTICO: mantém proporção mix/target)
        """
        if self.cfg.random_gain_prob <= 0:
            return batch

        B = batch["stereo"].shape[0]
        apply_mask = torch.rand(B) < self.cfg.random_gain_prob

        if not apply_mask.any():
            return batch

        # Sorteia ganho em dB e converte para linear
        gain_db = torch.empty(B).uniform_(
            self.cfg.random_gain_db_min,
            self.cfg.random_gain_db_max,
        )
        gain_linear = torch.pow(10.0, gain_db / 20.0)

        # Zera o efeito para amostras não selecionadas
        gain_linear = torch.where(apply_mask, gain_linear, torch.ones_like(gain_linear))
        gain_linear = gain_linear.to(
            device=batch["stereo"].device, dtype=batch["stereo"].dtype
        )

        batch["stereo"] = batch["stereo"] * gain_linear[:, None, None]

        if "sources" in batch:
            batch["sources"] = batch["sources"] * gain_linear[:, None, None]

        return batch

    # ─────────────────────────────────────────────────────────────────────────
    # 3. AWGN Online
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_awgn(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Adiciona ruído branco gaussiano independente por canal ao mix estéreo.

        NÃO contamina as sources (targets BSS) — isso é intencional:
        o modelo deve aprender a isolar as fontes mesmo na presença de
        ruído adicional que não existia no dataset offline.

        SNR é sorteada POR AMOSTRA no intervalo configurado.
        """
        if self.cfg.awgn_prob <= 0:
            return batch

        stereo = batch["stereo"]  # [B, 2, S]
        B = stereo.shape[0]
        apply_mask = torch.rand(B) < self.cfg.awgn_prob

        if not apply_mask.any():
            return batch

        # Calcula RMS por amostra: [B]
        rms = torch.sqrt(
            (stereo.float() ** 2).mean(dim=(1, 2)) + 1e-12
        )

        # SNR aleatória por amostra
        snr_db = torch.empty(B).uniform_(
            self.cfg.awgn_snr_db_min,
            self.cfg.awgn_snr_db_max,
        )
        snr_linear = torch.pow(10.0, snr_db / 20.0)
        noise_rms = rms / (snr_linear.to(rms.device) + 1e-12)

        # Zera para amostras não selecionadas
        noise_rms = torch.where(
            apply_mask.to(noise_rms.device), noise_rms, torch.zeros_like(noise_rms)
        )

        # Gera ruído e escala
        noise = torch.randn_like(stereo) * noise_rms[:, None, None]
        batch["stereo"] = stereo + noise

        return batch

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Channel Swap (L ↔ R)
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_channel_swap(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Troca os canais L e R (espelhamento estéreo) com ajuste de labels.

        Para cada amostra onde o swap é aplicado:
          1. stereo[:, [0,1], :] → stereo[:, [1,0], :]
          2. Azimute: az_novo = (360 - az_original) % 360
             (espelha a direção horizontal)
          3. Heatmap: flip no eixo de azimute (dim=-2, pois shape [B, Az, El])

        Isso efetivamente DOBRA o dataset, ensinando ao modelo que
        "fonte à esquerda com sinal forte no mic 0" é equivalente a
        "fonte à direita com sinal forte no mic 1".
        """
        if self.cfg.channel_swap_prob <= 0:
            return batch

        B = batch["stereo"].shape[0]
        swap_mask = torch.rand(B) < self.cfg.channel_swap_prob

        if not swap_mask.any():
            return batch

        for i in range(B):
            if not swap_mask[i]:
                continue

            # 1. Troca canais de áudio
            batch["stereo"][i] = batch["stereo"][i].flip(0)  # flip dim 0 = [2, S]

            # 2. Espelha azimute nos rótulos de DoA
            if "doa" in batch:
                # doa shape: [B, N, 2] onde [:, :, 0] = azimute, [:, :, 1] = elevação
                batch["doa"][i, :, 0] = (360.0 - batch["doa"][i, :, 0]) % 360.0

            # 3. Flip horizontal do heatmap no eixo de azimute
            if "heatmap" in batch:
                # heatmap shape: [B, Az, El]
                batch["heatmap"][i] = batch["heatmap"][i].flip(0)  # flip dim 0 = Az

        return batch
