"""
noise_injector.py
=================
MÓDULO 3: Pipeline de Augmentation e Ruído

Implementa a cadeia completa de degradação realista:
  1. AWGN por canal com SNR variável
  2. Campo difuso (babble / ambient noise)
  3. Hardware mismatch (ganho + phase jitter)
  4. Clipping não-linear (hard / soft)
  5. Fontes em movimento com Efeito Doppler (via variação de RIR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import AugmentationConfig
from room_simulator import RoomSimulationResult
from utils.dsp_utils import (
    apply_hard_clipping,
    apply_phase_jitter,
    apply_soft_clipping,
    compute_rms,
    db_to_linear,
    generate_awgn,
    generate_diffuse_noise_field,
    normalize_signal,
)

logger = logging.getLogger(__name__)


@dataclass
class AugmentedResult:
    """Resultado do pipeline de augmentation."""
    stereo_mix_augmented:  np.ndarray        # (2, n_samples) — sinal degradado (x do modelo)
    source_signals_clean:  List[np.ndarray]  # target BSS — reverberado mas limpo, mic 0
    source_signals_dry:    List[np.ndarray]  # sinais secos originais
    augmentation_log:      dict              # o que foi aplicado e com quais parâmetros

    # Passthrough dos metadados geométricos
    base_result:           RoomSimulationResult


class NoiseInjector:
    """
    Aplica a cadeia de augmentation ao sinal estéreo simulado.

    Cada etapa é aplicada com probabilidade configurável,
    garantindo diversidade extrema no dataset final.

    Usage:
    ------
    >>> injector = NoiseInjector(aug_config)
    >>> augmented = injector.apply(simulation_result, rng)
    """

    def __init__(self, config: AugmentationConfig):
        self.cfg = config

    def apply(
        self,
        sim_result: RoomSimulationResult,
        rng:        np.random.Generator,
        noise_audio: Optional[np.ndarray] = None,   # babble noise externo
    ) -> AugmentedResult:
        """
        Aplica toda a cadeia de augmentation ao sinal estéreo.

        Parameters
        ----------
        sim_result  : resultado da simulação de sala (antes de qualquer degradação)
        rng         : Generator para reprodutibilidade
        noise_audio : áudio de babble noise externo (mono), opcional

        Returns
        -------
        AugmentedResult com sinal degradado + ground truths preservados
        """
        # Cópia de trabalho — preserva o original como ground truth
        stereo = sim_result.stereo_mix.copy().astype(np.float64)

        aug_log = {
            "awgn_applied":   False, "awgn_snr_db": None,
            "diffuse_applied": False, "diffuse_snr_db": None,
            "gain_mismatch_db": None,
            "phase_jitter_samples": None,
            "clipping_applied": False, "clipping_type": None,
        }

        n_samples = stereo.shape[1]

        # ── ETAPA 1: Hardware Mismatch (aplica ANTES do ruído para ser fisicamente correto)
        stereo, aug_log = self._apply_hardware_mismatch(stereo, rng, aug_log, sim_result.sample_rate)

        # ── ETAPA 2: Ruído Difuso (campo ambiental)
        stereo, aug_log = self._apply_diffuse_noise(
            stereo, n_samples, rng, aug_log, noise_audio, sim_result.sample_rate
        )

        # ── ETAPA 3: AWGN (ruído térmico do ADC)
        stereo, aug_log = self._apply_awgn(stereo, rng, aug_log)

        # ── ETAPA 4: Clipping não-linear
        stereo, aug_log = self._apply_clipping(stereo, rng, aug_log)

        # 🟢 CORREÇÃO CRÍTICA: Normalização calibrada.
        # Se a mistura atinge o pico e é atenuada, OS SINAIS ISOLADOS (TARGETS)
        # TAMBÉM DEVEM SER ATENUADOS NA MESMA PROPORÇÃO. Isso impede a rede de
        # precisar gerar máscaras maiores que 1.0 (o que é impossível com Sigmoid).
        peak = np.max(np.abs(stereo))
        
        if peak > 0.98:
            scale_factor = 0.95 / peak
            stereo = stereo * scale_factor
            
            # Escala os Ground Truths na mesma proporção exata do sinal misturado
            clean_targets =[s * scale_factor for s in sim_result.source_at_mic0]
            dry_targets   =[s * scale_factor for s in sim_result.source_signals_dry]
        else:
            clean_targets = sim_result.source_at_mic0
            dry_targets   = sim_result.source_signals_dry

        return AugmentedResult(
            stereo_mix_augmented = stereo.astype(np.float32),
            source_signals_clean = clean_targets,  # Alvos com escala matemática correta
            source_signals_dry   = dry_targets,    # Alvos secos com escala matemática correta
            augmentation_log     = aug_log,
            base_result          = sim_result,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ETAPA 1: Hardware Mismatch
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_hardware_mismatch(
        self,
        stereo:  np.ndarray,
        rng:     np.random.Generator,
        aug_log: dict,
        sample_rate: int,
    ) -> Tuple[np.ndarray, dict]:
        """
        Simula imperfeições de fabricação entre microfones L e R:
          - Diferença de ganho (tolerância MEMS ≈ ±3dB)
          - Phase jitter (clock I2S imperfeito)
        """
        # 1.1 — Gain Mismatch no canal R
        if rng.random() < self.cfg.gain_mismatch_probability:
            gain_db    = rng.uniform(*self.cfg.gain_mismatch_range)
            gain_linear = db_to_linear(gain_db)
            stereo[1]  *= gain_linear
            aug_log["gain_mismatch_db"] = float(gain_db)

        # 1.2 — Phase Jitter no canal R
        if rng.random() < self.cfg.phase_jitter_probability:
            jitter_samples = rng.uniform(*self.cfg.phase_jitter_samples_range)
            # Direção aleatória do jitter
            if rng.random() > 0.5:
                jitter_samples = -jitter_samples

            stereo[1] = apply_phase_jitter(
                signal          = stereo[1].astype(np.float32),
                jitter_samples  = jitter_samples,
                sample_rate     = sample_rate,
            ).astype(np.float64)
            aug_log["phase_jitter_samples"] = float(jitter_samples)

        return stereo, aug_log

    # ─────────────────────────────────────────────────────────────────────────
    # ETAPA 2: Ruído Difuso
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_diffuse_noise(
        self,
        stereo:      np.ndarray,
        n_samples:   int,
        rng:         np.random.Generator,
        aug_log:     dict,
        noise_audio: Optional[np.ndarray] = None,
        sample_rate: int = 16_000,
    ) -> Tuple[np.ndarray, dict]:
        """
        Injeta campo difuso esférico simulado.

        Se `noise_audio` (babble real) for fornecido, usa como base
        e aplica descorrelação espacial sintética entre os canais.
        """
        if rng.random() > self.cfg.diffuse_noise_probability:
            return stereo, aug_log

        snr_db = rng.uniform(*self.cfg.diffuse_noise_snr_range)
        signal_rms = compute_rms(stereo)
        target_noise_rms = signal_rms / (db_to_linear(snr_db) + 1e-12)

        if noise_audio is not None and len(noise_audio) > 0:
            # Usa ruído real — tile/crop
            if len(noise_audio) < n_samples:
                noise_audio = np.tile(noise_audio, int(np.ceil(n_samples / len(noise_audio))))
            start = rng.integers(0, len(noise_audio) - n_samples + 1)
            base  = noise_audio[start : start + n_samples].astype(np.float64)

            # Descorrelação entre canais via filtro de atraso sintético
            diffuse = np.stack([
                base,
                apply_phase_jitter(
                    base.astype(np.float32),
                    rng.uniform(1.0, 5.0),
                    sample_rate,
                ).astype(np.float64)
            ])
        else:
            # Campo difuso totalmente sintético
            diffuse = generate_diffuse_noise_field(
                n_samples   = n_samples,
                n_channels  = 2,
                sample_rate = sample_rate,
                rng         = rng,
                spectrum_type = rng.choice(["pink", "brown", "white"]),
            ).astype(np.float64)

        # Calibra amplitude para SNR alvo
        diffuse_rms  = compute_rms(diffuse)
        diffuse_scaled = diffuse * (target_noise_rms / (diffuse_rms + 1e-12))

        stereo  += diffuse_scaled
        aug_log["diffuse_applied"]  = True
        aug_log["diffuse_snr_db"]   = float(snr_db)

        return stereo, aug_log

    # ─────────────────────────────────────────────────────────────────────────
    # ETAPA 3: AWGN
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_awgn(
        self,
        stereo:  np.ndarray,
        rng:     np.random.Generator,
        aug_log: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        Adiciona Ruído Branco Gaussiano independente por canal.
        Simula ruído térmico do ADC e pré-amplificador.
        """
        if rng.random() > self.cfg.awgn_probability:
            return stereo, aug_log

        snr_db = rng.uniform(*self.cfg.awgn_snr_range)
        noise  = generate_awgn(
            signal      = stereo,
            snr_db      = snr_db,
            rng         = rng,
            independent_channels = True,   # canal L e R com ruído independente
        ).astype(np.float64)

        stereo += noise
        aug_log["awgn_applied"] = True
        aug_log["awgn_snr_db"]  = float(snr_db)

        return stereo, aug_log

    # ─────────────────────────────────────────────────────────────────────────
    # ETAPA 4: Clipping
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_clipping(
        self,
        stereo:  np.ndarray,
        rng:     np.random.Generator,
        aug_log: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        Simula saturação do diafragma / overflow do ADC.
        Hard clipping: truncamento brusco
        Soft clipping: distorção por tanh
        """
        if rng.random() > self.cfg.clipping_probability:
            return stereo, aug_log

        threshold = rng.uniform(*self.cfg.clipping_threshold_range)

        if self.cfg.clipping_type == "hard":
            clip_type = "hard"
        elif self.cfg.clipping_type == "soft":
            clip_type = "soft"
        else:
            clip_type = rng.choice(["hard", "soft"])

        stereo_f32 = stereo.astype(np.float32)

        if clip_type == "hard":
            result = apply_hard_clipping(stereo_f32, threshold)
        else:
            result = apply_soft_clipping(stereo_f32, threshold)

        aug_log["clipping_applied"] = True
        aug_log["clipping_type"]    = clip_type
        aug_log["clipping_threshold"] = float(threshold)

        return result.astype(np.float64), aug_log


# ─────────────────────────────────────────────────────────────────────────────
# Doppler / Fonte em Movimento (extensão do MÓDULO 3)
# ─────────────────────────────────────────────────────────────────────────────

class MovingSourceSimulator:
    """
    Simula fontes em movimento via variação de RIR ao longo do tempo.

    Estratégia: divide o áudio em frames curtos, calcula RIR para cada
    posição da trajetória e aplica OLA (Overlap-Add) para transição suave.

    Isso produz:
      - Variação contínua de TDoA (pistas ITD dinâmicas)
      - Efeito Doppler realista (pitch shift dependente da velocidade radial)
      - Variação de nível (ILD dinâmico)

    Usage:
    ------
    >>> mss = MovingSourceSimulator(config, room, mic_positions)
    >>> stereo_moving = mss.simulate_trajectory(audio, trajectory, rng)
    """

    FRAME_DURATION_MS = 50    # ms por frame de RIR
    OVERLAP_RATIO     = 0.50  # 50% overlap para OLA suave

    def __init__(
        self,
        room_config:   "RoomConfig",
        room_dims:     np.ndarray,
        mic_positions: np.ndarray,   # (2, 3)
        absorption_material,         # objeto pra.Material
        sample_rate:   int = 16_000,
    ):
        self.cfg          = room_config
        self.room_dims    = room_dims
        self.mic_pos      = mic_positions
        self.material     = absorption_material
        self.sr           = sample_rate
        self.frame_len    = int(self.FRAME_DURATION_MS * sample_rate / 1000)
        self.hop_len      = int(self.frame_len * (1 - self.OVERLAP_RATIO))

    def simulate_trajectory(
        self,
        source_audio: np.ndarray,            # (n_samples,) mono
        trajectory:   np.ndarray,            # (n_frames, 3)
        rng:          np.random.Generator,
    ) -> np.ndarray:
        """
        Sintetiza sinal estéreo de uma fonte em movimento.

        Returns
        -------
        np.ndarray shape (2, n_samples)
        """
        import pyroomacoustics as pra

        n_samples   = len(source_audio)
        n_traj      = len(trajectory)
        output      = np.zeros((2, n_samples), dtype=np.float64)
        window      = np.hanning(self.frame_len)

        for frame_idx in range(0, n_samples - self.frame_len, self.hop_len):
            # Posição da fonte neste frame
            traj_idx = min(int(frame_idx / n_samples * n_traj), n_traj - 1)
            src_pos  = trajectory[traj_idx]

            # Chunk de áudio desta janela
            chunk = source_audio[frame_idx : frame_idx + self.frame_len] * window

            # Simula sala mínima para este frame (max_order baixo = rápido)
            try:
                room_frame = pra.ShoeBox(
                    self.room_dims.tolist(),
                    fs        = self.sr,
                    materials = self.material,
                    max_order = 3,        # baixo para eficiência por frame
                )
                room_frame.add_microphone_array(self.mic_pos.T)
                room_frame.add_source(src_pos.tolist(), signal=chunk)
                room_frame.simulate()

                frame_stereo = room_frame.mic_array.signals
                frame_stereo = RoomSimulator._trim_or_pad(frame_stereo, self.frame_len)

                # Overlap-Add
                end = min(frame_idx + self.frame_len, n_samples)
                actual_len = end - frame_idx
                output[:, frame_idx:end] += frame_stereo[:, :actual_len]

            except Exception as e:
                logger.debug(f"Frame {frame_idx} moving source error: {e}")
                continue

        # Normaliza para compensar acúmulo do OLA
        ola_gain = 1.0 / (1.0 + self.OVERLAP_RATIO)
        output  *= ola_gain

        return output.astype(np.float32)