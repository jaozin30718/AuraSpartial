"""
ground_truth_extractor.py
=========================
MÓDULO 4: Extrator de Ground Truth

Empacota os rótulos perfeitos que a rede neural usará nas Loss Functions:
  - Coordenadas angulares (DoA) por fonte e por frame
  - Áudios isolados por fonte (target BSS / SI-SDR)
  - Máscaras VAD por fonte
  - Mapa térmico espacial 2D/3D (heatmap de energia por ângulo)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from room_simulator import RoomSimulationResult, SourceMetadata
from noise_injector import AugmentedResult
from utils.dsp_utils import compute_rms, compute_si_sdr
from utils.geometry_utils import compute_tdoa

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthPackage:
    """
    Pacote completo de rótulos para uma amostra de treino.

    Arrays shapes (com N = n_sources, T = n_frames, S = n_samples):
    """
    # ── DoA Ground Truth ─────────────────────────────────────────────────────
    # (N, T, 2) — azimute e elevação em graus para cada frame de tempo
    doa_azimuth_elevation:  np.ndarray     # float32

    # (N, T, 3) — posições cartesianas absolutas
    source_positions_xyz:   np.ndarray     # float32

    # (N, T) — distâncias fonte-array (metros)
    source_distances:       np.ndarray     # float32

    # (N, T) — TDoA em amostras
    tdoa_samples_per_frame: np.ndarray     # float32

    # ── BSS Ground Truth ─────────────────────────────────────────────────────
    # (N, S) — sinal reverberado isolado de cada fonte no mic 0
    source_images_mic0:     np.ndarray     # float32

    # (N, S) — sinais secos originais
    source_signals_dry:     np.ndarray     # float32

    # ── VAD ──────────────────────────────────────────────────────────────────
    # (N, T) — bool: fonte ativa neste frame?
    vad_masks_per_frame:    np.ndarray     # bool

    # (N, S) — VAD em nível de amostra
    vad_masks_sample_level: np.ndarray     # bool

    # ── Heatmap Espacial ─────────────────────────────────────────────────────
    # (T, Az_bins, El_bins) — energia por célula angular por frame
    spatial_heatmap:        np.ndarray     # float32

    # ── Metadados Escalares ──────────────────────────────────────────────────
    n_sources:              int
    n_frames:               int
    n_samples:              int
    frame_duration_ms:      float
    sample_rate:            int
    room_dims:              np.ndarray     # (3,)
    rt60_target:            float
    rt60_measured:          Optional[float]

    # ── Métricas de Qualidade ────────────────────────────────────────────────
    si_sdr_per_source:      np.ndarray     # (N,) — SI-SDR das imagens isoladas
    augmentation_log:       dict


class GroundTruthExtractor:
    """
    Extrai e formata todos os labels necessários para treinamento.

    Parâmetros de resolução temporal:
      - frame_duration_ms: janela de análise para DoA e VAD (ex: 20ms)
      - heatmap_az_bins:   resolução azimutal do mapa térmico (ex: 72 → 5° por bin)
      - heatmap_el_bins:   resolução de elevação (ex: 37 → 5° por bin, -90° a +90°)

    Usage:
    ------
    >>> extractor = GroundTruthExtractor(frame_duration_ms=20.0)
    >>> gt = extractor.extract(augmented_result)
    """

    def __init__(
        self,
        frame_duration_ms: float = 20.0,
        heatmap_az_bins:   int   = 72,      # 360° / 5° = 72 bins
        heatmap_el_bins:   int   = 37,      # 180° / 5° = 37 bins (-90 a +90)
    ):
        self.frame_ms      = frame_duration_ms
        self.az_bins       = heatmap_az_bins
        self.el_bins       = heatmap_el_bins

        # Pré-computa meshgrid do heatmap uma única vez (evita alocar a cada frame)
        _az = np.arange(heatmap_az_bins, dtype=np.float32)
        _el = np.arange(heatmap_el_bins, dtype=np.float32)
        self._grid_az, self._grid_el = np.meshgrid(_az, _el, indexing="ij")  # Garante ordem (Az, El)

    def extract(self, aug_result: AugmentedResult) -> GroundTruthPackage:
        """
        Processa o AugmentedResult e retorna o pacote completo de labels.
        """
        sim = aug_result.base_result
        sr  = sim.sample_rate
        N   = len(sim.sources_meta)
        S   = sim.stereo_mix.shape[1]
        T   = int(np.ceil(S / (self.frame_ms * sr / 1000)))
        frame_len = int(self.frame_ms * sr / 1000)

        # ── DoA por frame ─────────────────────────────────────────────────────
        doa_az_el   = np.zeros((N, T, 2),  dtype=np.float32)
        src_xyz     = np.zeros((N, T, 3),  dtype=np.float32)
        src_dist    = np.zeros((N, T),     dtype=np.float32)
        tdoa_frames = np.zeros((N, T),     dtype=np.float32)
        vad_frames  = np.zeros((N, T),     dtype=bool)

        for i, meta in enumerate(sim.sources_meta):
            if meta.is_moving and meta.trajectory is not None:
                # Fonte em movimento: interpola DoA por frame
                n_traj = len(meta.trajectory)
                for t in range(T):
                    traj_idx = min(int(t / T * n_traj), n_traj - 1)
                    pos_t    = meta.trajectory[traj_idx]
                    az, el, dist = self._cartesian_to_spherical(pos_t, sim.mic_array_center)
                    tdoa_s, tdoa_smp = compute_tdoa(
                        pos_t, sim.mic_positions[0], sim.mic_positions[1],
                        sample_rate=sr
                    )
                    doa_az_el[i, t]   = [az, el]
                    src_xyz[i, t]     = pos_t
                    src_dist[i, t]    = dist
                    tdoa_frames[i, t] = tdoa_smp
            else:
                # Fonte estática: repete valores
                doa_az_el[i, :] = [meta.azimuth_deg, meta.elevation_deg]
                src_xyz[i, :]   = meta.position
                src_dist[i, :]  = meta.distance_m
                tdoa_frames[i, :] = meta.tdoa_samples

            # VAD por frame
            if meta.vad_mask is not None:
                for t in range(T):
                    start = t * frame_len
                    end   = min(S, start + frame_len)
                    vad_frames[i, t] = np.any(meta.vad_mask[start:end])

        # ── BSS Ground Truth ─────────────────────────────────────────────────
        source_images = np.zeros((N, S), dtype=np.float32)
        source_dry    = np.zeros((N, S), dtype=np.float32)

        for i in range(N):
            img = aug_result.source_signals_clean[i][:S]
            dry = aug_result.source_signals_dry[i][:S]
            source_images[i, :len(img)] = img
            source_dry[i,    :len(dry)] = dry

        # VAD nível de amostra
        vad_samples = np.zeros((N, S), dtype=bool)
        for i, meta in enumerate(sim.sources_meta):
            if meta.vad_mask is not None:
                vad_samples[i, :len(meta.vad_mask)] = meta.vad_mask[:S]

        # ── Heatmap Espacial ─────────────────────────────────────────────────
        heatmap = self._build_spatial_heatmap(
            sim, aug_result.stereo_mix_augmented,
            sr, T, frame_len, vad_frames,
        )

        # ── SI-SDR por fonte ─────────────────────────────────────────────────
        si_sdr_values = np.zeros(N, dtype=np.float32)
        mix_mono = aug_result.stereo_mix_augmented.mean(axis=0)
        for i in range(N):
            if compute_rms(source_images[i]) > 1e-6:
                si_sdr_values[i] = compute_si_sdr(source_images[i], mix_mono)

        return GroundTruthPackage(
            doa_azimuth_elevation  = doa_az_el,
            source_positions_xyz   = src_xyz,
            source_distances       = src_dist,
            tdoa_samples_per_frame = tdoa_frames,
            source_images_mic0     = source_images,
            source_signals_dry     = source_dry,
            vad_masks_per_frame    = vad_frames,
            vad_masks_sample_level = vad_samples,
            spatial_heatmap        = heatmap,
            n_sources              = N,
            n_frames               = T,
            n_samples              = S,
            frame_duration_ms      = self.frame_ms,
            sample_rate            = sr,
            room_dims              = sim.room_dims,
            rt60_target            = sim.rt60_target,
            rt60_measured          = sim.rt60_measured,
            si_sdr_per_source      = si_sdr_values,
            augmentation_log       = aug_result.augmentation_log,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Heatmap Espacial
    # ─────────────────────────────────────────────────────────────────────────

    def _build_spatial_heatmap(
        self,
        sim:       RoomSimulationResult,
        stereo:    np.ndarray,         # (2, S)
        sr:        int,
        n_frames:  int,
        frame_len: int,
        vad_frames: np.ndarray,        # (N, T)
    ) -> np.ndarray:
        """
        Constrói heatmap espacial (T, Az_bins, El_bins) via GCC-PHAT por frame.

        Para cada frame de tempo:
          1. Calcula correlação cruzada generalizada (GCC-PHAT) entre os dois microfones
          2. Projeta o pico de TDoA em bins de azimute/elevação
          3. Preenche com energia real das fontes ativas (ground truth)

        Returns
        -------
        np.ndarray shape (T, Az_bins, El_bins) dtype float32
        """
        heatmap = np.zeros((n_frames, self.az_bins, self.el_bins), dtype=np.float32)

        for t in range(n_frames):
            start = t * frame_len
            end   = min(stereo.shape[1], start + frame_len)

            # Identifica fontes ativas neste frame
            for i, meta in enumerate(sim.sources_meta):
                # Avalia energia do sinal REVERBERADO (captura caudas de eco e RT60 mesmo após VAD falso)
                src_img = sim.source_at_mic0[i][start:end]
                energy  = float(compute_rms(src_img)) if len(src_img) > 0 else 0.0

                if energy < 1e-4:
                    continue  # Fonte inaudível neste frame

                # Converte posição da fonte em bin angular
                if meta.is_moving and meta.trajectory is not None:
                    n_traj   = len(meta.trajectory)
                    traj_idx = min(int(t / n_frames * n_traj), n_traj - 1)
                    pos      = meta.trajectory[traj_idx]
                    az, el, dist = self._cartesian_to_spherical(pos, sim.mic_array_center)
                else:
                    az  = meta.azimuth_deg
                    el  = meta.elevation_deg

                # Mapeia ângulos para bins
                az_bin = int((az % 360.0) / 360.0 * self.az_bins)
                el_bin = int((el + 90.0)  / 180.0 * self.el_bins)

                az_bin = np.clip(az_bin, 0, self.az_bins - 1)
                el_bin = np.clip(el_bin, 0, self.el_bins - 1)

                # Preenchimento com suavização gaussiana ao redor do bin central
                heatmap[t] = self._add_gaussian_blob(
                    grid     = heatmap[t],
                    center   = (az_bin, el_bin),
                    sigma    = 2.0,
                    amplitude = energy,
                )

        # Normaliza por frame para [0, 1] — vetorizado
        frame_max = heatmap.max(axis=(1, 2), keepdims=True)  # (T, 1, 1)
        mask = frame_max > 1e-9
        heatmap = np.where(mask, heatmap / (frame_max + 1e-12), heatmap)

        return heatmap

    def _add_gaussian_blob(
        self,
        grid:      np.ndarray,     # (Az, El)
        center:    Tuple[int, int],
        sigma:     float,
        amplitude: float,
    ) -> np.ndarray:
        """Adiciona blob gaussiano 2D com wrapping circular no azimute."""
        c_az, c_el = center
        
        # Distância com wrap-around (circularidade 360°) no Azimute
        dist_az = np.abs(self._grid_az - c_az)
        dist_az = np.minimum(dist_az, self.az_bins - dist_az)
        
        # Distância padrão na elevação
        dist_el = self._grid_el - c_el
        
        blob = amplitude * np.exp(-(dist_az**2 + dist_el**2) / (2 * sigma**2))
        return grid + blob.astype(np.float32)

    @staticmethod
    def _cartesian_to_spherical(pos, center):
        from utils.geometry_utils import cartesian_to_spherical
        return cartesian_to_spherical(pos, center)