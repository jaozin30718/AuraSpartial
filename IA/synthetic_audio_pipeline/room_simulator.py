"""
room_simulator.py
=================
MÓDULO 1: Motor Geométrico — sala 3D aleatória com ISM via pyroomacoustics
MÓDULO 2: Motor de Convolução — RIR + síntese do sinal estéreo

Referências:
  - Scheibler, R. et al. (2018) — pyroomacoustics: A Python package for
    audio room simulation and array processing algorithms.
  - Allen, J.B. & Berkley, D.A. (1979) — Image Source Method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyroomacoustics as pra

from config import RoomConfig, SPEED_OF_SOUND
from dataset_loaders import AudioFileIndex, AudioLoader, PrefetchQueue
from utils.geometry_utils import (
    cartesian_to_spherical,
    compute_tdoa,
    generate_vad_mask,
    sample_random_position_in_room,
    build_spline_trajectory,
)
from utils.dsp_utils import normalize_signal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses de Resultado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceMetadata:
    """Metadados completos de uma fonte sonora simulada."""
    source_id:      int
    position:       np.ndarray            # (3,) — [x, y, z] metros
    azimuth_deg:    float
    elevation_deg:  float
    distance_m:     float
    tdoa_seconds:   float                 # TDoA teórico entre mic0 e mic1
    tdoa_samples:   float
    is_moving:      bool = False
    trajectory:     Optional[np.ndarray] = None  # (n_frames, 3) se móvel
    vad_mask:       Optional[np.ndarray] = None  # (n_samples,) bool


@dataclass
class RoomSimulationResult:
    """Resultado completo de uma simulação de sala."""
    # Sinais
    stereo_mix:        np.ndarray         # (2, n_samples) — mistura estéreo
    source_signals_dry: List[np.ndarray]  # lista de (n_samples,) — fontes secas
    source_at_mic0:    List[np.ndarray]   # (n_samples,) por fonte — reverberado, mic 0

    # Metadados
    room_dims:         np.ndarray         # (3,) — [W, L, H]
    rt60_target:       float
    rt60_measured:     Optional[float]
    mic_positions:     np.ndarray         # (2, 3) — posições dos 2 mics
    mic_array_center:  np.ndarray         # (3,)
    sources_meta:      List[SourceMetadata]

    # Sample rate
    sample_rate:       int

    # RIRs brutas (para debug/análise)
    rirs:              Optional[List[np.ndarray]] = None  # [source][mic]


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 1 & 2: RoomSimulator
# ─────────────────────────────────────────────────────────────────────────────

class RoomSimulator:
    """
    Motor principal de simulação acústica baseado no Image Source Method.

    Responsabilidades:
    1. Gerar sala 3D aleatória com materiais de absorção variáveis
    2. Posicionar array estéreo e N fontes sonoras
    3. Calcular RIRs via pyroomacoustics (ISM)
    4. Convolucionar fontes secas → sinal estéreo físico

    Usage:
    ------
    >>> sim = RoomSimulator(room_config, audio_loader, source_index)
    >>> result = sim.simulate(rng=np.random.default_rng(42))
    """

    def __init__(
        self,
        config:       RoomConfig,
        audio_loader: AudioLoader,
        source_index: AudioFileIndex,
        noise_index:  Optional[AudioFileIndex] = None,
        moving_source_probability: float = 0.20,
    ):
        self.config     = config
        self.loader     = audio_loader
        self.src_index  = source_index
        self.noise_index = noise_index
        self.moving_prob = moving_source_probability
        self._n_samples  = int(config.audio_duration_seconds * config.sample_rate)

    # ─────────────────────────────────────────────────────────────────────────
    # API Pública
    # ─────────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        rng: np.random.Generator,
    ) -> RoomSimulationResult:
        """
        Executa uma simulação completa e retorna todos os sinais e metadados.

        Pipeline interno:
          1. _sample_room_geometry()     → dimensões + absorção
          2. _create_pra_room()          → objeto ShoeBox pyroomacoustics
          3. _place_microphone_array()   → posições dos 2 mics
          4. _place_sources()            → posições + áudios das fontes
          5. _simulate_and_convolve()    → RIR + convolução
          6. _package_result()           → RoomSimulationResult
        """
        # ── 1. Geometria ──────────────────────────────────────────────────────
        room_dims, rt60, absorption_coeffs = self._sample_room_geometry(rng)

        # ── 2. Criar sala PRA ─────────────────────────────────────────────────
        room = self._create_pra_room(room_dims, absorption_coeffs, rt60)

        # ── 3. Microfones ─────────────────────────────────────────────────────
        mic_positions, array_center = self._place_microphone_array(room_dims, rng)
        mic_array = np.array(mic_positions).T   # shape (3, 2) para PRA

        # ── 4. Fontes ─────────────────────────────────────────────────────────
        n_sources = int(rng.integers(
            self.config.n_sources_range[0],
            self.config.n_sources_range[1] + 1,
        ))

        source_positions =[]
        source_audios    =[]   # sinais mono secos
        occupied_zones   = list(mic_positions)
        vads_list        =[]   # Armazena as máscaras para resgatar no metadado

        for i in range(n_sources):
            pos = sample_random_position_in_room(
                room_dims    = room_dims,
                wall_margin  = self.config.source_wall_margin,
                height_range = (0.30, room_dims[2] - 0.30),
                forbidden_zones = occupied_zones,
                min_separation  = self.config.source_mic_min_dist,
                rng = rng,
            )
            source_positions.append(pos)
            occupied_zones.append(pos)

            # Carrega áudio seco
            path  = self.src_index.sample_path(rng)
            audio = self.loader.load_mono(
                path, self.config.audio_duration_seconds, rng
            )
            
            # Aplica VAD ANTES de inserir na simulação física da sala
            vad = generate_vad_mask(
                n_samples           = self._n_samples,
                sample_rate         = self.config.sample_rate,
                silence_probability = 0.30,
                rng                 = rng,
            )
            audio = audio * vad
            
            vads_list.append(vad)
            source_audios.append(audio)

        # ── 5. Adiciona mics e fontes à sala PRA ─────────────────────────────
        room.add_microphone_array(mic_array)

        for i, (pos, audio) in enumerate(zip(source_positions, source_audios)):
            room.add_source(pos.tolist(), signal=audio)

        # ── 6. Simula (RIR + Convolução) ─────────────────────────────────────
        logger.debug(f"Simulando sala {room_dims} RT60={rt60:.2f}s ...")
        room.simulate()

        # ── 7. Extrai sinais por microfone ────────────────────────────────────
        # room.mic_array.signals: shape (n_mics, n_samples)
        mic_signals = room.mic_array.signals   # (2, n_samples)

        # Garante comprimento correto (ISM pode adicionar amostras de rabo)
        mic_signals = self._trim_or_pad(mic_signals, self._n_samples)

        # ── 8. Extrai sinais isolados por fonte no mic 0 ──────────────────────
        # Re-executa convolução manualmente para isolamento de fonte
        source_at_mic0 = self._extract_isolated_sources(
            room, source_audios, mic_idx=0, rng=rng
        )

        # ── 9. Metadados angulares + VAD ──────────────────────────────────────
        sources_meta =[]
        for i, pos in enumerate(source_positions):
            az, el, dist = cartesian_to_spherical(pos, array_center)
            tdoa_s, tdoa_smp = compute_tdoa(
                pos, mic_positions[0], mic_positions[1],
                speed_of_sound=SPEED_OF_SOUND,
                sample_rate=self.config.sample_rate,
            )

            # Resgata o VAD físico que foi efetivamente aplicado ao áudio
            vad = vads_list[i]

            meta = SourceMetadata(
                source_id     = i,
                position      = pos.astype(np.float32),
                azimuth_deg   = az,
                elevation_deg = el,
                distance_m    = dist,
                tdoa_seconds  = tdoa_s,
                tdoa_samples  = tdoa_smp,
                is_moving     = False,
                trajectory    = None,
                vad_mask      = vad,
            )
            sources_meta.append(meta)

        # ── 10. RT60 medido (opcional) ────────────────────────────────────────
        # Atribui diretamente o valor teórico para evitar o cálculo pesado de
        # room.measure_rt60() que custa ~1 segundo por sala simulada.
        rt60_measured = rt60

        return RoomSimulationResult(
            stereo_mix         = mic_signals.astype(np.float32),
            source_signals_dry = [a.astype(np.float32) for a in source_audios],
            source_at_mic0     = [s.astype(np.float32) for s in source_at_mic0],
            room_dims          = room_dims.astype(np.float32),
            rt60_target        = rt60,
            rt60_measured      = rt60_measured,
            mic_positions      = np.array(mic_positions, dtype=np.float32),
            mic_array_center   = array_center.astype(np.float32),
            sources_meta       = sources_meta,
            sample_rate        = self.config.sample_rate,
            rirs               = None,   # economiza memória; ativar se necessário
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Métodos Privados
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_room_geometry(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Amostra dimensões, RT60 e material de absorção da sala.

        Returns
        -------
        room_dims (3,), rt60 (s), absorption_coeffs dict para PRA
        """
        W = rng.uniform(*self.config.width_range)
        L = rng.uniform(*self.config.length_range)
        H = rng.uniform(*self.config.height_range)
        room_dims = np.array([W, L, H])

        rt60 = rng.uniform(*self.config.rt60_range)

        # Calcula coeficiente de absorção de Sabine correspondente
        # Fórmula de Eyring: α = 1 - exp(-0.163 * V / (S * RT60))
        V = W * L * H
        S = 2 * (W*L + W*H + L*H)   # área superficial total
        alpha_eyring = 1.0 - np.exp(-0.163 * V / (S * rt60 + 1e-9))
        alpha_eyring = float(np.clip(alpha_eyring, 0.01, 0.99))

        # PRA aceita coeficiente por parede ou material predefinido
        # Aqui usamos coeficiente uniforme para simplicidade controlada
        # (pode ser expandido para coeficientes por banda de oitava)
        materials = pra.Material(alpha_eyring)

        # Fallback: usa energia_absorption como dicionário por banda
        absorption_coeffs = {
            "material_obj":      materials,
        }

        return room_dims, rt60, absorption_coeffs

    def _create_pra_room(
        self,
        room_dims:        np.ndarray,
        absorption_coeffs: Dict,
        rt60:             float,
    ) -> pra.ShoeBox:
        """
        Instancia sala ShoeBox pyroomacoustics ultrarrápida.
        Ray Tracing e Air Absorption desativados para máxima velocidade.
        """
        # Limita a ordem de reflexão (4 é o padrão ideal para ML)
        max_order = 4

        room = pra.ShoeBox(
            room_dims.tolist(),
            fs            = self.config.sample_rate,
            materials     = absorption_coeffs["material_obj"],
            max_order     = max_order,
            ray_tracing   = False,          # DESATIVADO: economiza 90% do tempo de CPU
            air_absorption= False,          # DESATIVADO: irrelevante para TDoA/ILD em ML
            temperature   = 20.0,
        )

        return room

    def _place_microphone_array(
        self,
        room_dims: np.ndarray,
        rng:       np.random.Generator,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Posiciona array estéreo (2 mics) no centro da sala com orientação aleatória.

        Returns
        -------
        mic_positions : List[np.ndarray] de shape (3,) — posição de cada mic
        array_center  : np.ndarray shape (3,) — centro do array
        """
        baseline = rng.uniform(*self.config.mic_baseline_range)
        height   = rng.uniform(*self.config.mic_height_range)

        # Centro do array numa posição próxima ao centro da sala
        cx = rng.uniform(room_dims[0] * 0.3, room_dims[0] * 0.7)
        cy = rng.uniform(room_dims[1] * 0.3, room_dims[1] * 0.7)
        center = np.array([cx, cy, height])

        # Orientação do eixo do array no plano horizontal
        # 🟢 CORREÇÃO: Travar a orientação do array em 0.0 radianos (Eixo X).
        # Isso garante que a "frente" física dos microfones seja idêntica
        # ao referencial usado no cálculo do Rótulo de Azimute (Ground Truth).
        orientation_rad = 0.0

        half_base = np.array([
            np.cos(orientation_rad) * baseline / 2,
            np.sin(orientation_rad) * baseline / 2,
            0.0,
        ])

        mic0 = center - half_base
        mic1 = center + half_base

        # Garante que mics estão dentro da sala
        for mic in [mic0, mic1]:
            mic[:2] = np.clip(mic[:2], 0.1, room_dims[:2] - 0.1)
            mic[2]  = np.clip(mic[2],  0.1, room_dims[2]  - 0.1)

        return [mic0, mic1], center

    def _extract_isolated_sources(
        self,
        room:          pra.ShoeBox,
        source_audios: List[np.ndarray],
        mic_idx:       int,
        rng:           np.random.Generator,
    ) -> List[np.ndarray]:
        """
        Extrai o sinal reverberado de cada fonte individualmente no microfone `mic_idx`.
        Realiza convolução manual: x_i * h_{i, mic_idx}

        Essencial para calcular SI-SDR na fase de treino (ground truth BSS).
        """
        isolated = []

        for i, source in enumerate(room.sources):
            try:
                # Recupera a RIR da fonte i para o microfone mic_idx
                rir = room.rir[mic_idx][i]   # 1D array
            except (IndexError, AttributeError):
                logger.warning(f"RIR não disponível para source {i}, mic {mic_idx}.")
                isolated.append(np.zeros(self._n_samples, dtype=np.float32))
                continue

            import scipy.signal
            # Convolução via FFT (MUITO mais rápido que np.convolve direto)
            dry_signal = source_audios[i]
            wet = scipy.signal.fftconvolve(dry_signal, rir, mode='full')[:len(dry_signal)]

            wet = self._trim_or_pad(wet.reshape(1, -1), self._n_samples)[0]
            isolated.append(wet.astype(np.float32))

        return isolated

    @staticmethod
    def _trim_or_pad(signal: np.ndarray, target_len: int) -> np.ndarray:
        """
        Garante que o sinal tenha exatamente `target_len` amostras.
        Funciona para 1D e 2D (canais, amostras).
        """
        if signal.ndim == 1:
            if len(signal) >= target_len:
                return signal[:target_len]
            else:
                return np.pad(signal, (0, target_len - len(signal)))
        else:
            # 2D: (channels, samples)
            if signal.shape[1] >= target_len:
                return signal[:, :target_len]
            else:
                pad_w = target_len - signal.shape[1]
                return np.pad(signal, ((0, 0), (0, pad_w)))