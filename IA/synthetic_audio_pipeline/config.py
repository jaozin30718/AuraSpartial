"""
config.py
=========
Configurações centralizadas do pipeline usando dataclasses tipadas.
Todas as constantes físicas e ranges de simulação são definidos aqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS
# ─────────────────────────────────────────────────────────────────────────────

SPEED_OF_SOUND: float = 343.0          # m/s @ 20°C
DEFAULT_SAMPLE_RATE: int = 16_000      # Hz — compatível com modelos de fala
MAX_ORDER_ISM: int = 17                # Ordem máxima do Image Source Method


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 1 — Motor Geométrico
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoomConfig:
    """Parâmetros de geometria e acústica da sala virtual."""

    # Dimensões da sala [min, max] em metros
    width_range:  Tuple[float, float] = (3.0,  20.0)
    length_range: Tuple[float, float] = (3.0,  20.0)
    height_range: Tuple[float, float] = (2.4,   6.0)

    # RT60 em segundos [anecóico → concreto vazio]
    rt60_range: Tuple[float, float] = (0.10, 1.50)

    # Microfones — array estéreo
    mic_baseline_range: Tuple[float, float] = (0.06, 0.12)   # distância entre mics (m)
    mic_height_range:   Tuple[float, float] = (0.80, 1.80)   # altura do array (m)

    # Número de fontes simultâneas
    n_sources_range: Tuple[int, int] = (2, 6)

    # Margem mínima da fonte até a parede (m)
    source_wall_margin: float = 0.50

    # Margem mínima da fonte até o array (m)
    source_mic_min_dist: float = 0.30

    # Tamanho do chunk de áudio por amostra
    audio_duration_seconds: float = 4.0

    # Frequência de amostragem
    sample_rate: int = DEFAULT_SAMPLE_RATE

    # Materiais de absorção predefinidos (nome → coeficiente de Sabine médio)
    # Valores aproximados para 1kHz
    absorption_presets: Dict[str, float] = field(default_factory=lambda: {
        "anechoic":         0.95,
        "recording_studio": 0.65,
        "carpeted_room":    0.45,
        "furnished_office": 0.35,
        "empty_office":     0.25,
        "classroom":        0.20,
        "living_room":      0.30,
        "tiled_bathroom":   0.10,
        "empty_concrete":   0.05,
    })


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 3 — Augmentation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugmentationConfig:
    """Parâmetros de toda a cadeia de degradação realista."""

    # AWGN — SNR aleatório [dB]
    awgn_snr_range: Tuple[float, float] = (-5.0, 30.0)
    awgn_probability: float = 0.90            # prob. de aplicar AWGN

    # Ruído difuso (babble/ambiente)
    diffuse_noise_snr_range: Tuple[float, float] = (5.0, 35.0)
    diffuse_noise_probability: float = 0.60

    # Hardware mismatch — ganho entre canais [dB]
    gain_mismatch_range: Tuple[float, float] = (-3.0, 3.0)
    gain_mismatch_probability: float = 0.70

    # Phase jitter — amostras de desalinhamento no canal R
    phase_jitter_samples_range: Tuple[float, float] = (0.0, 2.0)  # amostras fracionárias
    phase_jitter_probability: float = 0.0  # Desativado para preservar labels de DoA e TDoA

    # Clipping — limiar antes do hard/soft clip
    clipping_threshold_range: Tuple[float, float] = (0.5, 0.99)
    clipping_probability: float = 0.30
    clipping_type: str = "mixed"              # "hard" | "soft" | "mixed"

    # Fontes em movimento (Doppler)
    moving_source_probability: float = 0.20
    moving_source_speed_range: Tuple[float, float] = (0.5, 3.0)   # m/s

    # Probabilidade de silêncio por fonte (VAD sintético)
    silence_probability_per_source: float = 0.30
    silence_min_duration_ms: float = 200.0
    silence_max_duration_ms: float = 1500.0


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 5 — Otimização e Entrega
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuração global do pipeline de geração paralela."""

    room:         RoomConfig         = field(default_factory=RoomConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # I/O
    output_dir:       str = "./synthetic_dataset"
    audio_source_dir: str = "./audio_sources"      # LibriSpeech / UrbanSound8K
    noise_source_dir: str = "./noise_sources"      # Babble noise files

    # Paralelismo
    n_workers:       int = max(1, os.cpu_count() - 2)  # type: ignore[arg-type]
    samples_per_job: int = 50           # amostras por worker batch
    total_samples:   int = 100_000      # total de amostras a gerar

    # HDF5
    hdf5_chunk_size:    int = 100       # amostras por arquivo .h5
    compression:        str = "lzf"     # "lzf" | "gzip" | None
    compression_level:  int = 4

    # WebDataset (alternativa)
    use_webdataset:     bool = False
    tar_shard_size_mb:  int  = 512

    # Seed global (None = totalmente aleatório)
    global_seed: Optional[int] = None

    # Logging
    log_every_n: int = 100
    verbose:     bool = True