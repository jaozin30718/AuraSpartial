"""
utils/geometry_utils.py
=======================
Funções auxiliares para posicionamento 3D, conversão de coordenadas
e cálculo de ângulos azimute/elevação.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def cartesian_to_spherical(
    source_pos: np.ndarray,
    array_center: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Converte coordenadas cartesianas em esféricas relativas ao array.

    Parameters
    ----------
    source_pos   : np.ndarray shape (3,) — posição da fonte [x, y, z]
    array_center : np.ndarray shape (3,) — centro do array de microfones

    Returns
    -------
    (azimuth_deg, elevation_deg, distance_m)
    Azimute  : 0° = frente (+Y), aumenta sentido horário visto de cima
    Elevação : 0° = horizonte, +90° = teto
    """
    delta = source_pos - array_center   # vetor relativo

    # distância euclidiana
    r = float(np.linalg.norm(delta))

    if r < 1e-9:
        return 0.0, 0.0, 0.0

    # azimute no plano XY (atan2 invertido para convenção acústica)
    azimuth_rad   = np.arctan2(delta[0], delta[1])       # frente = +Y
    elevation_rad = np.arcsin(np.clip(delta[2] / r, -1.0, 1.0))

    azimuth_deg   = float(np.degrees(azimuth_rad))  % 360.0
    elevation_deg = float(np.degrees(elevation_rad))

    return azimuth_deg, elevation_deg, r


def compute_tdoa(
    source_pos:  np.ndarray,
    mic_pos_1:   np.ndarray,
    mic_pos_2:   np.ndarray,
    speed_of_sound: float = 343.0,
    sample_rate: int      = 16_000,
) -> Tuple[float, float]:
    """
    Calcula o TDoA teórico entre dois microfones para uma fonte pontual.

    Returns
    -------
    (tdoa_seconds, tdoa_samples)
    """
    d1 = float(np.linalg.norm(source_pos - mic_pos_1))
    d2 = float(np.linalg.norm(source_pos - mic_pos_2))

    tdoa_s       = (d1 - d2) / speed_of_sound
    tdoa_samples = tdoa_s * sample_rate

    return tdoa_s, tdoa_samples


def sample_random_position_in_room(
    room_dims:      np.ndarray,          # [W, L, H]
    wall_margin:    float = 0.50,
    height_range:   Tuple[float, float] = (0.30, 2.00),
    forbidden_zones: List[np.ndarray] = None,  # posições já ocupadas
    min_separation:  float = 0.50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Amostra uma posição 3D válida dentro da sala, respeitando margens
    e zonas proibidas (posições já alocadas para outros elementos).

    Parameters
    ----------
    room_dims      : dimensões [W, L, H] em metros
    wall_margin    : margem mínima até qualquer parede
    height_range   : (min_z, max_z) em metros para posição da fonte
    forbidden_zones: lista de posições já ocupadas
    min_separation : distância mínima entre esta posição e cada zona proibida
    rng            : numpy Generator (para reprodutibilidade)

    Returns
    -------
    np.ndarray shape (3,) — posição válida [x, y, z]
    """
    if rng is None:
        rng = np.random.default_rng()

    forbidden_zones = forbidden_zones or []
    max_attempts = 500

    x_range = (wall_margin, room_dims[0] - wall_margin)
    y_range = (wall_margin, room_dims[1] - wall_margin)
    z_min   = max(height_range[0], wall_margin)
    z_max   = min(height_range[1], room_dims[2] - wall_margin)

    for _ in range(max_attempts):
        pos = np.array([
            rng.uniform(*x_range),
            rng.uniform(*y_range),
            rng.uniform(z_min, z_max),
        ])

        # Verifica separação de todas as zonas proibidas
        valid = all(
            np.linalg.norm(pos - fz) >= min_separation
            for fz in forbidden_zones
        )
        if valid:
            return pos

    raise RuntimeError(
        f"Não foi possível encontrar posição válida após {max_attempts} tentativas. "
        f"Sala muito pequena ou margem muito grande."
    )


def build_spline_trajectory(
    start_pos: np.ndarray,
    end_pos:   np.ndarray,
    n_points:  int,
    room_dims: np.ndarray,
    wall_margin: float = 0.30,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Gera uma trajetória suave (spline cúbica) para fonte em movimento.

    Returns
    -------
    np.ndarray shape (n_points, 3) — sequência de posições 3D
    """
    from scipy.interpolate import CubicSpline  # import local para evitar dep. global

    if rng is None:
        rng = np.random.default_rng()

    # Pontos de controle intermediários aleatórios
    n_control = max(2, n_points // 50)
    t_ctrl = np.linspace(0, 1, n_control + 2)

    # Interpola entre start e end com perturbações aleatórias
    waypoints = np.array([
        start_pos + t * (end_pos - start_pos) + rng.uniform(-0.5, 0.5, 3)
        for t in t_ctrl
    ])

    # Clipa dentro da sala
    waypoints[:, 0] = np.clip(waypoints[:, 0], wall_margin, room_dims[0] - wall_margin)
    waypoints[:, 1] = np.clip(waypoints[:, 1], wall_margin, room_dims[1] - wall_margin)
    waypoints[:, 2] = np.clip(waypoints[:, 2], wall_margin, room_dims[2] - wall_margin)

    # Força início e fim
    waypoints[0]  = start_pos
    waypoints[-1] = end_pos

    # Spline cúbica
    cs = CubicSpline(t_ctrl, waypoints, bc_type="clamped")
    t_fine = np.linspace(0, 1, n_points)

    trajectory = cs(t_fine)

    # Clip final de segurança
    trajectory[:, 0] = np.clip(trajectory[:, 0], wall_margin, room_dims[0] - wall_margin)
    trajectory[:, 1] = np.clip(trajectory[:, 1], wall_margin, room_dims[1] - wall_margin)
    trajectory[:, 2] = np.clip(trajectory[:, 2], wall_margin, room_dims[2] - wall_margin)

    return trajectory.astype(np.float32)


def generate_vad_mask(
    n_samples:           int,
    sample_rate:         int,
    silence_probability: float = 0.30,
    silence_min_ms:      float = 200.0,
    silence_max_ms:      float = 1500.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Gera máscara VAD binária com blocos de silêncio aleatórios.

    Returns
    -------
    np.ndarray shape (n_samples,) dtype bool — True = ativo, False = silêncio
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.ones(n_samples, dtype=bool)

    if rng.random() > silence_probability:
        return mask

    # Quantos blocos de silêncio inserir
    n_silence_blocks = rng.integers(1, 4)

    for _ in range(n_silence_blocks):
        dur_ms  = rng.uniform(silence_min_ms, silence_max_ms)
        dur_smp = int(dur_ms * sample_rate / 1000)
        start   = rng.integers(0, max(1, n_samples - dur_smp))
        end     = min(n_samples, start + dur_smp)
        mask[start:end] = False

    return mask