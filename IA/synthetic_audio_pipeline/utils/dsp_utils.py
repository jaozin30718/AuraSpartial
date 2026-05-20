"""
utils/dsp_utils.py
==================
Funções DSP core: geração de ruído, cálculo de energia, clipping,
jitter de fase e campo difuso.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def compute_rms(signal: np.ndarray) -> float:
    """Calcula RMS (Root Mean Square) de um sinal."""
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2) + 1e-12))


def db_to_linear(db: float) -> float:
    """Converte dB para escala linear de amplitude."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Converte amplitude linear para dB."""
    return 20.0 * np.log10(max(linear, 1e-12))


def generate_awgn(
    signal:      np.ndarray,
    snr_db:      float,
    rng:         np.random.Generator,
    independent_channels: bool = True,
) -> np.ndarray:
    """
    Gera ruído AWGN calibrado para uma SNR alvo.

    Parameters
    ----------
    signal    : np.ndarray shape (n_channels, n_samples) ou (n_samples,)
    snr_db    : SNR desejada em dB
    rng       : Generator para reprodutibilidade
    independent_channels : se True, gera ruído independente por canal

    Returns
    -------
    np.ndarray — ruído com mesma shape do sinal
    """
    signal_rms    = compute_rms(signal)
    target_snr    = db_to_linear(snr_db)

    if independent_channels and signal.ndim == 2:
        # Compute RMS and noise per channel
        noise = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            ch_rms = compute_rms(signal[i])
            noise_rms = ch_rms / (target_snr + 1e-12)
            noise[i] = rng.normal(0.0, noise_rms, signal.shape[1]).astype(np.float32)
    else:
        noise_rms = signal_rms / (target_snr + 1e-12)
        noise = rng.normal(0.0, noise_rms, signal.shape).astype(np.float32)

    return noise


def apply_hard_clipping(signal: np.ndarray, threshold: float) -> np.ndarray:
    """
    Hard clipping: limita brutalmente o sinal ao limiar ±threshold.
    Simula saturação de diafragma / ADC overflow.
    """
    return np.clip(signal, -threshold, threshold)


def apply_soft_clipping(signal: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft clipping via tangente hiperbólica.
    Distorção mais suave, simula compressão não-linear de driver MEMS.

    y = threshold * tanh(x / threshold)
    """
    return threshold * np.tanh(signal / (threshold + 1e-9)).astype(np.float32)


def apply_phase_jitter(
    signal:    np.ndarray,
    jitter_samples: float,
    sample_rate:    int,
) -> np.ndarray:
    """
    Aplica jitter de fase fracionário via interpolação sinc no domínio da frequência.

    Parameters
    ----------
    signal         : np.ndarray shape (n_samples,)
    jitter_samples : deslocamento fracionário em amostras (pode ser < 1.0)
    sample_rate    : taxa de amostragem

    Returns
    -------
    np.ndarray shape (n_samples,) — sinal com fase perturbada
    """
    n = len(signal)

    # Transformada de Fourier
    spectrum = np.fft.rfft(signal.astype(np.float64), n=n)

    # Vetor de frequências normalizado
    freqs = np.fft.rfftfreq(n)

    # Fase shift correspondente ao delay fracionário
    phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * jitter_samples)
    spectrum_shifted = spectrum * phase_shift

    result = np.fft.irfft(spectrum_shifted, n=n).astype(np.float32)
    return result


def generate_diffuse_noise_field(
    n_samples:   int,
    n_channels:  int,
    sample_rate: int,
    rng:         np.random.Generator,
    spectrum_type: str = "pink",
) -> np.ndarray:
    """
    Simula campo difuso esférico — versão vetorizada e otimizada.
    Gera todas as N_SOURCES fontes de uma vez via operações matriciais.
    """
    # Reduzido de 64 para 16: Economiza 75% do custo de Transformadas de Fourier (FFT)
    # mantendo a descorrelação espacial necessária para simular campo difuso.
    N_SOURCES = 16
    n_freq = n_samples // 2 + 1

    # Gera todas as N_SOURCES fontes de uma vez: shape (N_SOURCES, n_samples)
    white_batch = rng.normal(0.0, 1.0, (N_SOURCES, n_samples)).astype(np.float32)

    # FFT em batch: (N_SOURCES, n_freq)
    spectra = np.fft.rfft(white_batch, axis=1)

    # Coloração espectral vetorizada
    freqs = np.fft.rfftfreq(n_samples).astype(np.float32)  # (n_freq,)
    if spectrum_type == "pink":
        filt = np.where(freqs > 0, 1.0 / np.sqrt(freqs + 1e-12), 1.0)
    elif spectrum_type == "brown":
        filt = np.where(freqs > 0, 1.0 / (freqs + 1e-12), 1.0)
    else:  # white
        filt = np.ones(n_freq, dtype=np.float32)
    spectra *= filt[np.newaxis, :]  # broadcast (N_SOURCES, n_freq)

    # Atrasos de até ±6 amostras correspondem ao espaço acústico de um array de ~12cm
    delays = rng.uniform(-6.0, 6.0, (N_SOURCES, n_channels)).astype(np.float32)

    # Phase shift em batch: (N_SOURCES, n_channels, n_freq)
    # freqs: (n_freq,) -> (1, 1, n_freq)
    phase_shifts = np.exp(
        -1j * 2.0 * np.pi * freqs[np.newaxis, np.newaxis, :]
        * delays[:, :, np.newaxis]
    )  # (N_SOURCES, n_channels, n_freq)

    # Aplica delay por canal: spectra (N_SOURCES, 1, n_freq) * shifts
    shifted = spectra[:, np.newaxis, :] * phase_shifts  # (N_SOURCES, n_channels, n_freq)

    # iFFT em batch: (N_SOURCES, n_channels, n_samples)
    signals = np.fft.irfft(shifted, n=n_samples, axis=2).astype(np.float32)

    # Soma todas as fontes e normaliza
    diffuse = signals.sum(axis=0) / N_SOURCES  # (n_channels, n_samples)

    return diffuse



def normalize_signal(signal: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normaliza o RMS do sinal para um nível alvo."""
    current_rms = compute_rms(signal)
    return (signal * (target_rms / (current_rms + 1e-12))).astype(np.float32)


def compute_si_sdr(
    reference: np.ndarray,
    estimate:  np.ndarray,
) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio.
    Métrica de qualidade para separação de fontes.

    SDR escalado-invariante proposto por Le Roux et al. 2019.
    """
    reference = reference - np.mean(reference)
    estimate  = estimate  - np.mean(estimate)

    alpha     = np.dot(reference, estimate) / (np.dot(reference, reference) + 1e-12)
    target    = alpha * reference
    noise     = estimate - target

    si_sdr_val = 10.0 * np.log10(
        (np.dot(target, target) + 1e-12) /
        (np.dot(noise,  noise)  + 1e-12)
    )
    return float(si_sdr_val)