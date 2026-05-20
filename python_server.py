"""
Sistema Acústico v9 — Servidor Python
======================================
Correções v9 vs v8:
  • Header '<BIQHI>' alinhado com ESP32 v8 (19 bytes, uint32 sample_rate)
  • assert HEADER_SIZE == 19 (falha rápida)
  • Thread UDP dedicada + asyncio.Queue (sem bloqueio do event loop)
  • xcorr_fft: O(N log N) via FFT (substituí np.correlate)
  • dc_block_fast: scipy.lfilter em C (50× mais rápido)
  • asyncio.Lock para state compartilhado
  • soft_saturate com zona linear + knee
  • frame gap: máscara uint32 correta
  • history_db: deque(maxlen=100)
  • WebSocket throttle: broadcast a cada WS_THROTTLE_FRAMES
  • Validação n_samples no parse_binary
  • Protocolo PTP 4-timestamp (handler + envio de resposta)
  • Bounding box de sala para triangulação
  • Perda de pacote UDP logada
"""

import asyncio
import collections
import json
import math
import queue
import socket
import struct
import threading
import time
import traceback

import numpy as np
import scipy.signal as sp_signal

try:
    import sounddevice as sd
    PLAY_AUDIO = True
except ImportError:
    print("[AVISO] sounddevice não instalado — sem reprodução local.")
    PLAY_AUDIO = False


# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────
UDP_IP   = "0.0.0.0"
UDP_PORT = 5005
PTP_PORT = 5007   # porta no servidor para receber pings PTP
WS_PORT  = 8765
WS_THROTTLE_FRAMES = 3   # broadcast WS a cada N hops (~17 ms)

SAMPLE_RATE   = 44100
ANALYSIS_SIZE = 512
HOP_SIZE      = ANALYSIS_SIZE // 2  # 256

XCORR_MAX_LAG = 16
MIC_DIST_M    = 0.1

EMA_FAST = 0.3
EMA_SLOW = 0.1

# Geometria da sala (metros)
POS_NO_A  = (20.0, 12.5)  # Centro do galpão de 40x25 metros
ROOM_BBOX = (-1.0, -1.0, 41.0, 26.0)  # (x_min, y_min, x_max, y_max) ajustado pro galpão

SYNC_WINDOW_MS = 50

BAND_RANGES = {
    'low':  (80,   400),
    'mid':  (400,  3000),
    'high': (3000, 10000),
}

# ─────────────────────────────────────────────────────────────
# HEADER UDP
# '<BIQHH': B(1) I(4) Q(8) H(2) H(2) = 17 bytes
# ─────────────────────────────────────────────────────────────
HEADER_FMT  = '<BIQQHH'
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 25, (
    f"HEADER_SIZE={HEADER_SIZE} bytes, esperado 25! "
    f"Verifique HEADER_FMT='{HEADER_FMT}'"
)

MAX_PACKET_SAMPLES = 2048  # sanidade: rejeita pacotes gigantes


def parse_binary(data: bytes):
    """
    Desempacota header binário ESP32 3 Mics (25 bytes) e
    extrai amostras int16 → float64 normalizado.

    Retorna:
        (node_id, frame, ts_us, (left, right), sample_rate)
        ou (None, ...) se inválido.
    """
    if len(data) < HEADER_SIZE:
        return None, None, None, None, None

    node_id_byte, frame, ts_us_i2s0, ts_us_i2s1, n_samples, sr_rx = \
        struct.unpack_from(HEADER_FMT, data)

    node_id = chr(node_id_byte)

    # Validações de sanidade
    if node_id not in ('A', 'B', 'C', 'D'):
        return None, None, None, None, None
    if n_samples == 0 or n_samples > MAX_PACKET_SAMPLES:
        print(f"[PARSE] WARN: n_samples={n_samples} fora do range")
        return None, None, None, None, None
    if sr_rx not in (4410, 8000, 16000, 22050, 44100, 48000, 96000):
        print(f"[PARSE] WARN: sample_rate inesperado={sr_rx}")

    audio_bytes_needed = HEADER_SIZE + n_samples * 6
    if len(data) < audio_bytes_needed:
        return None, None, None, None, None

    raw = bytes(data[HEADER_SIZE: HEADER_SIZE + n_samples * 6])
    arr = np.frombuffer(raw, dtype=np.int16)

    mic1 = arr[0::3].astype(np.float64) / 32768.0
    mic2 = arr[1::3].astype(np.float64) / 32768.0
    mic3 = arr[2::3].astype(np.float64) / 32768.0

    return node_id, frame, ts_us_i2s0, (mic1, mic2, mic3), sr_rx


# ─────────────────────────────────────────────────────────────
# DC BLOCK — scipy.lfilter (C nativo, estado persistente)
# ─────────────────────────────────────────────────────────────
_dc_zi: dict = {}
_DC_B = np.array([1.0, -1.0])
_DC_A = np.array([1.0, -0.995])
_DC_ZI_TEMPLATE = sp_signal.lfilter_zi(_DC_B, _DC_A)


def dc_block_fast(x: np.ndarray, key: str) -> np.ndarray:
    """
    DC block IIR H(z)=(1-z⁻¹)/(1-0.995·z⁻¹), fc≈35 Hz.
    Implementação em C via scipy — ~50× mais rápido que loop Python.
    Estado persistente por chave (canal+nó).
    """
    global _dc_zi
    if key not in _dc_zi:
        _dc_zi[key] = _DC_ZI_TEMPLATE * x[0]
    y, _dc_zi[key] = sp_signal.lfilter(_DC_B, _DC_A, x, zi=_dc_zi[key])
    return y


# ─────────────────────────────────────────────────────────────
# HPF — Butterworth ordem 4, zi persistente por nó
# ─────────────────────────────────────────────────────────────
class HighPassFilter:
    def __init__(self, cutoff=80.0, fs=44100, order=4):
        nyq      = 0.5 * fs
        self.b, self.a = sp_signal.butter(
            order, cutoff / nyq, btype='high'
        )
        self._zi: dict = {}

    def process(self, nid, *channels):
        if nid not in self._zi:
            zi0 = sp_signal.lfilter_zi(self.b, self.a)
            self._zi[nid] = [zi0 * ch[0] for ch in channels]
        outs = []
        for i, ch in enumerate(channels):
            out, self._zi[nid][i] = sp_signal.lfilter(self.b, self.a, ch, zi=self._zi[nid][i])
            outs.append(out)
        return tuple(outs)


# ─────────────────────────────────────────────────────────────
# REDUÇÃO DE RUÍDO — OLA 50% + Wiener
# (código original mantido — sem bugs críticos nesta classe)
# ─────────────────────────────────────────────────────────────
class SpectralNoiseReducer:
    def __init__(self, sr=44100, fft_size=512, hop_size=None,
                 calib_time=10.0, smooth=0.91, over_sub=1.2,
                 noise_floor=0.05):
        self.sr          = sr
        self.fft_size    = fft_size
        self.hop_size    = hop_size or fft_size // 2
        self.smooth      = smooth
        self.over_sub    = over_sub
        self.noise_floor = noise_floor
        self.window      = np.hanning(fft_size).astype(np.float64)
        self.ola_norm    = np.sum(self.window ** 2) / self.hop_size
        self.calib_blocks_needed = max(
            10, int((calib_time * sr) / self.hop_size)
        )
        self._state: dict = {}

    def _init_node(self, nid, n_ch=3):
        n_bins = self.fft_size // 2 + 1
        self._state[nid] = {
            "calib_count": 0,
            "calibrated": False,
        }
        for i in range(n_ch):
            self._state[nid][f"noise_psd_{i}"] = np.full(n_bins, 1e-10)
            self._state[nid][f"gain_smooth_{i}"] = np.full(n_bins, self.noise_floor)
            self._state[nid][f"ola_buf_{i}"] = np.zeros(self.fft_size)
            self._state[nid][f"prev_{i}"] = np.zeros(self.hop_size)

    def _wiener_gain(self, sig_psd, noise_psd):
        snr  = sig_psd / (noise_psd * self.over_sub + 1e-12)
        return np.maximum(snr / (snr + 1.0), self.noise_floor)

    def _process_channel(self, nid, ch, samples):
        st       = self._state[nid]
        gk       = f"gain_smooth_{ch}"
        nk       = f"noise_psd_{ch}"
        ok       = f"ola_buf_{ch}"
        pk       = f"prev_{ch}"
        frame    = np.concatenate([st[pk], samples])
        windowed = frame * self.window
        spec     = np.fft.rfft(windowed)
        mag      = np.abs(spec)
        phase    = np.angle(spec)
        sig_psd  = mag ** 2

        if st["calibrated"]:
            raw_g    = self._wiener_gain(sig_psd, st[nk])
            st[gk]   = self.smooth * st[gk] + (1 - self.smooth) * raw_g
            gain     = st[gk]
        else:
            c        = st["calib_count"]
            st[nk]   = (c * st[nk] + sig_psd) / (c + 1)
            gain     = np.ones(len(mag))

        clean_spec  = (mag * gain) * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_spec) * self.window
        st[ok]     += clean_frame
        output      = st[ok][:self.hop_size] / self.ola_norm
        st[ok]      = np.roll(st[ok], -self.hop_size)
        st[ok][-self.hop_size:] = 0.0
        st[pk]      = samples.copy()
        return output

    def process(self, nid, *channels):
        if nid not in self._state:
            self._init_node(nid, len(channels))
        st = self._state[nid]
        if not st["calibrated"]:
            st["calib_count"] += 1
            if st["calib_count"] >= self.calib_blocks_needed:
                st["calibrated"] = True
                nf = [10 * np.log10(np.mean(st[f"noise_psd_{i}"]) + 1e-12) for i in range(len(channels))]
                nfs = " ".join([f"CH{i}={nf[i]:.1f}" for i in range(len(channels))])
                print(f"[NR] Nó '{nid}' calibrado | NF: {nfs} dBFS^2")
        return tuple(self._process_channel(nid, i, ch) for i, ch in enumerate(channels))


# ─────────────────────────────────────────────────────────────
# SOFT NOISE GATE — Envoltória IIR (mantido do v8, sem bugs)
# ─────────────────────────────────────────────────────────────
class SoftNoiseGate:
    def __init__(self, threshold_db=-55.0, attack_ms=5.0,
                 release_ms=80.0, ratio=0.05, sr=44100):
        self.threshold = 10.0 ** (threshold_db / 20.0)
        self.ratio     = ratio
        self.att_c     = np.exp(-1.0 / (sr * attack_ms  / 1000.0))
        self.rel_c     = np.exp(-1.0 / (sr * release_ms / 1000.0))
        self._env:  dict = {}
        self._gain: dict = {}

    def _init_node(self, nid, n_ch=3):
        self._env[nid] = [0.0] * n_ch
        self._gain[nid] = [self.ratio] * n_ch

    def _apply(self, samples, nid, ch):
        env = self._env[nid][ch]; gain = self._gain[nid][ch]
        out = np.empty_like(samples)
        for i, s in enumerate(samples):
            lv = abs(s)
            env = (self.att_c if lv > env else self.rel_c) * env + \
                  (1 - (self.att_c if lv > env else self.rel_c)) * lv
            tgt = 1.0 if env >= self.threshold else self.ratio
            gain = (self.att_c if tgt > gain else self.rel_c) * gain + \
                   (1 - (self.att_c if tgt > gain else self.rel_c)) * tgt
            out[i] = s * gain
        self._env[nid][ch] = env; self._gain[nid][ch] = gain
        return out

    def process(self, nid, *channels):
        if nid not in self._env:
            self._init_node(nid, len(channels))
        return tuple(self._apply(ch, nid, i) for i, ch in enumerate(channels))


# ─────────────────────────────────────────────────────────────
# CROSS-CORRELATION via FFT — O(N log N)
# ─────────────────────────────────────────────────────────────
def xcorr_fft(left: np.ndarray, right: np.ndarray,
              max_lag: int = XCORR_MAX_LAG) -> tuple[int, float]:
    """
    Correlação cruzada normalizada via FFT: O(N log N).
    Substitui np.correlate(mode='full') que é O(N²).
    """
    N    = len(left)
    norm = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if norm < 1e-12:
        return 0, 0.0

    nfft     = 1 << (2 * N - 1).bit_length()  # próxima potência de 2
    FL       = np.fft.rfft(left,  nfft)
    FR       = np.fft.rfft(right, nfft)
    corr     = np.fft.irfft(FL * np.conj(FR), nfft)
    # Rearranja: lags negativos vêm do final
    corr_full = np.concatenate([corr[-(N - 1):], corr[:N]]) / norm

    center = N - 1
    lo     = max(0,            center - max_lag)
    hi     = min(len(corr_full), center + max_lag + 1)
    region = corr_full[lo:hi]

    best_idx  = int(np.argmax(region))
    best_lag  = best_idx - (center - lo)
    coherence = float(region[best_idx])
    return best_lag, coherence


# ─────────────────────────────────────────────────────────────
# SOFT SATURATOR — zona linear + tanh acima do knee
# ─────────────────────────────────────────────────────────────
def soft_saturate(x: np.ndarray, drive_db=6.0,
                  makeup_db=-1.5, knee=0.7) -> np.ndarray:
    drive  = 10.0 ** (drive_db  / 20.0)
    makeup = 10.0 ** (makeup_db / 20.0)
    driven = x * drive
    mask   = np.abs(driven) < knee
    return np.where(mask, driven, np.tanh(driven)) * makeup


# ─────────────────────────────────────────────────────────────
# SEPARAÇÃO DE FONTES — Narrowband DoA + Histograma Angular
# ─────────────────────────────────────────────────────────────
class NarrowbandSourceSeparator:
    """
    Separa fontes sonoras usando DoA por banda de frequência.
    1. FFT de cada microfone
    2. Calcula diferença de fase entre pares de mics por bin
    3. Converte fase → ângulo de chegada (DoA)
    4. Constrói histograma angular ponderado pela magnitude
    5. Encontra picos = fontes distintas
    6. Calcula métricas (dB, freq dominante) por fonte
    """
    def __init__(self, sr=44100, fft_size=512, n_bins_hist=72,
                 min_db=-60.0, max_sources=3, peak_min_sep=10.0):
        self.sr = sr
        self.fft_size = fft_size
        self.n_bins_hist = n_bins_hist   # 72 bins = 5° por bin
        self.min_db = min_db
        self.max_sources = max_sources
        self.peak_min_sep = peak_min_sep  # separação mínima entre picos (graus)
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

    def process(self, channels):
        """
        channels: lista de 3 arrays numpy (mic1, mic2, mic3).
        Retorna lista de dicts, cada um representando uma fonte separada:
          { 'angle': float, 'db': float, 'dom_freq': float,
            'energy': float, 'confidence': float }
        """
        if len(channels) < 3:
            return []

        N = len(channels[0])
        if N < 64:
            return []

        # FFT dos 3 canais com janela para evitar vazamento
        window = np.hanning(N)
        specs = [np.fft.rfft(ch * window) for ch in channels]
        
        # Normalizar magnitude para que o dB reflita a escala temporal
        norm_factor = np.sum(window) / 2.0  # Fator de conversão (aproximado)
        mags = [np.abs(s) / norm_factor for s in specs]

        # Energia total média (para threshold)
        avg_mag = (mags[0] + mags[1] + mags[2]) / 3.0
        total_energy = float(np.sum(avg_mag ** 2))
        if total_energy < 1e-12:
            return []

        # Calcular DoA por bin de frequência usando fase entre pares
        # Par 1→2 e Par 1→3
        phase_12 = np.angle(specs[1] * np.conj(specs[0]))
        phase_13 = np.angle(specs[2] * np.conj(specs[0]))

        c = 343.0
        R = MIC_DIST_M

        # Converter diferenças de fase em ângulos (por bin de frequência)
        # Evitar divisão por zero nas freqs baixas
        omega = 2.0 * np.pi * self.freqs
        omega[0] = 1e-12  # DC

        # Time delays por bin
        tau_12 = phase_12 / omega
        tau_13 = phase_13 / omega

        # Ângulo por bin (mesma fórmula TDOA triangular)
        cos_t = (c / (math.sqrt(3) * R)) * (tau_13 - tau_12)
        sin_t = (c / (3 * R)) * (tau_12 + tau_13)
        angles_per_bin = np.degrees(np.arctan2(sin_t, cos_t))
        angles_per_bin = (angles_per_bin + 90) % 360  # Alinhar com canvas

        # Peso = magnitude média (ignora DC e freqs inválidas acima de Nyquist/2)
        weights = avg_mag.copy()
        weights[0] = 0  # ignorar DC
        # Ignorar bins abaixo de 80Hz e acima de 10kHz
        valid = (self.freqs >= 80) & (self.freqs <= 10000)
        weights[~valid] = 0

        # Construir histograma angular ponderado
        bin_edges = np.linspace(0, 360, self.n_bins_hist + 1)
        hist = np.zeros(self.n_bins_hist)
        energy_hist = np.zeros(self.n_bins_hist)  # Para dB por fonte
        freq_weight_hist = np.zeros(self.n_bins_hist)  # Para freq dominante

        for k in range(len(angles_per_bin)):
            if weights[k] < 1e-10:
                continue
            a = angles_per_bin[k] % 360
            b = int(a / 360.0 * self.n_bins_hist) % self.n_bins_hist
            hist[b] += weights[k]
            energy_hist[b] += weights[k] ** 2
            freq_weight_hist[b] += self.freqs[k] * weights[k]

        # Suavizar histograma (circular)
        kernel = np.array([0.15, 0.7, 0.15])
        hist_smooth = np.convolve(
            np.concatenate([hist[-1:], hist, hist[:1]]),
            kernel, mode='valid'
        )

        # Encontrar picos no histograma
        sources = []
        hist_copy = hist_smooth.copy()
        for _ in range(self.max_sources):
            peak_idx = int(np.argmax(hist_copy))
            peak_val = hist_copy[peak_idx]
            if peak_val < np.max(hist_smooth) * 0.15:  # threshold: 15% do pico máximo
                break

            angle = (peak_idx + 0.5) * (360.0 / self.n_bins_hist)

            # Energia desta fonte
            e = energy_hist[peak_idx]
            db = 20.0 * math.log10(math.sqrt(e) + 1e-12) if e > 1e-12 else -90.0

            # Freq dominante desta fonte
            dom_freq = freq_weight_hist[peak_idx] / (hist[peak_idx] + 1e-12)

            # Confiança: proporção da energia neste pico vs total
            conf = float(peak_val / (np.sum(hist_smooth) + 1e-12))

            if db > self.min_db:
                sources.append({
                    'angle': round(angle, 1),
                    'db': round(db, 1),
                    'dom_freq': round(max(0, dom_freq), 1),
                    'energy': round(float(e), 6),
                    'confidence': round(min(1.0, conf * 3.0), 3),
                })

            # Zerar região do pico encontrado (para achar o próximo)
            sep_bins = int(self.peak_min_sep / (360.0 / self.n_bins_hist))
            for j in range(-sep_bins, sep_bins + 1):
                idx = (peak_idx + j) % self.n_bins_hist
                hist_copy[idx] = 0

        return sources


# ─────────────────────────────────────────────────────────────
# AudioDSP — com xcorr_fft e deque
# ─────────────────────────────────────────────────────────────
class AudioDSP:
    def __init__(self, sr=SAMPLE_RATE):
        self.sr       = sr
        self.ema:      dict = {}
        self.noise:    dict = {}
        self.cal_buf:  dict = {}
        self.cal_done: dict = {}

    def _init_node(self, nid, n_ch=3):
        self.ema[nid] = {
            'db_L': -90.0, 'db_R': -90.0,
            'pk_L': -90.0, 'pk_R': -90.0,
            'angle': 0.0,  'conf': 0.0,
            'bands_L': np.full(3, -90.0),
            'bands_R': np.full(3, -90.0),
        }
        self.noise[nid] = {'rms': [0.0]*n_ch}
        self.cal_buf[nid] = [[] for _ in range(n_ch)]
        self.cal_done[nid] = False

    @staticmethod
    def _ema(prev, cur, alpha):
        return prev + alpha * (cur - prev)

    @staticmethod
    def _circular_ema(prev, cur, alpha):
        diff = (cur - prev + 180.0) % 360.0 - 180.0
        return (prev + alpha * diff) % 360.0

    @staticmethod
    def _safe_db(v):
        return 20.0 * math.log10(v) if v > 1e-9 else -90.0

    def _calibrate(self, nid, *channels):
        for i, ch in enumerate(channels):
            self.cal_buf[nid][i].append(ch)
        if len(self.cal_buf[nid][0]) >= 12:
            a = [np.concatenate(self.cal_buf[nid][i]) for i in range(len(channels))]
            for i in range(len(channels)):
                self.noise[nid]['rms'][i] = float(np.sqrt(np.mean(a[i]**2)))
            self.cal_done[nid] = True
            nfs = " ".join([f"CH{i}={self._safe_db(self.noise[nid]['rms'][i]):.1f}" for i in range(len(channels))])
            print(f"[CAL] Nó {nid}: NF {nfs} dB")
            del self.cal_buf[nid]

    def process(self, nid, *channels):
        if nid not in self.ema:
            self._init_node(nid, len(channels))
        N = len(channels[0])

        if not self.cal_done.get(nid, False):
            self._calibrate(nid, *[ch.copy() for ch in channels])

        channels = [ch - np.mean(ch) for ch in channels]
        left = channels[0]
        right = channels[1]

        zcr_L = int(np.sum(np.abs(np.diff(np.sign(left)))  > 0))
        zcr_R = int(np.sum(np.abs(np.diff(np.sign(right))) > 0))

        peak_L = float(np.max(np.abs(left)))
        peak_R = float(np.max(np.abs(right)))
        rms_L  = float(np.sqrt(np.mean(left**2)))
        rms_R  = float(np.sqrt(np.mean(right**2)))
        db_L   = self._safe_db(rms_L)
        db_R   = self._safe_db(rms_R)
        dbPk_L = self._safe_db(peak_L)
        dbPk_R = self._safe_db(peak_R)
        crest_L = float(peak_L / rms_L) if rms_L > 1e-9 else 0.0
        crest_R = float(peak_R / rms_R) if rms_R > 1e-9 else 0.0

        fft_L  = np.fft.rfft(left)
        fft_R  = np.fft.rfft(right)
        freqs  = np.fft.rfftfreq(N, 1.0 / self.sr)
        mag_L  = np.abs(fft_L) / (N / 2.0)
        mag_R  = np.abs(fft_R) / (N / 2.0)

        dom_freq_L = float(freqs[np.argmax(mag_L[1:]) + 1]) if N > 1 else 0.0
        dom_freq_R = float(freqs[np.argmax(mag_R[1:]) + 1]) if N > 1 else 0.0

        bands_L, bands_R = [], []
        for lo, hi in BAND_RANGES.values():
            mask = (freqs >= lo) & (freqs <= hi)
            eL = float(np.sqrt(np.mean(mag_L[mask]**2))) if np.any(mask) else 0.0
            eR = float(np.sqrt(np.mean(mag_R[mask]**2))) if np.any(mask) else 0.0
            bands_L.append(self._safe_db(eL))
            bands_R.append(self._safe_db(eR))

        is_silence = (db_L < -65.0 and db_R < -65.0)

        # DoA 3Mics (Triângulo M1, M2, M3)
        if is_silence:
            angle_raw = self.ema[nid]['angle']
            conf_raw = 0.0
            lag21, coh21, lag31, coh31 = 0, 0.0, 0, 0.0
            best_lag, coherence = 0, 0.0
        elif len(channels) >= 3:
            lag21, coh21 = xcorr_fft(channels[1], channels[0], XCORR_MAX_LAG)
            lag31, coh31 = xcorr_fft(channels[2], channels[0], XCORR_MAX_LAG)
            t21 = lag21 / self.sr
            t31 = lag31 / self.sr
            
            c = 343.0
            R = MIC_DIST_M # Assumindo MIC_DIST_M = raio do centro até cada mic
            
            cos_t = (c / (math.sqrt(3) * R)) * (t31 - t21)
            sin_t = (c / (3 * R)) * (t21 + t31)
            
            # Normalizar se exceder 1 (acontece com ruidos)
            norm = math.sqrt(cos_t**2 + sin_t**2)
            if norm > 0 and norm < 5.0: # Sanity check
                angle_raw = math.degrees(math.atan2(sin_t, cos_t))
                # Rotacionar +90 para alinhar com o eixo Y do canvas web
                angle_raw = (angle_raw + 90) % 360
            else:
                angle_raw = self.ema[nid]['angle'] # Mantem
            conf_raw = max(0.0, min(1.0, (coh21 + coh31) / 2.0))
        else:
            # Fallback 2 mics
            best_lag, coherence = xcorr_fft(left, right, XCORR_MAX_LAG)
            itd_s = best_lag / self.sr
            sin_a = max(-1.0, min(1.0, (itd_s * 343.0) / MIC_DIST_M))
            angle_raw = math.degrees(math.asin(sin_a))
            conf_raw  = coherence

        ild = db_R - db_L

        e = self.ema[nid]
        e['db_L']  = self._ema(e['db_L'],  db_L,   EMA_FAST)
        e['db_R']  = self._ema(e['db_R'],  db_R,   EMA_FAST)
        e['pk_L']  = self._ema(e['pk_L'],  dbPk_L, EMA_FAST)
        e['pk_R']  = self._ema(e['pk_R'],  dbPk_R, EMA_FAST)
        
        # Filtro de Jitter: Interpolação circular (protege a borda 360/0)
        # e atualiza forte apenas quando há confiança real no ângulo
        alpha_angle = EMA_SLOW * max(0.05, conf_raw)
        e['angle'] = self._circular_ema(e['angle'], angle_raw, alpha_angle)
        
        e['conf']  = self._ema(e['conf'],  conf_raw,  EMA_SLOW)
        for i in range(3):
            e['bands_L'][i] = self._ema(e['bands_L'][i], bands_L[i], EMA_FAST)
            e['bands_R'][i] = self._ema(e['bands_R'][i], bands_R[i], EMA_FAST)

        nf_L = self._safe_db(self.noise[nid]['rms'][0]) \
               if self.cal_done.get(nid) else -90.0
        nf_R = self._safe_db(self.noise[nid]['rms'][1]) \
               if self.cal_done.get(nid) else -90.0

        itd_lag_val = best_lag if len(channels) < 3 else lag21
        coh_val = coherence if len(channels) < 3 else conf_raw

        return {
            "peak_L":    round(peak_L,  6), "peak_R":    round(peak_R,  6),
            "rms_L":     round(rms_L,   6), "rms_R":     round(rms_R,   6),
            "crest_L":   round(crest_L, 2), "crest_R":   round(crest_R, 2),
            "zcr_L":     zcr_L,             "zcr_R":     zcr_R,
            "db_L":      round(float(e['db_L']), 1),
            "db_R":      round(float(e['db_R']), 1),
            "low_L":     round(float(e['bands_L'][0]), 1),
            "mid_L":     round(float(e['bands_L'][1]), 1),
            "high_L":    round(float(e['bands_L'][2]), 1),
            "low_R":     round(float(e['bands_R'][0]), 1),
            "mid_R":     round(float(e['bands_R'][1]), 1),
            "high_R":    round(float(e['bands_R'][2]), 1),
            "dom_freq_L": round(dom_freq_L, 1),
            "dom_freq_R": round(dom_freq_R, 1),
            "angle":     round(float(e['angle']), 1),
            "ild":       round(ild, 2),
            "itd_lag":   itd_lag_val,
            "coherence": round(coh_val, 3),
            "confidence":round(float(e['conf']), 3),
            "dbPk_L":    round(float(e['pk_L']), 1),
            "dbPk_R":    round(float(e['pk_R']), 1),
            "noise_L_db":round(nf_L, 1),
            "noise_R_db":round(nf_R, 1),
        }


# ─────────────────────────────────────────────────────────────
# TRIANGULAÇÃO — com validação de bounding box
# ─────────────────────────────────────────────────────────────
def triangulate(angA, angB, posA, posB):
    radA = math.radians(90.0 - angA)
    radB = math.radians(90.0 - angB)
    det  = math.cos(radB) * math.sin(radA) \
         - math.sin(radB) * math.cos(radA)
    if abs(det) < 1e-6:
        return None
    dx = posB[0] - posA[0]
    dy = posB[1] - posA[1]
    u  = (dy * math.cos(radB) - dx * math.sin(radB)) / det
    if u < 0:
        return None
    x = posA[0] + u * math.cos(radA)
    y = posA[1] + u * math.sin(radA)
    # Valida bounding box da sala
    xmin, ymin, xmax, ymax = ROOM_BBOX
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return None
    return (x, y)


# ─────────────────────────────────────────────────────────────
# INSTÂNCIAS DSP GLOBAIS
# ─────────────────────────────────────────────────────────────
hpf = HighPassFilter(cutoff=80.0, fs=SAMPLE_RATE, order=4)
noise_reducer = SpectralNoiseReducer(
    sr=SAMPLE_RATE, fft_size=ANALYSIS_SIZE, hop_size=HOP_SIZE,
    calib_time=10.0, smooth=0.91, over_sub=4.0, noise_floor=0.005,
)
soft_gate = SoftNoiseGate(
    threshold_db=-55.0, attack_ms=5.0, release_ms=80.0,
    ratio=0.05, sr=SAMPLE_RATE,
)
dsp = AudioDSP(sr=SAMPLE_RATE)
source_sep = NarrowbandSourceSeparator(
    sr=SAMPLE_RATE, fft_size=ANALYSIS_SIZE,
    n_bins_hist=72, min_db=-60.0, max_sources=3, peak_min_sep=15.0,
)


# ─────────────────────────────────────────────────────────────
# ESTADO GLOBAL
# ─────────────────────────────────────────────────────────────
accum_nr:  dict = {}
accum_dsp: dict = {}
last_frame_per_node: dict = {}

state = {
    "nodes":  {
        "A": {"raw": None,
              "history_db": collections.deque(maxlen=100),
              "peak": -90.0, "avg": -90.0},
    },
    "events": [],
}
_state_lock = None   # inicializado no main() (asyncio.Lock)

frames_total  = 0
ws_frame_cnt: dict = {}  # throttle por nó
connected:    set  = set()


# ─────────────────────────────────────────────────────────────
# PLAYBACK LOCAL
# ─────────────────────────────────────────────────────────────
audio_q    = queue.Queue(maxsize=300)
_pb_buffer = np.zeros((0, 2), dtype=np.float32)

def _audio_callback(outdata, frames, time_info, status):
    global _pb_buffer
    if status:
        print(f"[PLAYBACK] {status}")
    try:
        while not audio_q.empty():
            _pb_buffer = np.concatenate(
                (_pb_buffer, audio_q.get_nowait())
            )
        if len(_pb_buffer) >= frames:
            outdata[:] = _pb_buffer[:frames]
            _pb_buffer  = _pb_buffer[frames:]
        else:
            n = len(_pb_buffer)
            if n > 0:
                outdata[:n] = _pb_buffer
                outdata[n:] = 0.0
                _pb_buffer  = np.zeros((0, 2), dtype=np.float32)
            else:
                outdata[:] = 0.0
    except Exception as exc:
        outdata[:] = 0.0
        print(f"[PLAYBACK CB] {exc}")

def _playback_worker():
    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=2,
            dtype='float32', blocksize=1024,
            latency='high', callback=_audio_callback,
        ):
            print("[PLAYBACK] Stream ativo.")
            while audio_q.qsize() < 40:
                time.sleep(0.01)
            while True:
                time.sleep(1)
    except Exception as exc:
        print(f"[PLAYBACK] Erro: {exc}")

if PLAY_AUDIO:
    threading.Thread(target=_playback_worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# PTP — Thread dedicada para responder pings de sincronização
# ─────────────────────────────────────────────────────────────
def _ptp_worker():
    """
    Thread que recebe pings PTP dos ESP32 e responde com
    3 timestamps (t1_echo, t2, t3) para cálculo de offset.
    """
    ptp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ptp_sock.bind((UDP_IP, PTP_PORT))
    ptp_sock.settimeout(1.0)
    print(f"[PTP] Servidor escutando na porta {PTP_PORT}")

    while True:
        try:
            data, addr = ptp_sock.recvfrom(64)
            if len(data) < 9:
                continue
            t1_us        = struct.unpack_from('<q', data, 0)[0]
            t2_us        = int(time.time() * 1e6)   # recebimento
            node_id      = chr(data[8])
            t3_us        = int(time.time() * 1e6)   # envio
            response     = struct.pack('<qqq', t1_us, t2_us, t3_us)
            ptp_sock.sendto(response, (addr[0], 5006))
            print(f"[PTP] Nó '{node_id}' | RTT_proc="
                  f"{t3_us - t2_us} µs | ip={addr[0]}")
        except socket.timeout:
            pass
        except Exception as exc:
            print(f"[PTP] Erro: {exc}")

threading.Thread(target=_ptp_worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# on_audio — Acumula e analisa (com lock)
# ─────────────────────────────────────────────────────────────
def _compute_on_audio(node_id, frame, ts_us, *channels):
    """Função síncrona de análise — chamada via run_in_executor."""
    global frames_total

    if node_id not in accum_dsp:
        accum_dsp[node_id] = {"ch": [[] for _ in channels], "ts": ts_us}

    for i, ch in enumerate(channels):
        accum_dsp[node_id]["ch"][i].append(ch)

    total = sum(len(b) for b in accum_dsp[node_id]["ch"][0])
    if total < ANALYSIS_SIZE:
        return None

    flats = [np.concatenate(accum_dsp[node_id]["ch"][i]) for i in range(len(channels))]
    ts_blk = accum_dsp[node_id]["ts"]

    accum_dsp[node_id] = {
        "ch": [[f[ANALYSIS_SIZE:]] for f in flats],
        "ts": ts_us,
    }

    analysis_chs = [f[:ANALYSIS_SIZE] for f in flats]
    result = dsp.process(node_id, *analysis_chs)
    result['timestamp'] = ts_blk
    result['frame'] = frame
    frames_total += 1

    # Separação de fontes (Narrowband DoA)
    # Só roda se houver som real (evita fontes fantasma no silêncio)
    db_avg_quick = (result.get('db_L', -90) + result.get('db_R', -90)) / 2.0
    if db_avg_quick > -45.0:
        result['sources'] = source_sep.process(analysis_chs)
    else:
        result['sources'] = []

    return result


async def on_audio(node_id, frame, ts_us, *channels):
    """Wrapper assíncrono com lock para atualização do state."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _compute_on_audio, node_id, frame, ts_us, *channels
    )
    if result is None:
        return None

    async with _state_lock:
        if node_id not in state["nodes"]:
            state["nodes"][node_id] = {
                "raw": None,
                "history_db": collections.deque(maxlen=100),
                "peak": -90.0, "avg": -90.0,
            }
        node   = state["nodes"][node_id]
        node["raw"] = result
        db_avg = (result["db_L"] + result["db_R"]) / 2.0
        node["history_db"].append(db_avg)
        if db_avg > node["peak"]:
            node["peak"] = db_avg
        node["avg"] = round(
            sum(node["history_db"]) / len(node["history_db"]), 1
        )

        # Gerar eventos individuais POR FONTE separada
        now_ts = time.time()
        for src in result.get('sources', []):
            src_ang = src['angle']
            src_db = src['db']
            src_conf = src['confidence']
            if src_db <= -45.0 or src_conf < 0.1:
                continue

            # Distância por SPL da fonte individual
            dist_spl = max(0.5, min(40.0, 10.0 ** ((-40.0 - src_db) / 20.0)))

            # Distância por coerência da fonte (DRR)
            coh = max(0.01, min(1.0, src_conf))
            dist_coh = 0.5 * math.exp(4.5 * (1.0 - coh))
            dist_est = max(0.5, min(40.0, 0.4 * dist_spl + 0.6 * dist_coh))

            # Ângulo 360° → coordenadas do mundo
            # 0° = Norte (+Y), 90° = Leste (+X), 180° = Sul (-Y), 270° = Oeste (-X)
            ang_rad = math.radians(src_ang)
            x_est = POS_NO_A[0] + dist_est * math.sin(ang_rad)
            y_est = POS_NO_A[1] + dist_est * math.cos(ang_rad)

            # Clamp ao bounding box
            xmin, ymin, xmax, ymax = ROOM_BBOX
            x_est = max(xmin, min(xmax, x_est))
            y_est = max(ymin, min(ymax, y_est))

            state["events"].append({
                "x": round(x_est, 2),
                "y": round(y_est, 2),
                "intensity": round(src_db, 1),
                "confidence": round(src_conf, 2),
                "dom_freq": src.get('dom_freq', 0),
                "ts": now_ts,
            })

        now = time.time()
        state["events"] = [e for e in state["events"]
                           if now - e["ts"] < 2.0]
        # Limitar para evitar JSON gigante no WebSocket
        if len(state["events"]) > 50:
            state["events"] = state["events"][-50:]

    return state


# ─────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────
async def handle_client(ws):
    connected.add(ws)
    print(f"[WS] +cliente (total={len(connected)})")
    try:
        await ws.wait_closed()
    finally:
        connected.discard(ws)
        print(f"[WS] -cliente (total={len(connected)})")


async def broadcast(msg: str):
    if not connected:
        return
    clients = list(connected)
    results = await asyncio.gather(
        *[c.send(msg) for c in clients],
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"[WS] Envio falhou: {r}")


# ─────────────────────────────────────────────────────────────
# THREAD UDP DEDICADA
# ─────────────────────────────────────────────────────────────
udp_data_queue: asyncio.Queue  # inicializado no main()

def _udp_receiver_thread(loop_ref, q_ref, sock_ref):
    print(f"[UDP-THREAD] Recebendo em {UDP_IP}:{UDP_PORT}")
    while True:
        try:
            data, _ = sock_ref.recvfrom(4096)
            loop_ref.call_soon_threadsafe(q_ref.put_nowait, data)
        except Exception as exc:
            print(f"[UDP-THREAD] {exc}")
            time.sleep(0.001)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
async def process_packet(data: bytes):
    """Processa um pacote UDP: parse → HPF → NR → gate → DSP → WS."""
    global frames_total

    node_id, frame, ts_us, audio, sr_rx = parse_binary(data)
    if node_id is None:
        return
    mic1_raw, mic2_raw, mic3_raw = audio

    # Detecção de pacotes perdidos (uint32, máscara correta)
    if node_id in last_frame_per_node:
        gap = (frame - last_frame_per_node[node_id]) & 0xFFFFFFFF
        if gap > 1:
            print(f"[GAP] Nó {node_id}: {gap-1} pacote(s) perdido(s) "
                  f"(frame #{frame})")
            ramp      = np.linspace(0.0, 1.0, len(mic1_raw))
            mic1_raw  = mic1_raw  * ramp
            mic2_raw  = mic2_raw  * ramp
            mic3_raw  = mic3_raw  * ramp
    last_frame_per_node[node_id] = frame

    # 1. HPF 80 Hz
    hp = hpf.process(node_id, mic1_raw, mic2_raw, mic3_raw)

    # 2. Acumula para NR (hops de HOP_SIZE)
    if node_id not in accum_nr:
        accum_nr[node_id] = [[] for _ in range(3)]
    for i in range(3):
        accum_nr[node_id][i].append(hp[i])

    total_nr = sum(len(b) for b in accum_nr[node_id][0])
    nr_out = [[] for _ in range(3)]

    while total_nr >= HOP_SIZE:
        flats = [np.concatenate(accum_nr[node_id][i]) for i in range(3)]
        hops = [f[:HOP_SIZE] for f in flats]
        accum_nr[node_id] = [[f[HOP_SIZE:]] for f in flats]
        total_nr -= HOP_SIZE

        cleans = noise_reducer.process(node_id, *hops)
        gated = soft_gate.process(node_id, *cleans)
        for i in range(3):
            nr_out[i].append(gated[i])

    if not nr_out[0]:
        return

    outs = [np.concatenate(nr_out[i]) for i in range(3)]

    # 3. Soft saturação (tanh com zona linear)
    outs = [soft_saturate(o, drive_db=6.0, makeup_db=-1.5, knee=0.7) for o in outs]

    # 4. Playback (Nó A)
    if PLAY_AUDIO and node_id == "A":
        pb_L = dc_block_fast(outs[0], "A_L")
        pb_R = dc_block_fast(outs[1], "A_R")
        chunk = np.column_stack((
            pb_L.astype(np.float32),
            pb_R.astype(np.float32),
        ))
        try:
            audio_q.put_nowait(chunk)
        except queue.Full:
            pass  # descarte silencioso por jitter severo

    # 5. Análise DSP + state update
    packet = await on_audio(node_id, frame, ts_us, outs[0], outs[1], outs[2])

    # 6. Broadcast WebSocket com throttle
    if packet and connected:
        ws_frame_cnt[node_id] = ws_frame_cnt.get(node_id, 0) + 1
        if ws_frame_cnt[node_id] >= WS_THROTTLE_FRAMES:
            ws_frame_cnt[node_id] = 0
            msg = json.dumps(
                packet, 
                separators=(',', ':'),
                default=lambda o: list(o) if isinstance(o, collections.deque) else str(o)
            )
            await broadcast(msg)


async def main_loop():
    last_log = time.time()
    while True:
        try:
            data = await asyncio.wait_for(
                udp_data_queue.get(), timeout=1.0
            )
            await process_packet(data)
        except asyncio.TimeoutError:
            pass
        except Exception as exc:
            print(f"[LOOP] {exc}")
            traceback.print_exc()
            await asyncio.sleep(0.01)

        # Log periódico
        now = time.time()
        if now - last_log >= 3.0:
            last_log = now
            parts = []
            for nid, st in noise_reducer._state.items():
                cal  = "[OK]" if st["calibrated"] else f"[{st['calib_count']}]"
                gm_L = float(np.mean(st["gain_smooth_0"]))
                gm_R = float(np.mean(st["gain_smooth_1"]))
                parts.append(f"{nid}:cal={cal} g=L{gm_L:.2f}/R{gm_R:.2f}")
            raw_a  = state["nodes"].get("A", {}).get("raw")
            db_str = ""
            if raw_a:
                db_str = (f" | dB={raw_a['db_L']:.1f}/{raw_a['db_R']:.1f}"
                          f" ang={raw_a['angle']:.1f}deg")
            print(f"[INFO] frames={frames_total} | "
                  f"{' | '.join(parts)}{db_str} | "
                  f"ev={len(state['events'])} ws={len(connected)}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
async def main():
    global udp_data_queue, _state_lock

    _state_lock    = asyncio.Lock()
    udp_data_queue = asyncio.Queue(maxsize=500)
    loop           = asyncio.get_event_loop()

    # Socket bloqueante para a thread UDP dedicada
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_IP, UDP_PORT))

    threading.Thread(
        target=_udp_receiver_thread,
        args=(loop, udp_data_queue, udp_sock),
        daemon=True,
    ).start()

    print("=" * 60)
    print("  Sistema Acústico v9 — Servidor DSP")
    print(f"  UDP:{UDP_PORT} | WS:{WS_PORT} | PTP:{PTP_PORT}")
    print(f"  HEADER: {HEADER_FMT} = {HEADER_SIZE} bytes")
    print(f"  Análise: {ANALYSIS_SIZE} smp = "
          f"{ANALYSIS_SIZE/SAMPLE_RATE*1000:.1f} ms")
    print(f"  Hop NR : {HOP_SIZE} smp = "
          f"{HOP_SIZE/SAMPLE_RATE*1000:.1f} ms")
    print("=" * 60)

    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT):
        await main_loop()


if __name__ == "__main__":
    import websockets
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[EXIT] Encerrado.")