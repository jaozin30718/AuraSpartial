import math
import numpy as np
import scipy.signal as sp_signal
from config import (
    SAMPLE_RATE, ANALYSIS_SIZE, HOP_SIZE, XCORR_MAX_LAG, MIC_DIST_M, EMA_FAST, EMA_SLOW, BAND_RANGES, ROOM_BBOX,
    HPF_CUTOFF, HPF_ORDER, DC_BLOCK_POLE,
    NR_CALIB_TIME, NR_SMOOTH, NR_OVER_SUB, NR_NOISE_FLOOR,
    GATE_THRESHOLD_DB, GATE_ATTACK_MS, GATE_RELEASE_MS, GATE_RATIO,
    SRC_MIN_DB, SRC_MAX_COUNT, SRC_PEAK_MIN_SEP
)

# ─────────────────────────────────────────────────────────────
# DC BLOCK — Filtro IIR de rejeição de componente DC
# ─────────────────────────────────────────────────────────────
_dc_zi = {}
_DC_B = np.array([1.0, -1.0])
_DC_A = np.array([1.0, -DC_BLOCK_POLE])
_DC_ZI_TEMPLATE = sp_signal.lfilter_zi(_DC_B, _DC_A)


def dc_block_fast(x: np.ndarray, key: str) -> np.ndarray:
    """
    DC block IIR H(z)=(1-z⁻¹)/(1-0.995·z⁻¹), fc≈35 Hz.
    Implementação rápida via scipy.lfilter com estado persistente.
    """
    global _dc_zi
    if key not in _dc_zi:
        _dc_zi[key] = _DC_ZI_TEMPLATE * x[0]
    y, _dc_zi[key] = sp_signal.lfilter(_DC_B, _DC_A, x, zi=_dc_zi[key])
    return y


# ─────────────────────────────────────────────────────────────
# HPF — Filtro Passa-Altas Butterworth Ordem 4
# ─────────────────────────────────────────────────────────────
class HighPassFilter:
    def __init__(self, cutoff=HPF_CUTOFF, fs=44100, order=HPF_ORDER):
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
# REDUÇÃO DE RUÍDO — Wiener OLA 50%
# ─────────────────────────────────────────────────────────────
class SpectralNoiseReducer:
    def __init__(self, sr=44100, fft_size=512, hop_size=None,
                 calib_time=NR_CALIB_TIME, smooth=NR_SMOOTH, over_sub=NR_OVER_SUB,
                 noise_floor=NR_NOISE_FLOOR):
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
# SOFT NOISE GATE — Envoltória IIR e atenuação suave no silêncio
# ─────────────────────────────────────────────────────────────
class SoftNoiseGate:
    def __init__(self, threshold_db=GATE_THRESHOLD_DB, attack_ms=GATE_ATTACK_MS,
                 release_ms=GATE_RELEASE_MS, ratio=GATE_RATIO, sr=44100):
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
# CORRELAÇÃO CRUZADA GCC-PHAT via FFT
# ─────────────────────────────────────────────────────────────
def xcorr_fft(left: np.ndarray, right: np.ndarray, max_lag: int = XCORR_MAX_LAG) -> tuple[int, float]:
    """
    Algoritmo GCC-PHAT (Generalized Cross-Correlation) O(N log N) via FFT.
    """
    N    = len(left)
    window = np.hanning(N)

    nfft     = 1 << (2 * N - 1).bit_length()
    FL       = np.fft.rfft(left * window, nfft)
    FR       = np.fft.rfft(right * window, nfft)
    
    cross = FL * np.conj(FR)
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    gcc_phat = cross / mag
    
    corr = np.fft.irfft(gcc_phat, nfft)
    corr_full = np.concatenate([corr[-(N - 1):], corr[:N]])

    center = N - 1
    lo     = max(0,            center - max_lag)
    hi     = min(len(corr_full), center + max_lag + 1)
    region = corr_full[lo:hi]

    best_idx  = int(np.argmax(region))
    best_lag  = best_idx - (center - lo)
    coherence = float(region[best_idx])
    return best_lag, coherence


# ─────────────────────────────────────────────────────────────
# SOFT SATURATOR — Compressão e corte suave analógico
# ─────────────────────────────────────────────────────────────
def soft_saturate(x: np.ndarray, drive_db=6.0,
                  makeup_db=-1.5, knee=0.7) -> np.ndarray:
    drive  = 10.0 ** (drive_db  / 20.0)
    makeup = 10.0 ** (makeup_db / 20.0)
    driven = x * drive
    mask   = np.abs(driven) < knee
    return np.where(mask, driven, np.tanh(driven)) * makeup


# ─────────────────────────────────────────────────────────────
# SEPARAÇÃO DE FONTES ACOUSTICAS — Narrowband DoA (360°)
# ─────────────────────────────────────────────────────────────
class NarrowbandSourceSeparator:
    def __init__(self, sr=44100, fft_size=512, n_bins_hist=72,
                 min_db=SRC_MIN_DB, max_sources=SRC_MAX_COUNT, peak_min_sep=SRC_PEAK_MIN_SEP):
        self.sr = sr
        self.fft_size = fft_size
        self.n_bins_hist = n_bins_hist
        self.min_db = min_db
        self.max_sources = max_sources
        self.peak_min_sep = peak_min_sep
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

    def process(self, channels):
        if len(channels) < 2:
            return []

        N = len(channels[0])
        if N < 64:
            return []

        window = np.hanning(N)
        specs = [np.fft.rfft(ch * window) for ch in channels]
        
        norm_factor = np.sum(window) / 2.0
        mags = [np.abs(s) / norm_factor for s in specs]

        avg_mag = np.mean(mags, axis=0)
        total_energy = float(np.sum(avg_mag ** 2))
        if total_energy < 1e-12:
            return []

        phase_RL = np.angle(specs[1] * np.conj(specs[0]))

        omega = 2.0 * np.pi * self.freqs
        omega[0] = 1e-12

        tau_RL = phase_RL / omega
        
        if len(channels) >= 3:
            spec_center = (specs[0] + specs[1]) / 2.0
            phase_FC = np.angle(specs[2] * np.conj(spec_center))
            tau_FC = phase_FC / omega
            X = tau_RL
            Y = tau_FC / (math.sqrt(3) / 2.0)
            angles_per_bin = np.degrees(np.arctan2(X, Y))
        else:
            sin_vals = (tau_RL * 343.0) / MIC_DIST_M
            sin_vals = np.clip(sin_vals, -1.0, 1.0)
            angles_per_bin = 180.0 - np.degrees(np.arcsin(sin_vals))
            
        angles_per_bin = (angles_per_bin + 360) % 360

        weights = avg_mag.copy()
        weights[0] = 0
        valid = (self.freqs >= 80) & (self.freqs <= 900)
        weights[~valid] = 0

        hist = np.zeros(self.n_bins_hist)
        energy_hist = np.zeros(self.n_bins_hist)
        freq_weight_hist = np.zeros(self.n_bins_hist)

        for k in range(len(angles_per_bin)):
            if weights[k] < 1e-10:
                continue
            a = angles_per_bin[k] % 360
            b = int(a / 360.0 * self.n_bins_hist) % self.n_bins_hist
            hist[b] += weights[k]
            energy_hist[b] += weights[k] ** 2
            freq_weight_hist[b] += self.freqs[k] * weights[k]

        kernel = np.array([0.15, 0.7, 0.15])
        hist_smooth = np.convolve(
            np.concatenate([hist[-1:], hist, hist[:1]]),
            kernel, mode='valid'
        )

        sources = []
        hist_copy = hist_smooth.copy()
        for _ in range(self.max_sources):
            peak_idx = int(np.argmax(hist_copy))
            peak_val = hist_copy[peak_idx]
            if peak_val < np.max(hist_smooth) * 0.15:
                break

            angle = (peak_idx + 0.5) * (360.0 / self.n_bins_hist)

            e = energy_hist[peak_idx]
            db = 20.0 * math.log10(math.sqrt(e) + 1e-12) if e > 1e-12 else -90.0

            dom_freq = freq_weight_hist[peak_idx] / (hist[peak_idx] + 1e-12)

            conf = float(peak_val / (np.sum(hist_smooth) + 1e-12))

            if db > self.min_db:
                sources.append({
                    'angle': round(angle, 1),
                    'db': round(db, 1),
                    'dom_freq': round(max(0, dom_freq), 1),
                    'energy': round(float(e), 6),
                    'confidence': round(min(1.0, conf * 3.0), 3),
                })

            sep_bins = int(self.peak_min_sep / (360.0 / self.n_bins_hist))
            for j in range(-sep_bins, sep_bins + 1):
                idx = (peak_idx + j) % self.n_bins_hist
                hist_copy[idx] = 0

        return sources


# ─────────────────────────────────────────────────────────────
# AUDIODSP — Processador e analisador acústico principal do Nó
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

        # DoA 3Mics Broadband (Geometria L/R/F)
        if is_silence:
            angle_raw = self.ema[nid]['angle']
            conf_raw = 0.0
            lag_RL, coh_RL = 0, 0.0
            best_lag, coherence = 0, 0.0
        elif len(channels) >= 3:
            angle_raw = self.ema[nid]['angle']
            conf_raw  = self.ema[nid]['conf']
            lag_RL, coh_RL = 0, 0.0
            best_lag, coherence = 0, 0.0
        else:
            best_lag, coherence = xcorr_fft(left, right, XCORR_MAX_LAG)
            itd_s = best_lag / self.sr
            sin_a = max(-1.0, min(1.0, (itd_s * 343.0) / MIC_DIST_M))
            
            angle_raw = 180.0 - math.degrees(math.asin(sin_a))
            angle_raw = (angle_raw + 360.0) % 360.0
            conf_raw  = coherence

        ild = db_R - db_L

        e = self.ema[nid]
        e['db_L']  = self._ema(e['db_L'],  db_L,   EMA_FAST)
        e['db_R']  = self._ema(e['db_R'],  db_R,   EMA_FAST)
        e['pk_L']  = self._ema(e['pk_L'],  dbPk_L, EMA_FAST)
        e['pk_R']  = self._ema(e['pk_R'],  dbPk_R, EMA_FAST)
        
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

        itd_lag_val = best_lag if len(channels) < 3 else lag_RL
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
# TRIANGULAÇÃO ACOUSTICA — Cruzamento de DoA Linear
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
    xmin, ymin, xmax, ymax = ROOM_BBOX
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return None
    return (x, y)


# ─────────────────────────────────────────────────────────────
# INSTÂNCIAS DE PROCESSADORES GLOBAIS
# ─────────────────────────────────────────────────────────────
hpf = HighPassFilter(cutoff=HPF_CUTOFF, fs=SAMPLE_RATE, order=HPF_ORDER)
noise_reducer = SpectralNoiseReducer(
    sr=SAMPLE_RATE, fft_size=ANALYSIS_SIZE, hop_size=HOP_SIZE,
    calib_time=NR_CALIB_TIME, smooth=NR_SMOOTH, over_sub=NR_OVER_SUB, noise_floor=NR_NOISE_FLOOR,
)
soft_gate = SoftNoiseGate(
    threshold_db=GATE_THRESHOLD_DB, attack_ms=GATE_ATTACK_MS, release_ms=GATE_RELEASE_MS,
    ratio=GATE_RATIO, sr=SAMPLE_RATE,
)
dsp = AudioDSP(sr=SAMPLE_RATE)
source_sep = NarrowbandSourceSeparator(
    sr=SAMPLE_RATE, fft_size=ANALYSIS_SIZE,
    n_bins_hist=72, min_db=SRC_MIN_DB, max_sources=SRC_MAX_COUNT, peak_min_sep=SRC_PEAK_MIN_SEP,
)
