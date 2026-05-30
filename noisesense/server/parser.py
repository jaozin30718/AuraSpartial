import struct
import numpy as np
from config import HEADER_FMT, HEADER_SIZE, MAX_PACKET_SAMPLES

# ─────────────────────────────────────────────────────────────
# DECODIFICADOR DE DADOS BINÁRIOS UDP DO ESP32
# ─────────────────────────────────────────────────────────────
def parse_binary(data: bytes):
    """
    Desempacota header binário ESP32 3 Mics (25 bytes) e
    extrai amostras int16 → float64 normalizado.

    Retorna:
        (node_id, frame, ts_us, (mic1, mic2, mic3), sr_rx)
        ou (None, ...) se for um pacote inválido.
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
