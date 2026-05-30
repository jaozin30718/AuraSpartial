import socket
import struct
import time
import threading
import asyncio
import numpy as np
import state
from config import UDP_IP, UDP_PORT, PTP_PORT, SAMPLE_RATE, PLAY_AUDIO, LISTEN_NODE
from dsp import dc_block_fast, soft_saturate

# ─────────────────────────────────────────────────────────────
# PTP — Thread dedicada para responder pings de sincronização
# ─────────────────────────────────────────────────────────────
def _ptp_worker():
    """
    Recebe pings PTP dos ESP32 e responde com 8 bytes de offset 
    temporal Epoch <-> Uptime para a sincronização de fase de amostragem.
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
            node_id      = chr(data[8])
            
            # Epoch atual do servidor em microsegundos
            t2_us        = int(time.time() * 1e6)   
            offset       = t2_us - t1_us
            
            response     = struct.pack('<q', offset)
            ptp_sock.sendto(response, (addr[0], 5006))
            
            print(f"[PTP] Nó '{node_id}' sincronizado | offset={offset} µs | ip={addr[0]}")
        except socket.timeout:
            pass
        except Exception as exc:
            print(f"[PTP] Erro: {exc}")


def start_ptp_worker():
    t = threading.Thread(target=_ptp_worker, daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────
# PLAYBACK AUDIO LOCAL (SOUNDDEVICE)
# ─────────────────────────────────────────────────────────────
def _audio_callback(outdata, frames, time_info, status):
    if status:
        print(f"[PLAYBACK] {status}")
    try:
        while not state.audio_q.empty():
            state._pb_buffer = np.concatenate(
                (state._pb_buffer, state.audio_q.get_nowait())
            )
        if len(state._pb_buffer) >= frames:
            outdata[:] = state._pb_buffer[:frames]
            state._pb_buffer  = state._pb_buffer[frames:]
        else:
            n = len(state._pb_buffer)
            if n > 0:
                outdata[:n] = state._pb_buffer
                outdata[n:] = 0.0
                state._pb_buffer  = np.zeros((0, 2), dtype=np.float32)
            else:
                outdata[:] = 0.0
    except Exception as exc:
        outdata[:] = 0.0
        print(f"[PLAYBACK CB] {exc}")


def _playback_worker():
    import sounddevice as sd
    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=2,
            dtype='float32', blocksize=1024,
            latency='high', callback=_audio_callback,
        ):
            print("[PLAYBACK] Stream ativo.")
            while state.audio_q.qsize() < 40:
                time.sleep(0.01)
            while True:
                time.sleep(1)
    except Exception as exc:
        print(f"[PLAYBACK] Erro: {exc}")


def start_playback_worker():
    if PLAY_AUDIO:
        t = threading.Thread(target=_playback_worker, daemon=True)
        t.start()
        return t
    return None


# ─────────────────────────────────────────────────────────────
# THREAD RECEPTORA UDP RAPIDA (NÃO BLOQUEANTE DO EVENT LOOP)
# ─────────────────────────────────────────────────────────────
def _udp_receiver_thread(loop_ref, q_ref, sock_ref):
    print(f"[UDP-THREAD] Recebendo em {UDP_IP}:{UDP_PORT}")
    def _safe_put(d):
        try:
            q_ref.put_nowait(d)
        except asyncio.QueueFull:
            pass
    while True:
        try:
            data, _ = sock_ref.recvfrom(4096)
            loop_ref.call_soon_threadsafe(_safe_put, data)
        except Exception as exc:
            print(f"[UDP-THREAD] {exc}")
            time.sleep(0.001)


def start_udp_receiver(loop, q, sock):
    t = threading.Thread(
        target=_udp_receiver_thread,
        args=(loop, q, sock),
        daemon=True,
    )
    t.start()
    return t
