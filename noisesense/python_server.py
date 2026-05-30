import sys
import os

# Adiciona a subpasta 'server' ao caminho de busca do Python para permitir importações planas
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

import asyncio
import collections
import json
import math
import socket
import time
import traceback
import queue
import numpy as np
import websockets

import config
import state
from parser import parse_binary
from dsp import hpf, noise_reducer, soft_gate, dsp, source_sep, dc_block_fast, soft_saturate
from workers import start_ptp_worker, start_playback_worker, start_udp_receiver

# Fila assíncrona UDP principal
udp_data_queue = None

# ─────────────────────────────────────────────────────────────
# PROCESSAMENTO DE ÁUDIO SÍNCRONO (RUN IN EXECUTOR)
# ─────────────────────────────────────────────────────────────
def _compute_on_audio(node_id, frame, ts_us, *channels):
    """Função síncrona de análise — executada fora do Event Loop para evitar travamento."""
    if node_id not in state.accum_dsp:
        state.accum_dsp[node_id] = {"ch": [[] for _ in channels], "ts": ts_us}

    for i, ch in enumerate(channels):
        state.accum_dsp[node_id]["ch"][i].append(ch)

    total = sum(len(b) for b in state.accum_dsp[node_id]["ch"][0])
    if total < config.ANALYSIS_SIZE:
        return None

    flats = [np.concatenate(state.accum_dsp[node_id]["ch"][i]) for i in range(len(channels))]
    ts_blk = state.accum_dsp[node_id]["ts"]

    state.accum_dsp[node_id] = {
        "ch": [[f[config.ANALYSIS_SIZE:]] for f in flats],
        "ts": ts_us,
    }

    analysis_chs = [f[:config.ANALYSIS_SIZE] for f in flats]

    # --- PROTEÇÃO DE HARDWARE: DETECÇÃO DE MIC 3 MORTO ---
    if len(analysis_chs) >= 3:
        rms_3 = float(np.sqrt(np.mean(analysis_chs[2]**2)))
        
        # Se o cabo I2S soltar, o sinal cai para ZERO digital absoluto.
        db_3 = 20.0 * math.log10(rms_3) if rms_3 > 1e-6 else -120.0
        
        # Desativa o mic 3 se ele estiver eletricamente morto
        if db_3 < -100.0:
            analysis_chs = analysis_chs[:2] 

    result = dsp.process(node_id, *analysis_chs)
    result['timestamp'] = ts_blk
    result['frame'] = frame
    state.frames_total += 1

    # Separação de fontes (Narrowband DoA)
    db_avg_quick = (result.get('db_L', -90) + result.get('db_R', -90)) / 2.0
    if db_avg_quick > -45.0:
        result['sources'] = source_sep.process(analysis_chs)
    else:
        result['sources'] = []

    # 🔗 SINCRONIZAÇÃO ABSOLUTA: SETA = ESFERA
    if result['sources']:
        best_src = max(result['sources'], key=lambda s: s['confidence'])
        target_ang = best_src['angle']
        
        # Suaviza o movimento da seta DoA
        curr_ang = dsp.ema[node_id]['angle']
        diff = (target_ang - curr_ang + 180.0) % 360.0 - 180.0
        new_ang = (curr_ang + 0.4 * diff) % 360.0
        
        dsp.ema[node_id]['angle'] = new_ang
        dsp.ema[node_id]['conf']  = best_src['confidence']
        
        result['angle'] = round(new_ang, 1)
        result['confidence'] = best_src['confidence']

    return result


# ─────────────────────────────────────────────────────────────
# ATUALIZAÇÃO ASSÍNCRONA DE ESTADO COM TRATAMENTO DE PICOS
# ─────────────────────────────────────────────────────────────
async def on_audio(node_id, frame, ts_us, *channels):
    """Wrapper assíncrono com lock para atualização do dicionário de estado."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _compute_on_audio, node_id, frame, ts_us, *channels
    )
    if result is None:
        return None

    async with state._state_lock:
        if node_id not in state.state["nodes"]:
            state.state["nodes"][node_id] = {
                "raw": None,
                "history_db": collections.deque(maxlen=100),
                "peak": -90.0, "avg": -90.0,
            }
        node = state.state["nodes"][node_id]
        node["raw"] = result
        
        # Converter tipagens NumPy para nativos do Python para evitar falha do JSON
        db_avg = float((result["db_L"] + result["db_R"]) / 2.0)
        node["history_db"].append(db_avg)
        if db_avg > node["peak"]:
            node["peak"] = db_avg
        node["avg"] = round(
            sum(node["history_db"]) / len(node["history_db"]), 1
        )
        
        # 1. RASTREADOR DE AMBIENTE
        if "ambient_db" not in node:
            node["ambient_db"] = db_avg
        node["ambient_db"] = 0.95 * node["ambient_db"] + 0.05 * db_avg
        
        amb_dist = 1.5
        result['ambient_dist'] = amb_dist
        result['ambient_db'] = round(node["ambient_db"], 1)

        # 2. DETECTOR DE TRANSIENTES (Sons de impacto > 6dB acima do fundo)
        is_peak = (db_avg > node["ambient_db"] + 6.0)
        now_ts = time.time()
        if is_peak:
            for src in result.get('sources', []):
                src_ang = src['angle']
                src_db = src['db']
                src_conf = src['confidence']
                if src_db <= -45.0 or src_conf < 0.1:
                    continue

                dist_spl = max(0.5, min(40.0, 10.0 ** ((-40.0 - src_db) / 20.0)))
                coh = max(0.01, min(1.0, src_conf))
                dist_coh = 0.5 * math.exp(4.5 * (1.0 - coh))
                dist_est = max(0.5, min(40.0, 0.4 * dist_spl + 0.6 * dist_coh))

                state.state["events"].append({
                    "node": node_id,
                    "dist": round(dist_est, 2),
                    "angle": round(src_ang, 1),
                    "intensity": round(src_db, 1),
                    "confidence": round(src_conf, 2),
                    "dom_freq": src.get('dom_freq', 0),
                    "ts": now_ts,
                })

        # Limpar eventos com mais de 1.5s
        state.state["events"] = [e for e in state.state["events"] if now_ts - e["ts"] < 1.5]
        if len(state.state["events"]) > 50:
            state.state["events"] = state.state["events"][-50:]

    return state.state


# ─────────────────────────────────────────────────────────────
# SERVIDOR WEBSOCKET
# ─────────────────────────────────────────────────────────────
async def handle_client(ws):
    state.connected.add(ws)
    print(f"[WS] +cliente (total={len(state.connected)})")
    try:
        await ws.wait_closed()
    finally:
        state.connected.discard(ws)
        print(f"[WS] -cliente (total={len(state.connected)})")


async def broadcast(msg: str):
    if not state.connected:
        return
    clients = list(state.connected)
    results = await asyncio.gather(
        *[c.send(msg) for c in clients],
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"[WS] Envio falhou: {r}")


# ─────────────────────────────────────────────────────────────
# PROCESSAMENTO DOS PACOTES RECEBIDOS
# ─────────────────────────────────────────────────────────────
async def process_packet(data: bytes):
    """Encaminha o pacote pelo fluxo: parsing -> HPF -> NR -> NoiseGate -> DSP -> WS."""
    node_id, frame, ts_us, audio, sr_rx = parse_binary(data)
    if node_id is None:
        return
    mic1_raw, mic2_raw, mic3_raw = audio

    # Detecção de falhas ou perdas de frames
    if node_id in state.last_frame_per_node:
        gap = (frame - state.last_frame_per_node[node_id]) & 0xFFFFFFFF
        if gap > 1:
            print(f"[GAP] Nó {node_id}: {gap-1} pacote(s) perdido(s) (frame #{frame})")
            ramp      = np.linspace(0.0, 1.0, len(mic1_raw))
            mic1_raw  = mic1_raw  * ramp
            mic2_raw  = mic2_raw  * ramp
            mic3_raw  = mic3_raw  * ramp
    state.last_frame_per_node[node_id] = frame

    # 1. HPF 80 Hz
    hp = hpf.process(node_id, mic1_raw, mic2_raw, mic3_raw)

    # 2. Acumulador OLA de redução de ruído espectral
    if node_id not in state.accum_nr:
        state.accum_nr[node_id] = [[] for _ in range(3)]
    for i in range(3):
        state.accum_nr[node_id][i].append(hp[i])

    total_nr = sum(len(b) for b in state.accum_nr[node_id][0])
    nr_out = [[] for _ in range(3)]

    while total_nr >= config.HOP_SIZE:
        flats = [np.concatenate(state.accum_nr[node_id][i]) for i in range(3)]
        hops = [f[:config.HOP_SIZE] for f in flats]
        state.accum_nr[node_id] = [[f[config.HOP_SIZE:]] for f in flats]
        total_nr -= config.HOP_SIZE

        cleans = noise_reducer.process(node_id, *hops)
        gated = soft_gate.process(node_id, *cleans)
        for i in range(3):
            nr_out[i].append(gated[i])

    if not nr_out[0]:
        return

    outs = [np.concatenate(nr_out[i]) for i in range(3)]

    # 3. Playback de áudio em tempo real com sounddevice
    if config.PLAY_AUDIO and node_id == config.LISTEN_NODE:
        pb_L = soft_saturate(dc_block_fast(outs[0], f"{node_id}_pb_L"), drive_db=6.0, makeup_db=-1.5, knee=0.7)
        pb_R = soft_saturate(dc_block_fast(outs[1], f"{node_id}_pb_R"), drive_db=6.0, makeup_db=-1.5, knee=0.7)
        chunk = np.column_stack((
            pb_L.astype(np.float32),
            pb_R.astype(np.float32),
        ))
        try:
            state.audio_q.put_nowait(chunk)
        except queue.Full:
            pass

    # 4. Análise acústica principal
    packet = await on_audio(node_id, frame, ts_us, outs[0], outs[1], outs[2])

    # 5. Transmissão com Throttle via WebSocket
    if packet and state.connected:
        state.ws_frame_cnt[node_id] = state.ws_frame_cnt.get(node_id, 0) + 1
        if state.ws_frame_cnt[node_id] >= config.WS_THROTTLE_FRAMES:
            state.ws_frame_cnt[node_id] = 0
            msg = json.dumps(
                packet, 
                separators=(',', ':'),
                default=lambda o: list(o) if isinstance(o, collections.deque) else str(o)
            )
            await broadcast(msg)
            
            # Limpar eventos transmitidos (evita re-triangulação infinita)
            async with state._state_lock:
                state.state["events"].clear()


# ─────────────────────────────────────────────────────────────
# LAÇO PRINCIPAL DE EVENTOS
# ─────────────────────────────────────────────────────────────
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

        # Log periódico de status do console (3 segundos)
        now = time.time()
        if now - last_log >= 3.0:
            last_log = now
            parts = []
            for nid, st in noise_reducer._state.items():
                cal  = "[OK]" if st["calibrated"] else f"[{st['calib_count']}]"
                raw_n = state.state["nodes"].get(nid, {}).get("raw")
                if raw_n:
                    parts.append(f"{nid}:{cal} dB={raw_n['db_L']:.0f}/{raw_n['db_R']:.0f} ang={raw_n['angle']:.0f}°")
                else:
                    parts.append(f"{nid}:{cal}")
            print(f"[INFO] frames={state.frames_total} | {' | '.join(parts)} | ev={len(state.state['events'])} ws={len(state.connected)}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT PRINCIPAL DO SERVIDOR
# ─────────────────────────────────────────────────────────────
async def main():
    global udp_data_queue

    state._state_lock = asyncio.Lock()
    udp_data_queue    = asyncio.Queue(maxsize=500)
    loop              = asyncio.get_event_loop()

    # Soquete UDP síncrono para a thread dedicada do receptor
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((config.UDP_IP, config.UDP_PORT))

    # Iniciar os serviços paralelos em threads
    start_ptp_worker()
    start_playback_worker()
    start_udp_receiver(loop, udp_data_queue, udp_sock)

    print("=" * 60)
    print("  Sistema Acústico v9 — Servidor DSP (Modularizado)")
    print(f"  UDP:{config.UDP_PORT} | WS:{config.WS_PORT} | PTP:{config.PTP_PORT}")
    print(f"  HEADER: {config.HEADER_FMT} = {config.HEADER_SIZE} bytes")
    print(f"  Análise: {config.ANALYSIS_SIZE} smp = "
          f"{config.ANALYSIS_SIZE/config.SAMPLE_RATE*1000:.1f} ms")
    print(f"  Hop NR : {config.HOP_SIZE} smp = "
          f"{config.HOP_SIZE/config.SAMPLE_RATE*1000:.1f} ms")
    print("=" * 60)

    async with websockets.serve(handle_client, "0.0.0.0", config.WS_PORT):
        await main_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[EXIT] Encerrado.")