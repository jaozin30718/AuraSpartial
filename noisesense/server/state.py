import collections
import queue
import numpy as np

# ─────────────────────────────────────────────────────────────
# ESTADO GLOBAL E BUFFERS COMPARTILHADOS DO SERVIDOR
# ─────────────────────────────────────────────────────────────

# Acumuladores de processamento por nó
accum_nr = {}
accum_dsp = {}
last_frame_per_node = {}

# Estado do sistema transmitido via WebSocket
state = {
    "nodes": {
        "A": {"raw": None,
              "history_db": collections.deque(maxlen=100),
              "peak": -90.0, "avg": -90.0},
    },
    "events": [],
}

# Lock assíncrono para acesso concorrente ao dicionário state
_state_lock = None

# Estatísticas e controle de conexão
frames_total = 0
ws_frame_cnt = {}
connected = set()

# Filas e buffers de reprodução de áudio local
audio_q = queue.Queue(maxsize=300)
_pb_buffer = np.zeros((0, 2), dtype=np.float32)
