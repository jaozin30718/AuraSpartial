"""
dataset_loaders.py
==================
Carregadores de áudio com cache em memória e pré-carregamento assíncrono.
Suporta LibriSpeech, UrbanSound8K e arquivos .wav genéricos.
"""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import math

logger = logging.getLogger(__name__)


class AudioFileIndex:
    """
    Índice leve de arquivos de áudio — escaneia diretórios recursivamente
    e mantém lista de (path, duration_seconds) em memória.
    Evita recarregar metadados a cada chamada.
    """

    def __init__(self, root_dir: str, extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3")):
        self.root_dir   = Path(root_dir)
        self.extensions = extensions
        self._index: List[Path] = []
        self._build_index()

    def _build_index(self) -> None:
        logger.info(f"Indexando arquivos em: {self.root_dir}")
        for ext in self.extensions:
            self._index.extend(self.root_dir.rglob(f"*{ext}"))
        self._index = [p for p in self._index if p.stat().st_size > 1024]  # > 1KB
        logger.info(f"Total de arquivos indexados: {len(self._index)}")

    def sample_path(self, rng: np.random.Generator) -> Path:
        if not self._index:
            raise FileNotFoundError(f"Nenhum arquivo de áudio encontrado em {self.root_dir}")
        idx = rng.integers(0, len(self._index))
        return self._index[idx]

    def __len__(self) -> int:
        return len(self._index)


class AudioLoader:
    """
    Carregador de áudio com:
    - Conversão automática de sample rate
    - Mixdown para mono
    - Cache LRU em memória
    - Pool de threads para I/O assíncrono
    """

    MAX_CACHE_SIZE = 64      # arquivos em memória (Reduz RAM alocada drasticamente)

    def __init__(
        self,
        target_sr:   int = 16_000,
        n_io_threads: int = 4,
    ):
        self.target_sr    = target_sr
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: deque            = deque()  # O(1) popleft
        self._executor = ThreadPoolExecutor(max_workers=n_io_threads, thread_name_prefix="audio_io")
        self._lock = threading.Lock()

    def load_mono(
        self,
        path:       Path,
        duration_s: float,
        rng:        np.random.Generator,
        normalize:  bool = True,
    ) -> np.ndarray:
        """
        Carrega chunk mono de duração exata `duration_s`.
        Se o arquivo for mais curto, faz tile (repeat).

        Returns
        -------
        np.ndarray shape (n_samples,) dtype float32
        """
        key = str(path)

        # Tenta cache
        with self._lock:
            if key in self._cache:
                audio_full = self._cache[key]
            else:
                audio_full = self._read_and_resample(path)
                self._add_to_cache(key, audio_full)

        n_target = int(duration_s * self.target_sr)

        # Tile se necessário
        if len(audio_full) < n_target:
            repeats    = int(np.ceil(n_target / len(audio_full)))
            audio_full = np.tile(audio_full, repeats)

        # Crop aleatório
        max_start = len(audio_full) - n_target
        start     = rng.integers(0, max(1, max_start))
        chunk     = audio_full[start : start + n_target].copy()

        if normalize:
            from utils.dsp_utils import normalize_signal
            # Sorteia um volume RMS entre 0.02 e 0.15 para criar diversidade (Sinal vs Interferência)
            random_target_rms = rng.uniform(0.02, 0.15)
            chunk = normalize_signal(chunk, target_rms=random_target_rms)

        return chunk.astype(np.float32)

    def _read_and_resample(self, path: Path) -> np.ndarray:
        """Lê arquivo e reamostrado para self.target_sr."""
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:
            logger.warning(f"Erro ao ler {path}: {e}. Retornando silêncio.")
            return np.zeros(self.target_sr, dtype=np.float32)

        # Mixdown para mono
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Reamostrar se necessário
        if sr != self.target_sr:
            gcd  = math.gcd(self.target_sr, sr)
            data = resample_poly(data, self.target_sr // gcd, sr // gcd).astype(np.float32)

        return data

    def _add_to_cache(self, key: str, audio: np.ndarray) -> None:
        """Adiciona ao cache com política LRU O(1) via deque."""
        if len(self._cache_order) >= self.MAX_CACHE_SIZE:
            oldest = self._cache_order.popleft()  # O(1) — muito mais rápido que list.pop(0)
            del self._cache[oldest]

        self._cache[key]    = audio
        self._cache_order.append(key)

    def prefetch_async(self, paths: List[Path]) -> None:
        """Pré-carrega arquivos em background threads."""
        for path in paths:
            self._executor.submit(self._read_and_resample, path)

    def __del__(self):
        self._executor.shutdown(wait=False)


class PrefetchQueue:
    """
    Fila de pré-carregamento que mantém um buffer de áudios prontos,
    reduzindo latência de I/O no loop principal de geração.
    """

    def __init__(
        self,
        file_index:  AudioFileIndex,
        audio_loader: AudioLoader,
        duration_s:  float,
        buffer_size: int = 64,
        n_threads:   int = 4,
        seed:        int = 42,
    ):
        self._index  = file_index
        self._loader = audio_loader
        self._dur    = duration_s
        self._queue  = queue.Queue(maxsize=buffer_size)
        self._stop   = threading.Event()
        self._rng    = np.random.default_rng(seed)
        self._threads = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(n_threads)
        ]
        for t in self._threads:
            t.start()

    def _worker(self) -> None:
        local_rng = np.random.default_rng()
        while not self._stop.is_set():
            try:
                path  = self._index.sample_path(local_rng)
                audio = self._loader.load_mono(path, self._dur, local_rng)
                self._queue.put(audio, timeout=1.0)
            except queue.Full:
                pass
            except Exception as e:
                logger.debug(f"PrefetchQueue worker error: {e}")

    def get(self, timeout: float = 5.0) -> np.ndarray:
        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        self._stop.set()