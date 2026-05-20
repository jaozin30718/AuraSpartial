"""
batch_generator.py
==================
MÓDULO 5: Geração Paralela em Larga Escala

Orquestra todos os módulos anteriores em um pipeline paralelo de alto desempenho
usando multiprocessing com shared-nothing workers.

Arquitetura de paralelismo:
  - N workers independentes (um por CPU core)
  - Cada worker tem seu próprio RNG seed, AudioLoader e cache
  - Resultado enviado via Queue para escritor dedicado (I/O bound separado)
  - Barra de progresso em tempo real via tqdm

Performance esperada:
  ~50-200 amostras/min por core (depende do RT60 e tamanho da sala)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from config import PipelineConfig
from dataset_loaders import AudioFileIndex, AudioLoader
from ground_truth_extractor import GroundTruthExtractor
from noise_injector import NoiseInjector
from room_simulator import RoomSimulator
from utils.io_utils import HDF5Writer, WebDatasetWriter

logger = logging.getLogger(__name__)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(processName)-16s | %(levelname)7s | %(message)s",
    datefmt = "%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Worker Function (executado em processo separado)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_generate_samples(
    worker_id:     int,
    n_samples:     int,
    config:        PipelineConfig,
    result_queue:  mp.Queue,
    error_queue:   mp.Queue,
    done_queue:    mp.Queue,
    seed_offset:   int = 0,
) -> None:
    """
    Função de worker multiprocessing.

    Cada worker é completamente independente:
      - Seed único: base_seed + worker_id * 10000
      - Cache de áudio local (não compartilhado)
      - Sem estado global mutável

    Resultados enviados como dicionário de numpy arrays para evitar
    overhead de serialização de objetos Python complexos.
    """
    # Seed único e reprodutível por worker
    base_seed = (config.global_seed or int(time.time())) + seed_offset
    worker_seed = base_seed + worker_id * 10_000
    rng = np.random.default_rng(worker_seed)

    process_name = mp.current_process().name
    logger.info(f"[{process_name}] Iniciando — {n_samples} amostras, seed={worker_seed}")

    # ── Inicialização dos módulos (local por worker) ──────────────────────────
    try:
        src_index   = AudioFileIndex(config.audio_source_dir)
        noise_index = AudioFileIndex(config.noise_source_dir) if Path(config.noise_source_dir).exists() else None

        audio_loader = AudioLoader(
            target_sr    = config.room.sample_rate,
            n_io_threads = 2,
        )

        room_sim = RoomSimulator(
            config       = config.room,
            audio_loader = audio_loader,
            source_index = src_index,
            noise_index  = noise_index,
            moving_source_probability = config.augmentation.moving_source_probability,
        )

        noise_injector = NoiseInjector(config.augmentation)
        gt_extractor   = GroundTruthExtractor(frame_duration_ms=20.0)

    except Exception as e:
        error_queue.put({
            "worker_id": worker_id,
            "error":     str(e),
            "traceback": traceback.format_exc(),
        })
        return

    # ── Loop de geração ───────────────────────────────────────────────────────
    success_count = 0
    fail_count    = 0

    for sample_idx in range(n_samples):
        try:
            # 1. Simulação de sala
            sim_result = room_sim.simulate(rng=rng)

            # 2. Carrega babble noise (opcional)
            noise_audio = None
            if noise_index and len(noise_index) > 0:
                try:
                    noise_path  = noise_index.sample_path(rng)
                    noise_audio = audio_loader.load_mono(
                        noise_path,
                        config.room.audio_duration_seconds,
                        rng,
                        normalize=False,
                    )
                except Exception:
                    pass

            # 3. Augmentation
            aug_result = noise_injector.apply(sim_result, rng, noise_audio)

            # 4. Ground Truth
            gt_package = gt_extractor.extract(aug_result)

            # 5. Empacota para envio (converte para dicts serializáveis)
            payload = _pack_payload(aug_result, gt_package)
            result_queue.put(payload)

            success_count += 1

            if success_count % 10 == 0:
                logger.debug(
                    f"[{process_name}] {success_count}/{n_samples} amostras geradas."
                )

        except Exception as e:
            fail_count += 1
            logger.warning(
                f"[{process_name}] Falha na amostra {sample_idx}: {e}. "
                f"Continuando..."
            )
            if fail_count > n_samples * 0.10:
                logger.error(f"[{process_name}] Taxa de falha > 10%. Abortando worker.")
                break

    logger.info(
        f"[{process_name}] Finalizado — {success_count} OK, {fail_count} falhas."
    )
    # Sinaliza fim deste worker
    done_queue.put(worker_id)


def _pack_payload(aug_result, gt_package) -> dict:
    """Empacota resultado em dicionário de numpy arrays para IPC via Queue."""
    return {
        # Áudio
        "stereo_mix":          aug_result.stereo_mix_augmented,
        "source_images_mic0":  gt_package.source_images_mic0,
        "source_signals_dry":  gt_package.source_signals_dry,
        # Labels
        "doa_az_el":           gt_package.doa_azimuth_elevation,
        "source_xyz":          gt_package.source_positions_xyz,
        "tdoa_samples":        gt_package.tdoa_samples_per_frame,
        "vad_frames":          gt_package.vad_masks_per_frame,
        "vad_samples":         gt_package.vad_masks_sample_level,
        "spatial_heatmap":     gt_package.spatial_heatmap,
        "si_sdr":              gt_package.si_sdr_per_source,
        # Objetos completos para o Writer
        "__aug_result__":      aug_result,
        "__gt_package__":      gt_package,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BatchGenerator — Orquestrador Principal
# ─────────────────────────────────────────────────────────────────────────────

class BatchGenerator:
    """
    Orquestrador do pipeline paralelo de geração de dados.

    Responsabilidades:
      1. Distribuir trabalho entre N workers (multiprocessing)
      2. Coletar resultados via Queue
      3. Serializar em HDF5 ou WebDataset
      4. Monitorar progresso e taxa de geração
      5. Tolerância a falhas individuais de worker

    Usage:
    ------
    >>> config = PipelineConfig(total_samples=10_000, n_workers=8)
    >>> generator = BatchGenerator(config)
    >>> generator.generate()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Valida configuração antes de iniciar."""
        src_path = Path(self.config.audio_source_dir)
        if not src_path.exists():
            raise FileNotFoundError(
                f"Diretório de fontes de áudio não encontrado: {src_path}\n"
                f"Baixe LibriSpeech ou UrbanSound8K e configure 'audio_source_dir'."
            )

        if self.config.n_workers < 1:
            raise ValueError("n_workers deve ser >= 1")

        if self.config.total_samples < 1:
            raise ValueError("total_samples deve ser >= 1")

        logger.info(
            f"Configuração validada:\n"
            f"  Workers:        {self.config.n_workers}\n"
            f"  Total amostras: {self.config.total_samples:,}\n"
            f"  Output:         {self.config.output_dir}\n"
            f"  Formato:        {'WebDataset' if self.config.use_webdataset else 'HDF5'}"
        )

    def generate(self) -> None:
        """
        Inicia a geração paralela completa.
        Bloqueia até que todas as amostras sejam geradas e salvas.
        """
        # Inicia escritor de I/O PRIMEIRO para descobrir o estado (resume)
        writer = self._create_writer()

        # Pega a contagem global a partir do writer
        existing = getattr(writer, "_global_count", 0)
        total    = self.config.total_samples
        remaining = max(0, total - existing)

        if remaining == 0:
            logger.info(f"Dataset já possui {total} amostras. Geração concluída!")
            writer.close()
            return

        n_workers = self.config.n_workers

        # Distribui amostras faltantes entre workers
        samples_per_worker = [remaining // n_workers] * n_workers
        for i in range(remaining % n_workers):
            samples_per_worker[i] += 1

        logger.info(
            f"Retomando geração. Já existem {existing}/{total} amostras.\n"
            f"Iniciando {n_workers} workers para gerar as {remaining} restantes.\n"
            f"Distribuição: {samples_per_worker}"
        )

        # Filas de comunicação
        ctx          = mp.get_context("spawn")    # spawn = mais seguro que fork com threads
        result_queue = ctx.Queue(maxsize=500)
        error_queue  = ctx.Queue(maxsize=100)
        done_queue   = ctx.Queue()

        # Inicia workers
        processes = []
        for worker_id, n_smp in enumerate(samples_per_worker):
            if n_smp == 0:
                continue
            p = ctx.Process(
                target = _worker_generate_samples,
                args   = (
                    worker_id,
                    n_smp,
                    self.config,
                    result_queue,
                    error_queue,
                    done_queue,
                    worker_id * 100_000 + existing,   # offset de seed contínuo
                ),
                name   = f"AudioWorker-{worker_id:02d}",
                daemon = True,
            )
            p.start()
            processes.append(p)

        # Loop de coleta de resultados
        self._collection_loop(
            result_queue = result_queue,
            error_queue  = error_queue,
            done_queue   = done_queue,
            writer       = writer,
            n_workers    = len(processes),
            total        = total,
            initial      = existing,
        )

        # Aguarda todos os workers
        for p in processes:
            p.join(timeout=120)
            if p.is_alive():
                logger.warning(f"Worker {p.name} não terminou a tempo. Forçando término.")
                p.terminate()

        writer.close()
        logger.info(f"✅ Geração completa! {total:,} amostras salvas em {self.config.output_dir}")

    def _collection_loop(
        self,
        result_queue: mp.Queue,
        error_queue:  mp.Queue,
        done_queue:   mp.Queue,
        writer,
        n_workers:    int,
        total:        int,
        initial:      int = 0,
    ) -> None:
        """Loop principal de coleta e escritura de resultados."""
        workers_done = 0
        collected    = initial

        pbar = tqdm(
            total   = total,
            initial = initial,
            desc    = "Gerando amostras",
            unit    = "sample",
            dynamic_ncols = True,
        )

        start_time = time.time()

        while workers_done < n_workers:
            # Processa erros de workers
            while not error_queue.empty():
                err = error_queue.get_nowait()
                logger.error(
                    f"Erro no Worker {err.get('worker_id', '?')}: "
                    f"{err.get('error', 'desconhecido')}"
                )

            # Verifica sinal de fim de worker
            while not done_queue.empty():
                wid = done_queue.get_nowait()
                workers_done += 1
                logger.info(f"Worker {wid} finalizado. ({workers_done}/{n_workers})")

            if workers_done >= n_workers and result_queue.empty():
                break

            # Coleta resultado
            try:
                payload = result_queue.get(timeout=2.0)
            except Exception:
                continue

            # Escreve resultado
            try:
                writer.write(
                    payload["__aug_result__"],
                    payload["__gt_package__"],
                )
                collected += 1
                pbar.update(1)

                # Atualiza taxa de geração na barra de progresso
                elapsed = time.time() - start_time
                generated_now = collected - initial
                rate    = generated_now / elapsed if elapsed > 0 else 0
                eta     = (total - collected) / rate if rate > 0 else float("inf")
                pbar.set_postfix({
                    "rate": f"{rate:.1f}/s",
                    "ETA":  f"{eta/60:.1f}min" if eta < float("inf") else "∞"
                })

            except Exception as e:
                logger.error(f"Erro ao escrever amostra: {e}")

        pbar.close()

    def _create_writer(self):
        """Cria escritor adequado baseado na configuração."""
        if self.config.use_webdataset:
            return WebDatasetWriter(
                output_dir    = self.config.output_dir,
                shard_size_mb = self.config.tar_shard_size_mb,
            )
        else:
            return HDF5Writer(
                output_dir       = self.config.output_dir,
                samples_per_file = self.config.hdf5_chunk_size,
                compression      = self.config.compression,
                compression_opts = self.config.compression_level if self.config.compression == "gzip" else None,
            )