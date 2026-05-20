"""
utils/io_utils.py
=================
Serialização e deserialização de amostras em HDF5 e WebDataset (tar).

Schema HDF5 por sample group:
  /sample_{idx}/
    audio/
      stereo_mix        (2, S)  float32
      source_images     (N, S)  float32
      source_signals_dry (N, S) float32
    labels/
      doa_az_el         (N, T, 2) float32
      source_xyz        (N, T, 3) float32
      tdoa_samples      (N, T)   float32
      vad_frames        (N, T)   bool
      vad_samples       (N, S)   bool
      spatial_heatmap   (T, Az, El) float32
      si_sdr            (N,)     float32
    metadata/
      n_sources         scalar int
      sample_rate       scalar int
      rt60_target       scalar float32
      room_dims         (3,)    float32
      aug_log           JSON string
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
import threading
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from ground_truth_extractor import GroundTruthPackage
from noise_injector import AugmentedResult

logger = logging.getLogger(__name__)


def float32_to_int16(audio_float32: np.ndarray) -> np.ndarray:
    """Converte áudio float [-1.0, 1.0] para PCM 16-bits inteiro [-32768, 32767]."""
    clipped = np.clip(audio_float32, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


class HDF5Writer:
    """
    Escritor thread-safe de amostras em arquivos HDF5.

    Cada arquivo armazena `samples_per_file` amostras.
    Usa compressão LZF por padrão (6x mais rápido que gzip, ~80% do ratio).

    Usage:
    ------
    >>> writer = HDF5Writer("./output", samples_per_file=256, compression="lzf")
    >>> writer.write(aug_result, gt_package)
    >>> writer.close()
    """

    def __init__(
        self,
        output_dir:       str,
        samples_per_file: int  = 256,
        compression:      str  = "lzf",
        compression_opts: Optional[int] = None,
        resume:           bool = True,
    ):
        self.output_dir       = Path(output_dir)
        self.samples_per_file = samples_per_file
        self.compression      = compression
        self.compression_opts = compression_opts

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock         = threading.Lock()
        self._current_file: Optional[h5py.File] = None
        self._current_path: Optional[Path]       = None
        self._sample_count = 0
        self._file_index   = 0
        self._global_count = 0

        if resume:
            self._resume_state()
        else:
            self._open_new_file()

    def _resume_state(self) -> None:
        """Recupera o estado a partir dos shards existentes no diretório."""
        shards = sorted(self.output_dir.glob("shard_*.h5"))
        if not shards:
            self._open_new_file()
            return
            
        last_shard = shards[-1]
        self._file_index = int(last_shard.stem.split("_")[1])
        
        total_samples = 0
        last_shard_samples = 0
        
        # Conta amostras existentes
        for shard in shards:
            try:
                with h5py.File(shard, "r") as f:
                    count = len([k for k in f.keys() if k.startswith("sample_")])
                    total_samples += count
                    if shard == last_shard:
                        last_shard_samples = count
            except Exception as e:
                logger.warning(f"Aviso ao ler {shard} no resume: {e}")
                
        self._global_count = total_samples
        
        if last_shard_samples >= self.samples_per_file:
            self._file_index += 1
            self._open_new_file()
            logger.info(f"HDF5: Retomando na amostra {self._global_count} (criando novo shard).")
        else:
            self._current_path = last_shard
            self._current_file = h5py.File(str(last_shard), "a")
            self._sample_count = last_shard_samples
            self._file_index += 1
            logger.info(f"HDF5: Retomando (append) no arquivo {last_shard.name} a partir da amostra interna {self._sample_count}. Global: {self._global_count}.")

    def _open_new_file(self) -> None:
        """Abre novo arquivo HDF5."""
        fname  = f"shard_{self._file_index:05d}.h5"
        fpath  = self.output_dir / fname

        self._current_file  = h5py.File(str(fpath), "w", swmr=False)
        self._current_path  = fpath
        self._sample_count  = 0
        self._file_index   += 1
        logger.debug(f"HDF5: novo arquivo aberto → {fpath}")

    def write(
        self,
        aug_result: AugmentedResult,
        gt:         GroundTruthPackage,
    ) -> None:
        """
        Escreve uma amostra completa no arquivo HDF5 corrente.
        Thread-safe via lock interno.
        """
        with self._lock:
            if self._sample_count >= self.samples_per_file:
                self._current_file.close()
                self._open_new_file()

            grp_name = f"sample_{self._global_count:06d}"
            
            # Se já existir devido a uma execução interrompida brutalmente, apaga e recria
            if grp_name in self._current_file:
                del self._current_file[grp_name]
                
            grp = self._current_file.create_group(grp_name)

            self._write_audio(grp, aug_result, gt)
            self._write_labels(grp, gt)
            self._write_metadata(grp, gt, aug_result)

            self._sample_count  += 1
            self._global_count  += 1

            # 🟢 CORREÇÃO: Faz flush apenas a cada 50 amostras. 
            # Isso reduz as chamadas de I/O do sistema operacional em 98%.
            if self._sample_count % 50 == 0:
                self._current_file.flush()

            if self._global_count % 100 == 0:
                logger.info(f"HDF5: {self._global_count} amostras escritas.")

    def _write_audio(
        self,
        grp:        h5py.Group,
        aug_result: AugmentedResult,
        gt:         GroundTruthPackage,
    ) -> None:
        audio_grp = grp.create_group("audio")

        kw = dict(compression=self.compression, compression_opts=self.compression_opts)

        # 🟢 CORREÇÃO: Aplica a conversão para int16. O tamanho do áudio no HDF5 cai em 50%.
        audio_grp.create_dataset(
            "stereo_mix",
            data   = float32_to_int16(aug_result.stereo_mix_augmented),
            dtype  = "int16",
            **kw,
        )
        audio_grp.create_dataset(
            "source_images_mic0",
            data   = float32_to_int16(gt.source_images_mic0),
            dtype  = "int16",
            **kw,
        )
        audio_grp.create_dataset(
            "source_signals_dry",
            data   = float32_to_int16(gt.source_signals_dry),
            dtype  = "int16",
            **kw,
        )

    def _write_labels(self, grp: h5py.Group, gt: GroundTruthPackage) -> None:
        labels_grp = grp.create_group("labels")
        kw = dict(compression=self.compression, compression_opts=self.compression_opts)

        labels_grp.create_dataset("doa_az_el",         data=gt.doa_azimuth_elevation,  dtype="float32", **kw)
        labels_grp.create_dataset("source_xyz",        data=gt.source_positions_xyz,   dtype="float32", **kw)
        labels_grp.create_dataset("tdoa_samples",      data=gt.tdoa_samples_per_frame, dtype="float32", **kw)
        labels_grp.create_dataset("vad_frames",        data=gt.vad_masks_per_frame,    dtype=bool,      **kw)
        labels_grp.create_dataset("vad_samples",       data=gt.vad_masks_sample_level, dtype=bool,      **kw)
        
        # 🟢 CORREÇÃO CRÍTICA: Casting para float16. Reduz o tamanho final do Dataset em centenas de Gigabytes.
        labels_grp.create_dataset(
            "spatial_heatmap",   
            data=gt.spatial_heatmap.astype(np.float16), 
            dtype="float16", 
            **kw
        )
        
        labels_grp.create_dataset("si_sdr_per_source", data=gt.si_sdr_per_source,      dtype="float32")

    def _write_metadata(
        self,
        grp:        h5py.Group,
        gt:         GroundTruthPackage,
        aug_result: AugmentedResult,
    ) -> None:
        meta_grp = grp.create_group("metadata")

        meta_grp.attrs["n_sources"]      = gt.n_sources
        meta_grp.attrs["n_frames"]       = gt.n_frames
        meta_grp.attrs["n_samples"]      = gt.n_samples
        meta_grp.attrs["sample_rate"]    = gt.sample_rate
        meta_grp.attrs["rt60_target"]    = float(gt.rt60_target)
        meta_grp.attrs["rt60_measured"]  = float(gt.rt60_measured or -1.0)
        meta_grp.attrs["frame_ms"]       = gt.frame_duration_ms

        meta_grp.create_dataset("room_dims", data=gt.room_dims, dtype="float32")

        # Aug log como JSON
        meta_grp.attrs["augmentation_log"] = json.dumps(aug_result.augmentation_log)

    def close(self) -> None:
        with self._lock:
            if self._current_file and self._current_file.id.valid:
                self._current_file.close()
                logger.info(f"HDF5Writer fechado. Total: {self._global_count} amostras.")


class WebDatasetWriter:
    """
    Escritor de amostras em formato WebDataset (tar shards).
    Compatível com webdataset Python package e PyTorch DataLoader.

    Cada amostra é um conjunto de arquivos dentro do tar:
      {key}.stereo.npy
      {key}.sources.npy
      {key}.labels.npz
      {key}.meta.json
    """

    def __init__(
        self,
        output_dir:      str,
        shard_size_mb:   int = 512,
        resume:          bool = True,
    ):
        self.output_dir    = Path(output_dir)
        self.shard_max_bytes = shard_size_mb * 1024 * 1024
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock          = threading.Lock()
        self._shard_idx     = 0
        self._current_tar   = None
        self._current_bytes = 0
        self._global_count  = 0

        if resume:
            self._resume_state()
        else:
            self._open_new_shard()

    def _resume_state(self) -> None:
        """Recupera o estado a partir dos shards existentes no diretório."""
        shards = sorted(self.output_dir.glob("shard_*.tar"))
        if not shards:
            self._open_new_shard()
            return
            
        last_shard = shards[-1]
        self._shard_idx = int(last_shard.stem.split("_")[1])
        
        total_samples = 0
        for shard in shards:
            try:
                with tarfile.open(str(shard), "r") as t:
                    names = t.getnames()
                    samples_in_shard = len([n for n in names if n.endswith('.meta.json')])
                    total_samples += samples_in_shard
            except Exception as e:
                logger.warning(f"Aviso ao ler {shard} no resume: {e}")
                
        self._global_count = total_samples
        self._shard_idx += 1
        self._open_new_shard()
        logger.info(f"WebDataset: Retomando na amostra {self._global_count} (criando shard {self._shard_idx}).")

    def _open_new_shard(self) -> None:
        if self._current_tar:
            self._current_tar.close()
        fname = self.output_dir / f"shard_{self._shard_idx:05d}.tar"
        self._current_tar   = tarfile.open(str(fname), "w")
        self._current_bytes = 0
        self._shard_idx    += 1

    def _add_numpy_to_tar(
        self,
        tar:       tarfile.TarFile,
        name:      str,
        array:     np.ndarray,
    ) -> int:
        """Adiciona array numpy ao tar sem cópias excessivas de memória."""
        buf = io.BytesIO()
        np.save(buf, array)
        
        # 🟢 CORREÇÃO: Usar .tell() e passar o ponteiro diretamente.
        # Evita a alocação de um novo objeto de bytes gigante na memória RAM.
        size = buf.tell()
        buf.seek(0)

        info = tarfile.TarInfo(name=name)
        info.size = size
        tar.addfile(info, buf)  # Lemos o buffer diretamente da memória stream
        return size

    def write(
        self,
        aug_result: AugmentedResult,
        gt:         GroundTruthPackage,
    ) -> None:
        with self._lock:
            key = f"{self._global_count:08d}"
            written = 0

            # 🟢 CORREÇÃO: Aplica conversão para int16 na serialização dos arrays NPY
            written += self._add_numpy_to_tar(
                self._current_tar,
                f"{key}.stereo.npy",
                float32_to_int16(aug_result.stereo_mix_augmented),
            )
            written += self._add_numpy_to_tar(
                self._current_tar,
                f"{key}.sources.npy",
                float32_to_int16(gt.source_images_mic0),
            )

            # Labels como npz comprimido
            labels_buf = io.BytesIO()
            np.savez_compressed(
                labels_buf,
                doa_az_el       = gt.doa_azimuth_elevation,
                source_xyz      = gt.source_positions_xyz,
                tdoa_samples    = gt.tdoa_samples_per_frame,
                vad_frames      = gt.vad_masks_per_frame,
                # 🟢 CORREÇÃO: Casting do heatmap para float16 no formato .tar
                spatial_heatmap = gt.spatial_heatmap.astype(np.float16),
                si_sdr          = gt.si_sdr_per_source,
            )
            labels_buf.seek(0)
            labels_data = labels_buf.getvalue()
            info = tarfile.TarInfo(name=f"{key}.labels.npz")
            info.size = len(labels_data)
            self._current_tar.addfile(info, io.BytesIO(labels_data))
            written += len(labels_data)

            # Metadados JSON
            meta = {
                "n_sources":    gt.n_sources,
                "sample_rate":  gt.sample_rate,
                "rt60_target":  gt.rt60_target,
                "room_dims":    gt.room_dims.tolist(),
                "aug_log":      aug_result.augmentation_log,
            }
            meta_bytes = json.dumps(meta).encode()
            info = tarfile.TarInfo(name=f"{key}.meta.json")
            info.size = len(meta_bytes)
            self._current_tar.addfile(info, io.BytesIO(meta_bytes))
            written += len(meta_bytes)

            self._current_bytes += written
            self._global_count  += 1

            if self._current_bytes >= self.shard_max_bytes:
                self._open_new_shard()

    def close(self) -> None:
        if self._current_tar:
            self._current_tar.close()