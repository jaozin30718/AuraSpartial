"""
data/datamodule.py
==================
MÓDULO 1: LightningDataModule para dados de áudio sintético.

Suporta dois backends de dados:
  - HDF5: Carregamento indexado aleatório (ideal para treino local)
  - WebDataset: Streaming de tar shards (ideal para dados em nuvem)

A Collate Function customizada aplica Block Masking em tempo real,
garantindo que cada amostra do batch tenha uma máscara única.
"""

from __future__ import annotations

import collections
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pytorch_lightning as pl

from config.training_config import TrainingConfig, TrainingPhase
from data.masking import BlockMaskGenerator
from data.online_augmentations import OnlineAugConfig, OnlineAugmentor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset HDF5
# ─────────────────────────────────────────────────────────────────────────────

class HDF5AudioDataset(Dataset):
    """
    Dataset indexável que lê amostras de arquivos HDF5 shardados.

    Schema esperado (definido no pipeline de geração):
        /sample_{idx}/audio/stereo_mix         [2, S]
        /sample_{idx}/audio/source_images_mic0 [N, S]
        /sample_{idx}/labels/doa_az_el         [N, T_frames, 2]
        /sample_{idx}/labels/spatial_heatmap   [T_frames, Az, El]
        /sample_{idx}/labels/vad_frames        [N, T_frames]

    Args:
        hdf5_dir      : diretório contendo arquivos shard_*.h5
        sample_rate   : taxa de amostragem (Hz)
        max_samples   : limite de amostras (None = todos)
        cache_size    : número de amostras em cache RAM
    """

    def __init__(
        self,
        hdf5_dir:    str,
        sample_rate: int  = 16_000,
        max_samples: Optional[int] = None,
        cache_size:  int  = 512,
    ) -> None:
        super().__init__()
        self.hdf5_dir    = Path(hdf5_dir)
        self.sample_rate = sample_rate
        self.cache_size  = cache_size

        # Indexa todos os shards e constrói índice global
        self._index: List[Tuple[Path, str]] = []  # (arquivo, chave_grupo)
        self._build_index(max_samples)

        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_order: collections.deque = collections.deque()

        # Cache de ponteiros de disco para evitar I/O Thrashing
        self._h5_handles = {}

        logger.info(
            f"HDF5AudioDataset: {len(self._index)} amostras em "
            f"{self.hdf5_dir}"
        )

    def _build_index(self, max_samples: Optional[int]) -> None:
        """Varre todos os shards .h5 e constrói índice (path, chave)."""
        shards = sorted(self.hdf5_dir.glob("shard_*.h5"))
        if not shards:
            raise FileNotFoundError(
                f"Nenhum arquivo shard_*.h5 encontrado em {self.hdf5_dir}"
            )

        for shard_path in shards:
            with h5py.File(str(shard_path), "r") as f:
                for key in f.keys():
                    self._index.append((shard_path, key))
                    if max_samples and len(self._index) >= max_samples:
                        return

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int, _retries: int = 0) -> Dict[str, torch.Tensor]:
        """
        Retorna uma amostra completa do dataset.

        Returns:
            dict com:
                "stereo"   : [2, S]     — áudio estéreo misto
                "sources"  : [N, S]     — fontes isoladas (target BSS)
                "doa"      : [N, 2]     — azimute/elevação (médias temporais)
                "heatmap"  : [Az, El]   — mapa acústico médio temporal
                "vad"      : [N]        — atividade média por fonte
                "n_sources": int        — número de fontes
        """
        shard_path, sample_key = self._index[idx]
        cache_key = f"{shard_path}::{sample_key}"
        shard_str = str(shard_path)

        # Tenta cache de memória
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Mantém o handle do arquivo aberto em RAM por worker
            if shard_str not in self._h5_handles:
                self._h5_handles[shard_str] = h5py.File(shard_str, "r")
                
            f = self._h5_handles[shard_str]
            if sample_key not in f:
                raise KeyError(f"Amostra {sample_key} não encontrada no shard {shard_path}")
                
            grp = f[sample_key]

            # Lê do disco em int16 e converte para float32 no momento
            stereo  = torch.from_numpy(
                grp["audio/stereo_mix"][()].astype(np.float32)
            ) / 32768.0  # [2, S]

            sources = torch.from_numpy(
                grp["audio/source_images_mic0"][()].astype(np.float32)
            ) / 32768.0  # [N, S]

            doa_seq = torch.from_numpy(
                grp["labels/doa_az_el"][()].astype(np.float32)
            )  # [N, T_frames, 2]

            heatmap_seq = torch.from_numpy(
                grp["labels/spatial_heatmap"][()].astype(np.float32)
            )  # [T_frames, Az, El]

            vad_frames = torch.from_numpy(
                grp["labels/vad_frames"][()].astype(np.float32)
            )  # [N, T_frames]

            n_sources = int(grp["metadata"].attrs.get("n_sources", 2))

            doa     = doa_seq.mean(dim=1)           #[N, 2]
            heatmap = heatmap_seq.mean(dim=0)       #[Az, El]
            vad     = vad_frames.mean(dim=1)        # [N]

        except (KeyError, OSError, RuntimeError) as e:
            if _retries > 10:
                import logging
                logging.getLogger("HDF5AudioDataset").error(f"10+ falhas consecutivas, abortando em {idx}")
                raise RuntimeError(f"Falha ao carregar HDF5 após 10 tentativas. Último erro: {e}")
            # Se a amostra estiver corrompida, pula para a próxima de forma recursiva
            import logging
            logging.getLogger("HDF5AudioDataset").warning(
                f"Amostra corrompida pulada: {shard_path}::{sample_key} | Erro: {e}"
            )
            new_idx = (idx + 1) % len(self._index)
            return self.__getitem__(new_idx, _retries=_retries + 1)

        sample = {
            "stereo":   stereo,
            "sources":  sources,
            "doa":      doa,
            "heatmap":  heatmap,
            "vad":      vad,
            "n_sources": torch.tensor(n_sources, dtype=torch.long),
        }

        # Adiciona ao cache
        self._add_to_cache(cache_key, sample)
        return sample

    def _add_to_cache(self, key: str, sample: Dict[str, torch.Tensor]) -> None:
        if len(self._cache_order) >= self.cache_size:
            oldest = self._cache_order.popleft()
            del self._cache[oldest]
        self._cache[key]    = sample
        self._cache_order.append(key)

    def __del__(self) -> None:
        """Garante que todos os file handles HDF5 sejam fechados."""
        for handle in self._h5_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_handles.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Collate Function com Block Masking
# ─────────────────────────────────────────────────────────────────────────────

class AuraCollateFn:
    """
    Collate function customizada que:
    1. Padeia/corta amostras para comprimento uniforme
    2. Computa features DSP (IPD, GCC-PHAT) do áudio estéreo
    3. Aplica Block Masking 2D nos patches espectrais
    4. Retorna tensores prontos para o training_step

    Args:
        mask_generator : BlockMaskGenerator configurado
        stft_params    : parâmetros STFT para extração de features
        n_patches_time : para cálculo da grade de patches
        n_patches_freq : para cálculo da grade de patches
        training_phase : fase atual (muda o output do collate)
    """

    def __init__(
        self,
        mask_generator:  BlockMaskGenerator,
        n_fft:           int = 512,
        hop_length:      int = 128,
        n_patches_time:  int = 16,
        n_patches_freq:  int = 16,
        training_phase:  TrainingPhase = TrainingPhase.JEPA,
        online_aug_config: OnlineAugConfig = None,
        is_training:     bool = True,
    ) -> None:
        self.mask_gen     = mask_generator
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.n_pt         = n_patches_time
        self.n_pf         = n_patches_freq
        self.phase        = training_phase
        self.is_training  = is_training

        # Online augmentations (desativadas para validação)
        aug_cfg = online_aug_config or OnlineAugConfig(enabled=False)
        self.augmentor = OnlineAugmentor(aug_cfg)

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        B = len(batch)

        stereo_list  = [s["stereo"]  for s in batch]  
        sources_list = [s["sources"] for s in batch]  
        doa_list     =[s["doa"]     for s in batch]  
        heatmap_list = [s["heatmap"] for s in batch]  

        stereo  = self._pad_and_stack(stereo_list,  dim=1)   
        sources = self._pad_and_stack_3d(sources_list)        
        doa     = self._pad_and_stack(doa_list, dim=0)        
        heatmap = torch.stack(heatmap_list, dim=0)            

        mask_ctx, mask_tgt = self.mask_gen(
            batch_size = B,
            device     = stereo.device,
        ) 

        output = {
            "stereo":   stereo,      
            "mask_ctx": mask_ctx,    
            "mask_tgt": mask_tgt,    
        }

        if self.phase in (TrainingPhase.MULTITASK, TrainingPhase.LORA_FINETUNE):
            output.update({
                "sources": sources,   
                "doa":     doa,       
                "heatmap": heatmap,   
            })

        # 🟢 Augmentações Online — aplicadas APÓS collate, ANTES de retornar
        output = self.augmentor(output, is_training=self.is_training)

        return output





    @staticmethod
    def _pad_and_stack(
        tensors: List[torch.Tensor],
        dim:     int,
    ) -> torch.Tensor:
        """Padeia tensores para o mesmo shape e empilha no batch."""
        max_len = max(t.shape[dim] for t in tensors)
        padded  = []
        for t in tensors:
            pad_len = max_len - t.shape[dim]
            if pad_len > 0:
                pad_shape = list(t.shape)
                pad_shape[dim] = pad_len
                t = torch.cat([t, torch.zeros(pad_shape)], dim=dim)
            padded.append(t)
        return torch.stack(padded, dim=0)

    @staticmethod
    def _pad_and_stack_3d(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Padeia tensores 3D [N, S] → [B, N_max, S_max]."""
        n_max = max(t.shape[0] for t in tensors)
        s_max = max(t.shape[1] for t in tensors)
        result = torch.zeros(len(tensors), n_max, s_max)
        for i, t in enumerate(tensors):
            result[i, :t.shape[0], :t.shape[1]] = t
        return result


# ─────────────────────────────────────────────────────────────────────────────
# LightningDataModule
# ─────────────────────────────────────────────────────────────────────────────

class AuraDataModule(pl.LightningDataModule):
    """
    LightningDataModule completo para o pipeline AuraSpatial.

    Gerencia:
      - Datasets de treino e validação (HDF5)
      - Collate function com Block Masking
      - Atualização dinâmica da fase (JEPA → Multitask → LoRA)

    Args:
        config: TrainingConfig com todos os parâmetros
        n_patches_time: número de patches temporais (do AuraSpatialConfig)
        n_patches_freq: número de patches de frequência
    """

    def __init__(
        self,
        config:         TrainingConfig,
        n_patches_time: int = 16,
        n_patches_freq: int = 16,
    ) -> None:
        super().__init__()
        self.config        = config
        self.n_patches_time = n_patches_time
        self.n_patches_freq = n_patches_freq

        # Salva hiperparâmetros para o checkpoint do Lightning
        self.save_hyperparameters(ignore=["config"])

        # Mask generator (reutilizado entre treino e val)
        self.mask_generator = BlockMaskGenerator(
            n_patches_time = n_patches_time,
            n_patches_freq = n_patches_freq,
            mask_ratio     = config.jepa.mask_ratio,
            min_block_time = config.jepa.block_mask_min_time,
            max_block_time = config.jepa.block_mask_max_time,
            min_block_freq = config.jepa.block_mask_min_freq,
            max_block_freq = config.jepa.block_mask_max_freq,
        )

        self._current_phase = config.phase
        self._train_dataset: Optional[HDF5AudioDataset] = None
        self._val_dataset:   Optional[HDF5AudioDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Instancia datasets. Chamado em cada processo do DDP."""
        if stage in ("fit", None):
            self._train_dataset = HDF5AudioDataset(
                hdf5_dir    = self.config.train_hdf5_dir,
                sample_rate = 16_000,
            )
            self._val_dataset = HDF5AudioDataset(
                hdf5_dir    = self.config.val_hdf5_dir,
                sample_rate = 16_000,
                max_samples = 1000,   # val set menor para velocidade
            )
            logger.info(
                f"Dataset: {len(self._train_dataset)} treino, "
                f"{len(self._val_dataset)} validação"
            )

    def _build_collate_fn(self, is_training: bool = True) -> AuraCollateFn:
        """Constrói collate function para a fase atual."""
        return AuraCollateFn(
            mask_generator    = self.mask_generator,
            n_patches_time    = self.n_patches_time,
            n_patches_freq    = self.n_patches_freq,
            training_phase    = self._current_phase,
            online_aug_config = self.config.online_aug if is_training else None,
            is_training       = is_training,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size  = self.config.batch_size,
            shuffle     = True,
            num_workers = self.config.num_workers,
            pin_memory  = self.config.pin_memory,
            collate_fn  = self._build_collate_fn(is_training=True),
            drop_last   = True,
            # 🟢 CORREÇÃO: Força a CPU a preparar 4 batches extras na RAM antecipadamente
            persistent_workers = (self.config.num_workers > 0),
            prefetch_factor    = 4 if self.config.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size  = self.config.batch_size * 2,
            shuffle     = False,
            num_workers = max(1, self.config.num_workers // 2),
            pin_memory  = self.config.pin_memory,
            collate_fn  = self._build_collate_fn(is_training=False),
            drop_last   = False,
        )

    def set_phase(self, phase: TrainingPhase) -> None:
        """Atualiza a fase de treinamento (chamado pelo PhaseSwitcher callback)."""
        logger.info(f"DataModule: mudando fase {self._current_phase} → {phase}")
        self._current_phase = phase
