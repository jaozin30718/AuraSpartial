"""
config/training_config.py
=========================
Dataclasses tipadas para todas as fases de treinamento.
Centraliza hiperparâmetros com validação e defaults documentados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from data.online_augmentations import OnlineAugConfig


class TrainingPhase(str, Enum):
    """Enum das fases de curriculum learning."""
    JEPA          = "jepa"           # Fase 1: Auto-supervisionado
    MULTITASK     = "multitask"      # Fase 2: Supervisionado multitarefa
    LORA_FINETUNE = "lora_finetune"  # Fase 3: Fine-tuning de borda com LoRA


@dataclass
class OptimizerConfig:
    """Configuração do otimizador AdamW."""
    lr:           float = 1e-4
    weight_decay: float = 0.04
    betas:        Tuple[float, float] = (0.9, 0.999)
    eps:          float = 1e-8
    # Gradient clipping global
    gradient_clip_val:  float = 1.0
    gradient_clip_algo: str   = "norm"    # "norm" | "value"


@dataclass
class SchedulerConfig:
    """Configuração do scheduler Cosine Annealing com Linear Warmup."""
    warmup_ratio:  float = 0.10    # 10% das steps para warmup linear
    min_lr_ratio:  float = 0.01    # lr mínimo = min_lr_ratio * lr_max
    # Número de epochs por fase (usado para calcular total de steps)
    phase1_epochs: int = 100
    phase2_epochs: int = 50
    phase3_epochs: int = 10


@dataclass
class JEPAPhaseConfig:
    """Configuração da Fase 1: Treinamento Auto-supervisionado LeJEPA."""
    # Pesos das componentes da perda LeJEPA
    alpha_var:  float = 15.0      # peso de L_var (variância)
    beta_cov:   float = 15.0      # peso de L_cov (covariância)
    gamma_pred: float = 1.0       # peso de L_pred

    # Mascaramento em bloco
    mask_ratio:        float = 0.75
    block_mask_min_time:  int = 4   # tamanho mínimo do bloco temporal
    block_mask_max_time:  int = 16  # tamanho máximo do bloco temporal
    block_mask_min_freq:  int = 2   # tamanho mínimo do bloco de frequência
    block_mask_max_freq:  int = 8   # tamanho máximo do bloco de frequência

    # Tipo de perda preditiva
    pred_loss_type: str = "smooth_l1"   # "l2" | "smooth_l1" | "cosine"

    # Opções de gradiente
    # Se True: detach no z_target (LeJEPA clássico)
    # Se False: gradiente flui por ambos os encoders (mais instável mas rico)
    detach_target: bool = True


@dataclass
class MultitaskPhaseConfig:
    """Configuração da Fase 2: Treinamento Supervisionado Multitarefa."""
    # Pesos iniciais (antes do GradNorm)
    initial_weight_bss:     float = 1.0
    initial_weight_doa:     float = 1.0
    initial_weight_heatmap: float = 1.0

    # Estratégia de balanceamento
    # "uncertainty": Uncertainty Weighting (Kendall et al. 2018)
    # "gradnorm"   : GradNorm (Chen et al. 2018)
    # "fixed"      : Pesos fixos
    balancing_strategy: str = "uncertainty"

    # GradNorm específico
    gradnorm_alpha:      float = 1.5   # assimetria do GradNorm
    gradnorm_update_freq: int  = 10    # steps entre updates do GradNorm

    # SI-SDR
    sdr_pit: bool = True   # Permutation Invariant Training

    # Heatmap: raio da gaussiana sintética em bins
    heatmap_gaussian_sigma: float = 2.0

    # 🟢 O CHEAT CODE DE VELOCIDADE: 
    # Congela o Encoder e o Mamba. Treina APENAS os Heads. 
    # O Backprop fica 10x mais leve e a GPU 1050 voa.
    freeze_backbone: bool = True   
    backbone_lr_scale: float = 0.1


@dataclass
class LoRAConfig:
    """Configuração da Fase 3: Fine-tuning com LoRA (PEFT)."""
    r:             int   = 8        # rank das matrizes LoRA
    lora_alpha:    int   = 16       # escala LoRA (α/r = escala efetiva)
    lora_dropout:  float = 0.05
    lr:            float = 1e-3     # lr maior para fine-tuning rápido
    weight_decay:  float = 0.01
    # Módulos alvo dentro do Mamba onde LoRA será injetado
    target_modules: List[str] = field(default_factory=lambda: [
        "in_proj", "x_proj", "out_proj",   # projeções internas do Mamba
        "dt_proj",                          # projeção de time-step
    ])
    # Módulos a congelar (tudo exceto LoRA)
    modules_to_save: List[str] = field(default_factory=lambda: [])


@dataclass
class TrainingConfig:
    """Configuração raiz de todo o processo de treinamento."""
    # Fase atual
    phase: TrainingPhase = TrainingPhase.JEPA

    # Sub-configs
    optimizer:  OptimizerConfig      = field(default_factory=OptimizerConfig)
    scheduler:  SchedulerConfig      = field(default_factory=SchedulerConfig)
    jepa:       JEPAPhaseConfig      = field(default_factory=JEPAPhaseConfig)
    multitask:  MultitaskPhaseConfig = field(default_factory=MultitaskPhaseConfig)
    lora:       LoRAConfig           = field(default_factory=LoRAConfig)
    online_aug: OnlineAugConfig      = field(default_factory=OnlineAugConfig)

    # Hardware
    precision:               str  = "bf16-mixed"   # "32" | "16-mixed" | "bf16-mixed"
    accumulate_grad_batches: int  = 2     # batch efetivo = batch_size * 2 (32×2=64)
    devices:                 int  = 1              # GPUs
    strategy:                str  = "auto"         # "ddp" | "fsdp" | "auto"

    # Dataset
    train_hdf5_dir: str  = "./synthetic_dataset"
    val_hdf5_dir:   str  = "./synthetic_dataset_val"
    batch_size:     int  = 32
    num_workers:    int  = 2
    pin_memory:     bool = True

    # WandB
    wandb_project:  str  = "aura-spatial"
    wandb_entity:   str  = ""
    wandb_run_name: str  = "run-001"
    log_every_n_steps: int = 50

    # Checkpointing
    checkpoint_dir:   str  = "./checkpoints"
    save_top_k:       int  = 3
    monitor_metric:   str  = "val/loss_total"
    monitor_mode:     str  = "min"

    # Curriculum (thresholds para troca automática de fase)
    # 🟢 CORREÇÃO: Monitorar a loss preditiva real, não a total
    phase1_to_phase2_metric:    str   = "val/loss_pred" 
    
    # 🟢 CORREÇÃO: O ponto de ouro (0.25)
    phase1_to_phase2_threshold: float = 0.25   
    
    phase1_min_epochs:          int   = 20

