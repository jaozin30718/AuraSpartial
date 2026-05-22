"""
train.py
========
Entry point do pipeline completo de treinamento AuraSpatial.

Suporta 3 modos de execução:
  python train.py --phase jepa           # Fase 1: Pré-treino LeJEPA
  python train.py --phase multitask      # Fase 2: Fine-tuning supervisionado
  python train.py --phase lora_finetune  # Fase 3: LoRA para nova sala

Boas práticas MLOps:
  ✅ Reproducibilidade via seed global
  ✅ Profiling opcional via PyTorch Profiler
  ✅ Gradient checkpointing para modelos grandes
  ✅ Detecção automática de anomalias (torch.autograd.set_detect_anomaly)
  ✅ Resume automático do último checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    EarlyStopping,
    GradientAccumulationScheduler,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

from config.training_config import (
    TrainingConfig,
    TrainingPhase,
    OptimizerConfig,
    SchedulerConfig,
    JEPAPhaseConfig,
    MultitaskPhaseConfig,
    LoRAConfig,
)
from data.online_augmentations import OnlineAugConfig
from data.datamodule import AuraDataModule
from data.masking import BlockMaskGenerator
from lightning_module import AuraLightningModule
from lora_finetuner import LoRAFinetuner
from callbacks.phase_switcher import PhaseSwitcherCallback
from callbacks.wandb_audio_logger import WandBAudioLogger

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(name)-25s | %(levelname)7s | %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
logger = logging.getLogger(__name__)


class CleanupEmptyDirsCallback(pl.Callback):
    """Callback para apagar pastas vazias deixadas pelo ModelCheckpoint."""
    def on_train_epoch_start(self, trainer, pl_module):
        if not hasattr(trainer, 'checkpoint_callback') or trainer.checkpoint_callback is None:
            return
        ckpt_dir = Path(trainer.checkpoint_callback.dirpath)
        if ckpt_dir.exists():
            for p in ckpt_dir.glob("*"):
                if p.is_dir() and not any(p.iterdir()):
                    try:
                        p.rmdir()
                    except Exception:
                        pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AuraSpatial Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type    = str,
        default = "jepa",
        choices = ["jepa", "multitask", "lora_finetune"],
        help    = "Fase de treinamento",
    )
    parser.add_argument(
        "--checkpoint",
        type    = str,
        default = None,
        help    = "Path para checkpoint de resume/base (para Fase 2 e 3)",
    )
    parser.add_argument(
        "--resume-last", 
        action="store_true", 
        help="Carrega automaticamente o último checkpoint (last.ckpt) se existir"
    )
    parser.add_argument("--batch-size",  type=int,   default=4)     # GTX 1050 2GB
    parser.add_argument("--accumulate-grad-batches", type=int, default=None, help="Passos de acumulação de gradiente")
    parser.add_argument("--max-epochs",  type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--devices",     type=int,   default=1)
    parser.add_argument("--strategy",    type=str,   default="auto")
    parser.add_argument("--precision",   type=str,   default="16-mixed")  # fp16 (bf16 requer Ampere+)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--profile",     action="store_true", help="Ativa profiler")
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--compile",     action="store_true", help="Compila modelo (PyTorch 2.0+)")
    parser.add_argument(
        "--data-dir",
        type    = str,
        default = str(Path(__file__).parent.parent / "synthetic_audio_pipeline" / "dataset"),
    )
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--wandb-project", type=str, default="aura-spatial")
    parser.add_argument("--no-wandb", action="store_true", help="Desativa WandB")
    parser.add_argument("--no-checkpointing", action="store_true", help="Desativa Gradient Checkpointing")

    return parser.parse_args()


def build_trainer(
    args:       argparse.Namespace,
    config:     TrainingConfig,
    callbacks:  list,
    logger_obj,
    total_steps: int,
) -> Trainer:
    """Constrói o PyTorch Lightning Trainer com todas as configurações."""

    profiler = None
    if args.profile:
        profiler = PyTorchProfiler(
            dirpath      = f"{args.output_dir}/profiler",
            filename     = f"profile_{args.phase}",
            row_limit    = 20,
            sort_by_key  = "cuda_time_total",
        )

    return Trainer(
        # Epochs e steps
        max_epochs             = args.max_epochs,

        # Hardware
        accelerator            = "auto",
        devices                = args.devices,
        strategy               = args.strategy,
        precision              = args.precision,

        # Otimização de memória
        accumulate_grad_batches = config.accumulate_grad_batches,
        gradient_clip_val       = config.optimizer.gradient_clip_val,
        gradient_clip_algorithm = config.optimizer.gradient_clip_algo,

        # Callbacks
        callbacks              = callbacks,

        # Logging
        logger                 = logger_obj,
        log_every_n_steps      = config.log_every_n_steps,

        # Validação
        val_check_interval     = 1.0,    # valida a cada epoch
        num_sanity_val_steps   = 2,      # 2 batches de sanity check

        # Reprodutibilidade
        deterministic          = False,   # True = mais lento mas reprodutível
        benchmark              = True,    # cuDNN benchmark para performance

        # Profiler
        profiler               = profiler,

        # Detecção de anomalias (apenas em debug)
        detect_anomaly         = args.detect_anomaly,

        # Enable para modelos grandes
        enable_progress_bar    = True,
        enable_model_summary   = True,
    )


def main() -> None:
    args = parse_args()

    # ── Compatibilidade PyTorch 2.6+ ──────────────────────────────────────────
    torch.serialization.add_safe_globals([
        TrainingConfig, 
        TrainingPhase,
        OptimizerConfig,
        SchedulerConfig,
        JEPAPhaseConfig,
        MultitaskPhaseConfig,
        LoRAConfig,
        OnlineAugConfig,
    ])

    # ── Reproducibilidade ─────────────────────────────────────────────────────
    seed_everything(args.seed, workers=True)

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        logger.warning("Detecção de anomalias habilitada — treino será mais lento!")

    # ── Configuração ──────────────────────────────────────────────────────────
    phase = TrainingPhase(args.phase)

    # Tenta usar dataset_val separado; se não existir, usa o mesmo do treino
    val_dir_candidate = args.data_dir.replace("dataset", "dataset_val")
    if not Path(val_dir_candidate).exists() or val_dir_candidate == args.data_dir:
        val_dir_candidate = args.data_dir
        logger.warning(
            f"Diretório de validação não encontrado. "
            f"Usando o mesmo dataset para treino e validação: {args.data_dir}"
        )

    config = TrainingConfig(
        phase          = phase,
        train_hdf5_dir = args.data_dir,
        val_hdf5_dir   = val_dir_candidate,
        batch_size     = args.batch_size,
        checkpoint_dir = args.output_dir,
        wandb_project  = args.wandb_project,
    )
    config.optimizer.lr = args.lr
    if args.accumulate_grad_batches is not None:
        config.accumulate_grad_batches = args.accumulate_grad_batches

    # Resolve o checkpoint a ser carregado
    checkpoint_to_load = args.checkpoint
    if args.resume_last:
        last_ckpt = Path(args.output_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            checkpoint_to_load = str(last_ckpt)
            logger.info(f"Flag --resume-last ativada. Carregando automaticamente: {checkpoint_to_load}")
        else:
            logger.warning("Flag --resume-last ativada, mas 'last.ckpt' não foi encontrado.")

    # ── Fase 3: LoRA (pipeline separado) ─────────────────────────────────────
    if phase == TrainingPhase.LORA_FINETUNE:
        if not checkpoint_to_load:
            raise ValueError("--checkpoint ou --resume-last é obrigatório para LoRA fine-tuning!")

        finetuner = LoRAFinetuner(
            base_checkpoint   = checkpoint_to_load,
            finetune_data_dir = args.data_dir,
            config            = config,
            output_dir        = f"{args.output_dir}/lora_adapters",
            max_epochs        = args.max_epochs,
        )
        pl_module = finetuner.run()
        finetuner.save_adapter(pl_module)
        return

    # ── Fases 1 e 2: Pipeline principal ──────────────────────────────────────

    # Adiciona a pasta pai (IA) ao sys.path para conseguir importar aura_spatial
    sys.path.append(str(Path(__file__).parent.parent))
    
    from aura_spatial.model import AuraSpatialModel
    from aura_spatial.config import AuraSpatialConfig

    model_config = AuraSpatialConfig()
    model_config.predictor.use_checkpointing = not args.no_checkpointing
    model        = AuraSpatialModel(model_config)

    # Total de steps para o scheduler
    steps_per_epoch = max(1, 100_000 // args.batch_size)   # estimativa
    total_steps     = steps_per_epoch * args.max_epochs

    # Módulo Lightning
    if checkpoint_to_load and Path(checkpoint_to_load).exists():
        logger.info(f"Resumindo de checkpoint: {checkpoint_to_load}")
        pl_module = AuraLightningModule.load_from_checkpoint(
            checkpoint_to_load,
            model       = model,
            config      = config,
            total_steps = total_steps,
            strict      = False,
        )
        pl_module.phase = phase
        pl_module._freeze_for_phase(phase)
    else:
        pl_module = AuraLightningModule(
            model       = model,
            config      = config,
            total_steps = total_steps,
        )

    # DataModule
    dm = AuraDataModule(
        config         = config,
        n_patches_time = model_config.patch_embedder.n_patches_time,
        n_patches_freq = model_config.patch_embedder.n_patches_freq,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath        = f"{args.output_dir}/checkpoints/{phase.value}",
        filename       = f"aura_{{epoch:03d}}_{{val/loss_total:.4f}}_{phase.value}",
        monitor        = config.monitor_metric,
        mode           = config.monitor_mode,
        save_top_k     = 1,
        every_n_epochs = 1,
        save_last      = False,
        verbose        = True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress    = RichProgressBar()
    audio_logger = WandBAudioLogger(
        log_every_n_epochs = 5,
        n_samples_to_log   = 4,
    )

    # PhaseSwitcher (apenas na Fase 1 para transição automática)
    phase_switcher = PhaseSwitcherCallback(
        config      = config,
        datamodule  = dm,
        auto_switch = (phase == TrainingPhase.JEPA),
    )

    callbacks = [
        checkpoint_cb,
        lr_monitor,
        progress,
        audio_logger,
        phase_switcher,
        CleanupEmptyDirsCallback(),
    ]

    from callbacks.gradnorm_callback import GradNormCallback, GradientHealthMonitor

    # Adiciona GradNorm apenas quando estratégia = "gradnorm"
    if config.multitask.balancing_strategy == "gradnorm":
        callbacks.extend([
            GradNormCallback(
                shared_module_name    = "predictor",
                update_freq           = config.multitask.gradnorm_update_freq,
                log_freq              = config.log_every_n_steps,
                log_grad_norms        = True,
                log_weight_histogram  = True,
                imbalance_alert_ratio = 15.0,
                use_last_layer_only   = True,
            ),
            GradientHealthMonitor(
                explosion_threshold   = 50.0,
                vanishing_threshold   = 1e-7,
                vanishing_patience    = 30,
                check_freq            = 50,
                module_names_to_check = ["encoder", "predictor"],
            ),
        ])

    # ── Logger WandB ──────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project  = args.wandb_project,
            name     = f"{config.wandb_run_name}_{phase.value}",
            tags     = [phase.value, f"lr_{args.lr}", f"bs_{args.batch_size}"],
            log_model = True,
        )
        # Loga o modelo para o Model Registry do WandB
        wandb_logger.watch(model, log="parameters", log_freq=100)
    else:
        from pytorch_lightning.loggers import CSVLogger
        wandb_logger = CSVLogger(
            save_dir = args.output_dir,
            name     = f"csv_logs_{phase.value}",
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = build_trainer(
        args        = args,
        config      = config,
        callbacks   = callbacks,
        logger_obj  = wandb_logger,
        total_steps = total_steps,
    )

    # ── Log de configuração inicial ───────────────────────────────────────────
    param_counts = model.count_parameters()
    logger.info(
        f"\n{'='*60}\n"
        f"  AuraSpatial Training\n"
        f"  Fase         : {phase.value}\n"
        f"  Parâmetros   : {param_counts['total']:,} total\n"
        f"  Dispositivos : {args.devices}x {args.precision}\n"
        f"  Batch size   : {args.batch_size} "
        f"(efetivo: {args.batch_size * config.accumulate_grad_batches * args.devices})\n"
        f"  Learning Rate: {args.lr}\n"
        f"  Total steps  : {total_steps:,}\n"
        f"{'='*60}"
    )

    # ── torch.compile ─────────────────────────────────────────────────────────
    if args.compile:
        logger.info("[INFO] Compilando modelo com torch.compile... (O primeiro batch será lento!)")
        pl_module.model = torch.compile(pl_module.model)

    # ── Treina ────────────────────────────────────────────────────────────────
    # Se o checkpoint for de uma fase diferente (ex: jepa -> multitask), 
    # não passamos ckpt_path para trainer.fit para evitar resumir otimizador e epochs antigas.
    resume_fit_ckpt = None
    if checkpoint_to_load and Path(checkpoint_to_load).exists():
        if phase.value in Path(checkpoint_to_load).name:
            # 🟢 Modificado: Não retomar o estado do otimizador (resume_fit_ckpt = None) 
            # para evitar o erro 'different number of parameter groups' causado pela 
            # introdução do bitsandbytes ou congelamento de backbone.
            logger.info(f"Pesos retomados de {Path(checkpoint_to_load).name}. Estado do otimizador foi resetado por segurança.")
        else:
            logger.info(f"Iniciando fase '{phase.value}' (pesos carregados de {Path(checkpoint_to_load).name}). O estado de treino não será resumido.")

    trainer.fit(
        pl_module,
        datamodule   = dm,
        ckpt_path    = resume_fit_ckpt,
    )

    # ── Avaliação Final ───────────────────────────────────────────────────────
    logger.info("Iniciando avaliação no melhor checkpoint...")
    trainer.validate(pl_module, datamodule=dm, ckpt_path="best")

    logger.info(
        f"\n[OK] Treinamento Fase '{phase.value}' concluído!\n"
        f"   Melhor checkpoint: {checkpoint_cb.best_model_path}\n"
        f"   Melhor métrica   : {checkpoint_cb.best_model_score:.6f}"
    )


if __name__ == "__main__":
    main()


