"""
callbacks/phase_switcher.py
===========================
Callback que implementa o Curriculum Learning automático,
trocando entre as fases de treinamento baseado em métricas.
"""

from __future__ import annotations

import logging
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from config.training_config import TrainingPhase, TrainingConfig

logger = logging.getLogger(__name__)


class PhaseSwitcherCallback(Callback):
    """
    Gerencia a transição automática entre as fases de Curriculum Learning:
        Fase 1 (JEPA) → Fase 2 (Multitask) → Fase 3 (LoRA Fine-tune)

    A transição ocorre quando:
      1. A métrica de monitoramento (ex: val/loss_jepa_total) cai abaixo
         de um threshold configurado, E
      2. O número mínimo de epochs na fase atual foi cumprido.

    Args:
        config          : TrainingConfig com thresholds e limites
        datamodule      : AuraDataModule (para atualizar a fase da collate)
        auto_switch     : Se False, apenas registra; não faz switch automático
    """

    def __init__(
        self,
        config:      TrainingConfig,
        datamodule,                    # AuraDataModule
        auto_switch: bool = True,
    ) -> None:
        super().__init__()
        self.config      = config
        self.dm          = datamodule
        self.auto_switch = auto_switch
        self._epochs_in_current_phase: int = 0
        self._current_phase: TrainingPhase = config.phase

    def on_validation_epoch_end(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Verifica condições de troca após cada epoch de validação."""
        self._epochs_in_current_phase += 1

        if not self.auto_switch:
            return

        # ── Fase 1 → Fase 2 ───────────────────────────────────────────────────
        if (
            self._current_phase == TrainingPhase.JEPA
            and self._epochs_in_current_phase >= self.config.phase1_min_epochs
        ):
            metric = trainer.callback_metrics.get(
                self.config.phase1_to_phase2_metric
            )
            if (
                metric is not None
                and metric.item() < self.config.phase1_to_phase2_threshold
            ):
                self._switch_to(TrainingPhase.MULTITASK, trainer, pl_module)

    def _switch_to(
        self,
        new_phase: TrainingPhase,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Executa a transição de fase."""
        old_phase = self._current_phase
        logger.info(
            f"\n{'='*60}\n"
            f"  CURRICULUM: {old_phase.value} → {new_phase.value}\n"
            f"  Epoch: {trainer.current_epoch}\n"
            f"{'='*60}"
        )

        self._current_phase = new_phase
        self._epochs_in_current_phase = 0

        # Atualiza o módulo Lightning
        pl_module.set_training_phase(new_phase)

        # Atualiza o DataModule
        if self.dm is not None:
            self.dm.set_phase(new_phase)

        # Loga a transição no WandB
        if trainer.logger:
            trainer.logger.log_metrics(
                {"curriculum/phase": new_phase.value},
                step=trainer.global_step,
            )



