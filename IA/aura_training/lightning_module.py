"""
lightning_module.py
===================
AuraLightningModule: Orquestrador PyTorch Lightning do modelo AuraSpatial.

Implementa o Curriculum Learning em 3 fases:
  - Fase 1 (JEPA):      Pré-treino auto-supervisionado sem labels
  - Fase 2 (Multitask): Supervisionado com BSS + DoA + Heatmap
  - Fase 3 (LoRA):      Fine-tuning de borda com adaptação acústica

Boas práticas MLOps implementadas:
  ✅ Gradient clipping configurável
  ✅ Mixed precision (bf16/fp16)
  ✅ Gradient accumulation
  ✅ WandB logging com histogramas e visuais
  ✅ Checkpointing por métrica monitorada
  ✅ Learning rate scheduling com warmup
  ✅ Detecção e alerta de NaN/Inf na loss
  ✅ Métricas de validação separadas por fase
  ✅ Frozen/unfrozen dinâmico de módulos
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config.training_config import (
    TrainingConfig, TrainingPhase,
    JEPAPhaseConfig, MultitaskPhaseConfig,
)
from losses.bss_loss import BSSLoss
from losses.doa_loss import CircularMSELoss
from losses.multitask_loss import UncertaintyWeighting, GradNormBalancer

# Import do modelo (definido no script anterior)
# from model import AuraSpatialModel
# from modules.jepa_loss import LeJEPALoss

logger = logging.getLogger(__name__)


class AuraLightningModule(pl.LightningModule):
    """
    Módulo Lightning principal que orquestra todo o treinamento do AuraSpatial.

    Args:
        model          : instância de AuraSpatialModel
        config         : TrainingConfig completo
        total_steps    : total de steps de treino (para scheduler cosine)

    Attributes:
        model          : AuraSpatialModel
        phase          : TrainingPhase atual
        loss_fn_jepa   : LeJEPALoss (Fase 1)
        loss_fn_bss    : BSSLoss com PIT SI-SDR (Fase 2)
        loss_fn_doa    : CircularMSELoss (Fase 2)
        balancer       : UncertaintyWeighting ou GradNormBalancer (Fase 2)
    """

    def __init__(
        self,
        model,                          # AuraSpatialModel
        config:      TrainingConfig,
        total_steps: int = 100_000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model       = model
        self.cfg         = config
        self.phase       = config.phase
        self.total_steps = total_steps

        # ── Funções de Perda ──────────────────────────────────────────────────

        # Fase 1: LeJEPA (importado do módulo anterior)
        # self.loss_fn_jepa = LeJEPALoss(model.config.loss)
        # Aqui referenciamos diretamente do modelo para consistência:
        self.loss_fn_jepa = model.loss_fn   # LeJEPALoss

        # Fase 2: Supervisionadas
        self.loss_fn_bss = BSSLoss(
            n_fft      = 512,
            hop_length = 128,
            pit        = config.multitask.sdr_pit,
        )
        self.loss_fn_doa = CircularMSELoss(
            azimuth_weight   = 1.0,
            elevation_weight = 0.5,   # elevação tipicamente menos crítica
        )
        # pos_weight compensa o desbalanceamento: grid 36×18 = 648 pixels, ~3 positivos/amostra
        # razão negativo/positivo ≈ 200:1 → pos_weight=50 para forçar o modelo a aprender os picos
        self.loss_fn_heatmap = nn.BCEWithLogitsLoss(
            reduction  = "mean",
            pos_weight = torch.tensor(50.0),  # pesa 50x mais os pixels positivos
        )

        # Balanceamento multitarefa
        if config.multitask.balancing_strategy == "uncertainty":
            self.balancer = UncertaintyWeighting(n_tasks=3)
        elif config.multitask.balancing_strategy == "gradnorm":
            self.balancer = GradNormBalancer(
                n_tasks     = 3,
                alpha       = config.multitask.gradnorm_alpha,
                update_freq = config.multitask.gradnorm_update_freq,
            )
        else:
            # Fixed weights
            self.balancer = None

        # ── Inicialização de fase ─────────────────────────────────────────────
        self._freeze_for_phase(self.phase)

        # ── Tracking de métricas ──────────────────────────────────────────────
        self._train_loss_accum: List[float] = []
        self._step_outputs: List[Dict] = []

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Hook nativo do Lightning que roda assim que o batch chega na GPU."""
        if "stereo" in batch and "features" not in batch:
            batch["features"] = self._compute_dsp_features(batch["stereo"])
        return batch

    def _compute_dsp_features(self, stereo: torch.Tensor) -> torch.Tensor:
        """Calcula as features DSP diretamente na GPU usando Tensor Cores."""
        n_fft = 512
        hop_length = 128
        window = torch.hann_window(n_fft, device=stereo.device)

        stft_L = torch.stft(stereo[:, 0], n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        stft_R = torch.stft(stereo[:, 1], n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

        mag_L = torch.log1p(stft_L.abs())
        mag_R = torch.log1p(stft_R.abs())

        cross_spectrum = stft_L * stft_R.conj()
        ipd = torch.angle(cross_spectrum)
        cos_ipd = torch.cos(ipd)
        sin_ipd = torch.sin(ipd)

        eps = 1e-8
        gcc_phat = (cross_spectrum / (cross_spectrum.abs() + eps)).real
        
        features = torch.stack([mag_L, mag_R, cos_ipd, sin_ipd, gcc_phat], dim=1)
        features = features.transpose(2, 3)

        n_pt = self.model.config.patch_embedder.n_patches_time
        n_pf = self.model.config.patch_embedder.n_patches_freq
        T_target, F_target = n_pt * 8, n_pf * 16
        T_cur, F_cur = features.shape[2], features.shape[3]

        if T_cur < T_target:
            features = torch.nn.functional.pad(features, (0, 0, 0, T_target - T_cur))
        else:
            features = features[:, :, :T_target, :]
        
        if F_cur < F_target:
            features = torch.nn.functional.pad(features, (0, F_target - F_cur))
        else:
            features = features[:, :, :, :F_target]

        mags = features[:, :2, :, :]
        mean = mags.mean(dim=(-2, -1), keepdim=True)
        std  = mags.std(dim=(-2, -1), keepdim=True) + eps
        features[:, :2, :, :] = (mags - mean) / std
        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Gestão de Fases
    # ─────────────────────────────────────────────────────────────────────────

    def set_training_phase(self, phase: TrainingPhase) -> None:
        """
        Muda a fase de treinamento e ajusta parâmetros frozen/unfrozen.
        Chamado pelo PhaseSwitcherCallback.
        """
        logger.info(f"LightningModule: mudando fase {self.phase} → {phase}")
        self.phase = phase
        self._freeze_for_phase(phase)

        # Reconfigura otimizador para nova fase
        self.trainer.strategy.setup_optimizers(self.trainer)

    def _freeze_for_phase(self, phase: TrainingPhase) -> None:
        """
        Configura quais parâmetros estão frozen para cada fase.

        Fase 1 (JEPA):
          - Tronco completo: TREINÁVEL
          - Output heads (BSS, DoA, Heatmap): CONGELADOS

        Fase 2 (Multitask):
          - Tronco: TREINÁVEL (com lr reduzido)
          - Output heads: TREINÁVEL

        Fase 3 (LoRA):
          - Tronco: CONGELADO
          - LoRA adapters: TREINÁVEL
          - Output heads: CONGELADOS (fine-tune apenas acústico)
        """
        if phase == TrainingPhase.JEPA:
            self._set_grad(self.model.patch_embedder, True)
            self._set_grad(self.model.encoder,        True)
            self._set_grad(self.model.predictor,      True)
            self._set_grad(self.model.bss_head,       False)
            self._set_grad(self.model.doa_head,       False)
            self._set_grad(self.model.heatmap_head,   False)
            logger.info("Fase 1: Heads congelados, tronco treinável.")

        elif phase == TrainingPhase.MULTITASK:
            backbone_trainable = not self.cfg.multitask.freeze_backbone
            self._set_grad(self.model.patch_embedder, backbone_trainable)
            self._set_grad(self.model.encoder,        backbone_trainable)
            self._set_grad(self.model.predictor,      backbone_trainable)
            self._set_grad(self.model.bss_head,       True)
            self._set_grad(self.model.doa_head,       True)
            self._set_grad(self.model.heatmap_head,   True)
            logger.info(
                f"Fase 2: Heads treináveis. "
                f"Tronco {'treinável' if backbone_trainable else 'congelado'}."
            )

        elif phase == TrainingPhase.LORA_FINETUNE:
            # LoRA: tudo congelado exceto os adapters (gerenciado pela PEFT)
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Fase 3 LoRA: todos os params congelados (PEFT gerencia adapters).")

    @staticmethod
    def _set_grad(module: nn.Module, requires_grad: bool) -> None:
        """Ativa/desativa gradientes de todos os parâmetros de um módulo."""
        for param in module.parameters():
            param.requires_grad = requires_grad

    # ─────────────────────────────────────────────────────────────────────────
    # MÓDULO 2: Training Step — Fase 1 (LeJEPA Auto-supervisionado)
    # ─────────────────────────────────────────────────────────────────────────

    def _training_step_jepa(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features = batch["features"]    
        mask_ctx = batch["mask_ctx"]    
        mask_tgt = batch["mask_tgt"]

        # Forward pretraining já otimizado na classe Model
        z_pred, z_target, mask, metrics = self.model.forward_pretraining(features, mask=mask_ctx)
        
        loss_total, metrics = self.loss_fn_jepa(z_pred, z_target, mask=mask)
        loss_total = self._check_loss_health(loss_total, "jepa")

        self.log_dict({
            "train/loss_jepa_total": loss_total,
            "train/loss_pred":       metrics.get("loss/pred", 0.0),
            "train/loss_var":        metrics.get("loss/var",  0.0),
            "train/loss_cov":        metrics.get("loss/cov",  0.0),
            "train/lr":              self._get_current_lr(),
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss_total

    # ─────────────────────────────────────────────────────────────────────────
    # MÓDULO 3: Training Step — Fase 2 (Multitarefa Supervisionada)
    # ─────────────────────────────────────────────────────────────────────────

    def _training_step_multitask(
        self,
        batch:     Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Forward pass e loss para Fase 2: Multitarefa Supervisionada.

        Pipeline:
            features [B,5,T,F]
              → PatchEmbedder → tokens [B,L,D]
              → JEPAEncoder   → z [B,L,D]
              → MambaPredictor → h [B,L,D]
              ├─→ BSSHead     → masks [B,N*2,T,F]
              ├─→ DoAHead     → doa_pred [B,N,2]
              └─→ HeatmapHead → heatmap_pred [B,1,Rx,Ry]

            Losses:
              L_bss     = -SI-SDR (PIT)
              L_doa     = Circular MSE
              L_heatmap = BCE com gaussiana sintética 2D

            Balanceamento:
              L_total = UncertaintyWeighting([L_bss, L_doa, L_heatmap])
              ou      = GradNorm([...])

        Args:
            batch : output do AuraCollateFn multitask (contém labels supervisionados)

        Returns:
            loss: escalar diferenciável
        """
        cfg: MultitaskPhaseConfig = self.cfg.multitask

        features  = batch["features"]   # [B, 5, T, F] pré-computado na CPU
        sources   = batch["sources"]    # [B, N, S]
        doa_true  = batch["doa"]        # [B, N, 2]
        heatmap_t = batch["heatmap"]    # [B, Az, El]
        stereo    = batch["stereo"]     # 🟢 CORREÇÃO CRÍTICA AQUI (Adicionado)

        # ── Forward Completo ──────────────────────────────────────────────────
        outputs = self.model(features)

        masks_pred   = outputs["masks"]    # [B, N*2, T, F]
        doa_pred     = outputs["doa"]      # [B, N, 2]
        heatmap_pred = outputs["heatmap"]  # [B, 1, Rx, Ry]

        # ── L_BSS: SI-SDR com PIT ─────────────────────────────────────────────
        loss_bss, si_sdrs = self.loss_fn_bss(
            masks      = masks_pred,
            stereo     = stereo,
            references = sources,
        )

        # ── L_DoA: MSE Circular ───────────────────────────────────────────────
        # Alinha número de fontes (padding pode criar fontes extras)
        N_pred = doa_pred.shape[1]
        N_true = doa_true.shape[1]
        N      = min(N_pred, N_true)

        loss_doa = self.loss_fn_doa(
            pred   = doa_pred[:, :N],
            target = doa_true[:, :N],
        )

        # ── L_Heatmap: BCE com Gaussiana 2D Sintética ─────────────────────────
        # O heatmap_t já é a gaussiana 2D gerada no pipeline de augmentation
        # Redimensiona se necessário para coincidir com a predição
        if heatmap_t.shape != heatmap_pred[:, 0].shape:
            heatmap_t_resized = F.interpolate(
                heatmap_t.unsqueeze(1).float(),   # [B, 1, Az, El]
                size  = heatmap_pred.shape[-2:],
                mode  = "bilinear",
                align_corners = False,
            ).squeeze(1)  # [B, Rx, Ry]
        else:
            heatmap_t_resized = heatmap_t.float()

        # BCE com logits (heatmap_pred ainda é pré-sigmoid internamente)
        loss_heatmap = self.loss_fn_heatmap(
            heatmap_pred.squeeze(1),   # [B, Rx, Ry]
            heatmap_t_resized,
        )

        # ── Balanceamento de Gradiente ─────────────────────────────────────────
        losses = [loss_bss, loss_doa, loss_heatmap]

        if self.balancer is not None:
            if isinstance(self.balancer, UncertaintyWeighting):
                loss_total, weight_log = self.balancer(losses)

            elif isinstance(self.balancer, GradNormBalancer):
                loss_total, weight_log = self.balancer(losses)
        else:
            # Fixed weights
            w1 = self.cfg.multitask.initial_weight_bss
            w2 = self.cfg.multitask.initial_weight_doa
            w3 = self.cfg.multitask.initial_weight_heatmap
            loss_total = w1 * loss_bss + w2 * loss_doa + w3 * loss_heatmap
            weight_log = {
                "fixed/weight_bss": w1,
                "fixed/weight_doa": w2,
                "fixed/weight_heatmap": w3,
            }

        # ── Verificação de saúde ───────────────────────────────────────────────
        loss_total = self._check_loss_health(loss_total, "multitask")

        # ── Logging ───────────────────────────────────────────────────────────
        log_dict = {
            "train/loss_multitask_total": loss_total,
            "train/loss_bss":             loss_bss,
            "train/loss_doa":             loss_doa,
            "train/loss_heatmap":         loss_heatmap,
            "train/si_sdr_mean":          si_sdrs.mean(),
            "train/lr":                   self._get_current_lr(),
        }
        log_dict.update(weight_log)

        self.log_dict(
            log_dict,
            on_step  = True,
            on_epoch = True,
            prog_bar = True,
            sync_dist = True,
        )

        # ── PATCH DE INTEGRAÇÃO GRADNORM CORRIGIDO ────────────────────────────
        # NÃO use detach(). Mantenha as tensores originais no grafo computacional
        # para que o autograd.grad do GradNormCallback funcione.
        if self.balancer is not None and isinstance(self.balancer, GradNormBalancer):
            self._last_task_losses = [loss_bss, loss_doa, loss_heatmap]
        # ──────────────────────────────────────────────────────────────────────

        return loss_total

    # ─────────────────────────────────────────────────────────────────────────
    # Training Step Dispatcher
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(
        self,
        batch:     Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Dispatcher que roteia para o training_step correto baseado na fase atual.

        Args:
            batch     : batch do DataLoader (formato depende da fase)
            batch_idx : índice do batch no epoch

        Returns:
            loss: escalar a ser backpropagado pelo Lightning
        """
        if self.phase == TrainingPhase.JEPA:
            return self._training_step_jepa(batch, batch_idx)

        elif self.phase in (TrainingPhase.MULTITASK, TrainingPhase.LORA_FINETUNE):
            return self._training_step_multitask(batch, batch_idx)

        else:
            raise ValueError(f"Fase desconhecida: {self.phase}")

    # ─────────────────────────────────────────────────────────────────────────
    # Validation Step
    # ─────────────────────────────────────────────────────────────────────────

    def validation_step(
        self,
        batch:     Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Validação unificada para todas as fases.

        Fase 1: calcula apenas perda LeJEPA
        Fase 2/3: calcula todas as perdas + métricas de DoA em graus
        """
        # Garante a sincronia da fase ativa com a configuração de execução atual
        self.phase = self.cfg.phase

        features = batch["features"]   # [B, 5, T, F] pré-computado na CPU
        mask_ctx = batch.get("mask_ctx")
        mask_tgt = batch.get("mask_tgt")

        metrics_out = {}

        # ── Perda LeJEPA (sempre calculada para monitoramento de representação)
        if mask_ctx is not None:
            tokens_clean, pos_embed = self.model.patch_embedder(features, mask=None)
            with torch.no_grad():
                z_target = self.model.encoder.forward_target(tokens_clean)

            tokens_masked, _ = self.model.patch_embedder(features, mask=mask_ctx)
            z_ctx = self.model.encoder.forward_context(tokens_masked)

            pos_target = pos_embed * mask_tgt.float().unsqueeze(-1)
            z_pred = self.model.predictor(z_ctx, pos_embed_target=pos_target)

            loss_jepa, jepa_metrics = self.loss_fn_jepa(z_pred, z_target, mask=mask_tgt)

            metrics_out.update({
                "val/loss_jepa_total": loss_jepa,
                "val/loss_pred":       jepa_metrics.get("loss/pred", 0.0),
                "val/loss_var":        jepa_metrics.get("loss/var",  0.0),
                "val/loss_cov":        jepa_metrics.get("loss/cov",  0.0),
            })

        # ── Métricas Supervisionadas (Fase 2/3) ───────────────────────────────
        if self.phase in (TrainingPhase.MULTITASK, TrainingPhase.LORA_FINETUNE):
            outputs = self.model(features)

            if "doa" in batch:
                doa_pred  = outputs["doa"]    # [B, N, 2]
                doa_true  = batch["doa"]      # [B, N, 2]
                N = min(doa_pred.shape[1], doa_true.shape[1])

                doa_error = CircularMSELoss.angular_error_degrees(
                    doa_pred[:, :N], doa_true[:, :N]
                )  # [B, N]
                metrics_out["val/doa_mae_degrees"] = doa_error.mean()

                loss_doa_val = self.loss_fn_doa(doa_pred[:, :N], doa_true[:, :N])
                metrics_out["val/loss_doa"] = loss_doa_val

            if "sources" in batch and "stereo" in batch:
                loss_bss_val, si_sdrs = self.loss_fn_bss(
                    outputs["masks"], batch["stereo"], batch["sources"]
                )
                metrics_out["val/loss_bss"] = loss_bss_val
                metrics_out["val/si_sdr_mean"] = si_sdrs.mean()

            if "heatmap" in batch:
                heatmap_t_resized = F.interpolate(
                    batch["heatmap"].unsqueeze(1).float(),
                    size=outputs["heatmap"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                loss_heatmap_val = self.loss_fn_heatmap(
                    outputs["heatmap"].squeeze(1), heatmap_t_resized
                )
                metrics_out["val/loss_heatmap"] = loss_heatmap_val

        # Define a métrica principal de monitoramento dinamicamente por fase
        if self.phase == TrainingPhase.JEPA:
            primary_loss = metrics_out.get("val/loss_jepa_total", torch.tensor(0.0, device=self.device))
        elif self.phase in (TrainingPhase.MULTITASK, TrainingPhase.LORA_FINETUNE):
            # Recria o Total Loss com os mesmos pesos usados no treinamento
            w1 = self.cfg.multitask.initial_weight_bss
            w2 = self.cfg.multitask.initial_weight_doa
            w3 = self.cfg.multitask.initial_weight_heatmap
            
            l_bss = metrics_out.get("val/loss_bss", torch.tensor(0.0, device=self.device))
            l_doa = metrics_out.get("val/loss_doa", torch.tensor(0.0, device=self.device))
            l_heat = metrics_out.get("val/loss_heatmap", torch.tensor(0.0, device=self.device))
            
            primary_loss = (w1 * l_bss) + (w2 * l_doa) + (w3 * l_heat)
            metrics_out["val/loss_multitask_total"] = primary_loss
        else:
            primary_loss = torch.tensor(0.0, device=self.device)

        metrics_out["val/loss_total"] = primary_loss

        self.log_dict(
            metrics_out,
            on_step   = False,
            on_epoch  = True,
            prog_bar  = True,
            sync_dist = True,
        )

        return metrics_out

    # ─────────────────────────────────────────────────────────────────────────
    # MÓDULO 2: Configuração de Otimizador e Scheduler
    # ─────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> Dict:
        """
        Configura otimizador AdamW 8-bits (bitsandbytes) com scheduler Cosine Annealing + Warmup.
        """
        cfg = self.cfg.optimizer

        # ── Separação de grupos por peso de LR ───────────────────────────────
        param_groups = self._build_param_groups()

        # 🟢 CORREÇÃO: Uso de otimizador em 8-bits para economia massiva de VRAM
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr           = cfg.lr,
            betas        = cfg.betas,
            eps          = cfg.eps,
            weight_decay = cfg.weight_decay,
        )

        # ── Scheduler: Cosine Annealing com Linear Warmup ─────────────────────
        warmup_steps = int(self.total_steps * self.cfg.scheduler.warmup_ratio)
        min_lr_ratio = self.cfg.scheduler.min_lr_ratio

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(
                    max(1, self.total_steps - warmup_steps)
                )
                import math
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler":  scheduler,
                "interval":   "step",    
                "frequency":  1,
                "name":       f"lr_cosine_warmup_{self.phase.value}",
            },
            "monitor": self.cfg.monitor_metric,
        }

    def _build_param_groups(self) -> List[Dict]:
        """
        Constrói grupos de parâmetros com LR diferenciado por módulo.

        Parâmetros sem weight decay (seguindo ViT/BERT best practices):
          - Todas as bias
          - LayerNorm weight e bias
          - Positional embeddings
          - Mask tokens
        """
        cfg = self.cfg

        # Parâmetros que NÃO recebem weight decay
        no_decay_names = {"bias", "norm", "pos_embed", "mask_token"}

        def should_decay(name: str, param: torch.Tensor) -> bool:
            return not any(nd in name for nd in no_decay_names) and param.ndim >= 2

        # Fase LoRA: apenas parâmetros LoRA
        if self.phase == TrainingPhase.LORA_FINETUNE:
            lora_params = [
                p for n, p in self.model.named_parameters()
                if p.requires_grad and "lora_" in n
            ]
            logger.info(f"LoRA: {len(lora_params)} grupos de parâmetros ativos.")
            return [{"params": lora_params, "lr": cfg.lora.lr}]

        # Fase 2: LR reduzido ou Congelamento do tronco
        is_multitask = self.phase == TrainingPhase.MULTITASK
        freeze_backbone = is_multitask and getattr(cfg.multitask, "freeze_backbone", False)
        backbone_scale = cfg.multitask.backbone_lr_scale if is_multitask else 1.0

        backbone_modules = ["encoder", "patch_embedder", "predictor"]
        head_modules     = ["bss_head", "doa_head", "heatmap_head"]

        groups: List[Dict] = []

        for module_name in backbone_modules + head_modules:
            module = getattr(self.model, module_name, None)
            if module is None:
                continue

            is_backbone = module_name in backbone_modules

            # 🟢 O CHEAT CODE DE VELOCIDADE
            # Se for backbone e a flag de congelamento estiver ativa, desliga os gradientes.
            if is_backbone and freeze_backbone:
                for p in module.parameters():
                    p.requires_grad = False
                logger.info(f"❄️ Backbone congelado: {module_name} não será atualizado.")
                continue  # Pula este módulo inteiro (não vai pro otimizador)
            elif is_backbone:
                # Garante que descongela (caso venha congelado de outro canto)
                for p in module.parameters():
                    p.requires_grad = True

            lr_scale = backbone_scale if is_backbone else 1.0

            # Parâmetros com weight decay
            decay_params = [
                p for n, p in module.named_parameters()
                if p.requires_grad and should_decay(n, p)
            ]
            # Parâmetros sem weight decay
            no_decay_params = [
                p for n, p in module.named_parameters()
                if p.requires_grad and not should_decay(n, p)
            ]

            if decay_params:
                groups.append({
                    "params":       decay_params,
                    "lr":           cfg.optimizer.lr * lr_scale,
                    "weight_decay": cfg.optimizer.weight_decay,
                    "name":         f"{module_name}_decay",
                })
            if no_decay_params:
                groups.append({
                    "params":       no_decay_params,
                    "lr":           cfg.optimizer.lr * lr_scale,
                    "weight_decay": 0.0,
                    "name":         f"{module_name}_no_decay",
                })

        # Adiciona parâmetros do balanceador (Uncertainty Weighting)
        if self.balancer is not None and hasattr(self.balancer, "log_sigma_sq"):
            groups.append({
                "params":       [self.balancer.log_sigma_sq],
                "lr":           cfg.optimizer.lr * 0.1,   # lr menor para sigmas
                "weight_decay": 0.0,
                "name":         "uncertainty_weights",
            })

        logger.info(
            f"Grupos de parâmetros: {len(groups)} grupos, "
            f"escala backbone: {backbone_scale}"
        )
        return groups

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitários
    # ─────────────────────────────────────────────────────────────────────────

    def _get_current_lr(self) -> float:
        """Retorna LR atual do primeiro grupo de parâmetros."""
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return self.cfg.optimizer.lr
        if isinstance(schedulers, list):
            return schedulers[0].get_last_lr()[0]
        return schedulers.get_last_lr()[0]

    def _check_loss_health(self, loss: torch.Tensor, phase_name: str) -> torch.Tensor:
        """Detecta NaN/Inf na loss e faz skip do step se necessário."""
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"⚠️  Loss inválida detectada na fase '{phase_name}' "
                f"(step {self.global_step}). "
                f"Pulando step via zero loss."
            )
            # Retorna novo tensor zero com grad — evita mutação in-place no grafo
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss

    def _log_latent_stats(
        self,
        z_pred:   torch.Tensor,
        z_target: torch.Tensor,
        metrics:  Dict[str, float],
    ) -> None:
        """Loga estatísticas do espaço latente no WandB."""
        try:
            import wandb

            if not isinstance(self.trainer.logger, pl.loggers.WandbLogger):
                return

            # Estatísticas básicas de z_pred
            z_flat = z_pred.detach().float().reshape(-1, z_pred.shape[-1])
            std_per_dim = z_flat.std(dim=0)

            wandb.log({
                "latent/std_mean":    std_per_dim.mean().item(),
                "latent/std_min":     std_per_dim.min().item(),
                "latent/std_max":     std_per_dim.max().item(),
                "latent/dead_dims":   (std_per_dim < 0.01).sum().item(),
                # Histograma de ativações
                "latent/z_pred_hist": wandb.Histogram(
                    z_flat[:, :32].cpu().numpy()   # primeiras 32 dims
                ),
            }, step=self.global_step)
        except Exception as e:
            logger.debug(f"Erro no log de latent stats: {e}")

    def on_train_epoch_end(self) -> None:
        """Hook ao final de cada epoch de treino."""
        # Nota: grad_norm é logada via on_before_optimizer_step (antes do zero_grad)
        # Ao final do epoch os gradientes já foram zerados, então não há nada a logar.
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.phase == TrainingPhase.JEPA:
            self.model.update_target_encoder()

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Hook antes de cada step do otimizador — para logging de grad norm."""
        if self.global_step % self.cfg.log_every_n_steps == 0:
            try:
                norms = [
                    p.grad.detach().norm(2).item()
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
                if norms:
                    self.log(
                        "train/param_grad_norm_max",
                        max(norms),
                        prog_bar=False,
                    )
            except Exception:
                pass