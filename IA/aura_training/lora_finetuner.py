"""
lora_finetuner.py
=================
MÓDULO 4: Fine-tuning de Borda com LoRA (PEFT)

Adapta o modelo AuraSpatial para uma nova sala/ambiente acústico específico
usando Low-Rank Adaptation (LoRA) nas camadas lineares dos blocos Mamba.

Estratégia:
  1. Carrega checkpoint da Fase 2 (modelo completo treinado)
  2. Congela ABSOLUTAMENTE todos os parâmetros
  3. Injeta matrizes LoRA (rank r) nas projeções do Mamba
  4. Treina apenas os ~0.1-1% de parâmetros LoRA com dados da nova sala
  5. Salva adapter LoRA separadamente (< 10MB vs ~140MB do modelo completo)

Vantagens para deployment em borda:
  - Adaptação em minutos vs horas
  - Armazena apenas o diff (adapter LoRA)
  - Roda em hardware embarcado durante deployment
  - Reversível: remove adapter para voltar ao modelo base

Referências:
  Hu et al. (2021) — LoRA: Low-Rank Adaptation of Large Language Models
  HuggingFace PEFT — https://github.com/huggingface/peft
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import math
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

from config.training_config import TrainingConfig, TrainingPhase, LoRAConfig
from lightning_module import AuraLightningModule

logger = logging.getLogger(__name__)

# Import PEFT com fallback informativo
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.error(
        "PEFT não disponível! Instale com: pip install peft\n"
        "LoRA fine-tuning requer peft >= 0.6.0"
    )


def inject_lora_into_mamba(
    model:       torch.nn.Module,
    lora_config: LoRAConfig,
) -> torch.nn.Module:
    """
    Injeta adapters LoRA nas camadas lineares dos blocos Mamba.

    Identifica automaticamente as camadas alvo dentro dos blocos
    MambaLayer aninhados e aplica a configuração LoRA via PEFT.

    Args:
        model       : AuraSpatialModel (antes do LoRA)
        lora_config : LoRAConfig com parâmetros rank, alpha, etc.

    Returns:
        model modificado com adapters LoRA injetados
        Apenas os parâmetros LoRA requerem gradiente.

    Estrutura dos módulos alvo no MambaPredictor:
        model.predictor.mamba_layers[i].mamba.{in_proj, x_proj, out_proj, dt_proj}
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT não instalado. Execute: pip install peft>=0.6.0"
        )

    # ── Congela TODOS os parâmetros primeiro ──────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        f"Parâmetros congelados: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ── Injeta LoRA manualmente nos blocos Mamba ─────────────────────────────
    # O PEFT não suporta nativamente mamba_ssm, então aplicamos manualmente
    # usando a API de LoRALinear
    lora_params_total = 0
    n_adapters        = 0

    for layer_idx, mamba_layer in enumerate(model.predictor.mamba_layers):
        # 🟢 CORREÇÃO: Agora o Mamba é bidirecional, injetamos nas duas redes
        sub_modules = [mamba_layer.mamba_fwd, mamba_layer.mamba_bwd]

        for mamba_block in sub_modules:
            for target_module_name in lora_config.target_modules:
                if hasattr(mamba_block, target_module_name):
                    original_linear = getattr(mamba_block, target_module_name)

                    if not isinstance(original_linear, torch.nn.Linear):
                        logger.debug(
                            f"  Layer {layer_idx}.{target_module_name}: "
                            f"não é Linear, pulando."
                        )
                        continue

                    # Cria e substitui pela LoRALinear
                    lora_linear = LoRALinear(
                        linear     = original_linear,
                        r          = lora_config.r,
                        lora_alpha = lora_config.lora_alpha,
                        dropout    = lora_config.lora_dropout,
                    )
                    setattr(mamba_block, target_module_name, lora_linear)

                    # Conta parâmetros LoRA
                    lora_params = (
                        lora_linear.lora_A.numel() +
                        lora_linear.lora_B.numel()
                    )
                    lora_params_total += lora_params
                    n_adapters        += 1

                    logger.debug(
                        f"  LoRA injetado: layer[{layer_idx}].{target_module_name} "
                        f"({lora_params:,} params)"
                    )

    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"\nLoRA injection completa:\n"
        f"  Adapters injetados : {n_adapters}\n"
        f"  Parâmetros LoRA    : {lora_params_total:,}\n"
        f"  Total parâmetros   : {total_params:,}\n"
        f"  % treinável        : {100 * trainable / (total_params + 1e-8):.3f}%\n"
    )

    return model


class LoRALinear(torch.nn.Module):
    """
    Camada Linear com adapter LoRA.

    Computa: y = W_0 * x + (B * A) * x * (alpha/r)
    onde:
      W_0 : pesos originais CONGELADOS
      A   : matriz de projeção low-rank [r, in_features]  (inicializada com gaussiana)
      B   : matriz de expansão low-rank [out_features, r] (inicializada com zeros)

    A inicialização B=0 garante que no início do treino,
    o adapter não perturba o modelo pré-treinado.

    Args:
        linear     : nn.Linear original (seus pesos serão congelados)
        r          : rank das matrizes LoRA
        lora_alpha : fator de escala (scaling = alpha/r)
        dropout    : dropout aplicado à ativação LoRA
    """

    def __init__(
        self,
        linear:     torch.nn.Linear,
        r:          int   = 8,
        lora_alpha: int   = 16,
        dropout:    float = 0.05,
    ) -> None:
        super().__init__()

        in_features  = linear.in_features
        out_features = linear.out_features

        # Pesos originais (CONGELADOS — não requerem grad)
        self.weight = linear.weight   # referência, não cópia
        self.bias   = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Matrizes LoRA (TREINÁVEIS)
        self.lora_A = torch.nn.Parameter(
            torch.empty(r, in_features)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(out_features, r)   # B=0: sem perturbação inicial
        )

        # Inicialização de A: gaussiana (como no paper LoRA)
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Escala: alpha/r (controla magnitude da perturbação)
        self.scaling = lora_alpha / r
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity()

        # Metadados
        self.r          = r
        self.lora_alpha = lora_alpha
        self.in_features  = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_features]

        Returns:
            y: [..., out_features]
               = W_0 @ x + B @ A @ dropout(x) * scaling
        """
        # Caminho original (pesos congelados)
        y0 = torch.nn.functional.linear(x, self.weight, self.bias)

        # Caminho LoRA (apenas parâmetros A e B requerem grad)
        x_drop  = self.dropout(x)
        lora_out = (x_drop @ self.lora_A.T) @ self.lora_B.T
        lora_out = lora_out * self.scaling

        return y0 + lora_out   # [..., out_features]


class LoRAFinetuner:
    """
    Orquestrador do fine-tuning LoRA para adaptação acústica local.

    Usage:
    ------
    >>> finetuner = LoRAFinetuner(base_checkpoint_path, lora_data_dir, config)
    >>> finetuner.run()
    >>> finetuner.save_adapter("./adapter_sala_especifica.pt")
    """

    def __init__(
        self,
        base_checkpoint:  str,          # path do checkpoint Fase 2
        finetune_data_dir: str,         # dados da nova sala/ambiente
        config:           TrainingConfig,
        output_dir:       str = "./lora_adapters",
        max_epochs:       int = 10,
        wandb_run_name:   str = "lora-finetune",
    ) -> None:
        self.base_ckpt     = base_checkpoint
        self.data_dir      = finetune_data_dir
        self.config        = config
        self.output_dir    = Path(output_dir)
        self.max_epochs    = max_epochs
        self.wandb_name    = wandb_run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> AuraLightningModule:
        """
        Executa o pipeline completo de fine-tuning LoRA.

        Returns:
            pl_module: módulo Lightning com LoRA treinado
        """
        logger.info(
            f"\n{'='*60}\n"
            f"  Iniciando LoRA Fine-tuning\n"
            f"  Checkpoint base  : {self.base_ckpt}\n"
            f"  Dados de finetune: {self.data_dir}\n"
            f"  Output           : {self.output_dir}\n"
            f"{'='*60}"
        )

        # ── 1. Carrega modelo da Fase 2 ───────────────────────────────────────
        pl_module = AuraLightningModule.load_from_checkpoint(
            self.base_ckpt,
            strict = True,
            map_location = "cpu",
        )

        # ── 2. Injeta LoRA ────────────────────────────────────────────────────
        pl_module.model = inject_lora_into_mamba(
            pl_module.model,
            self.config.lora,
        )

        # ── 3. Atualiza fase ──────────────────────────────────────────────────
        pl_module.phase = TrainingPhase.LORA_FINETUNE

        # ── 4. DataModule específico para nova sala ───────────────────────────
        from data.datamodule import AuraDataModule
        from config.training_config import TrainingConfig

        finetune_config = TrainingConfig(
            phase           = TrainingPhase.LORA_FINETUNE,
            train_hdf5_dir  = self.data_dir,
            val_hdf5_dir    = self.data_dir,
            batch_size      = 8,              # batch menor para finetune rápido
            num_workers     = 4,
        )
        dm = AuraDataModule(finetune_config)

        # ── 5. Callbacks ──────────────────────────────────────────────────────
        callbacks = [
            ModelCheckpoint(
                dirpath    = str(self.output_dir),
                filename   = "lora_adapter_{epoch:02d}_{val/loss_total:.4f}",
                monitor    = "val/loss_total",
                mode       = "min",
                save_top_k = 1,
            ),
            EarlyStopping(
                monitor  = "val/loss_total",
                patience = 3,
                mode     = "min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

        # ── 6. Logger ──────────────────────────────────────────────────────────
        wandb_logger = WandbLogger(
            project  = self.config.wandb_project,
            name     = self.wandb_name,
            tags     = ["lora", "finetune", "edge"],
        )

        # ── 7. Trainer configurado para fine-tuning rápido ───────────────────
        trainer = Trainer(
            max_epochs               = self.max_epochs,
            precision                = self.config.precision,
            accumulate_grad_batches  = 2,    # menor que treino completo
            gradient_clip_val        = 0.5,  # clipping mais agressivo para finetune
            callbacks                = callbacks,
            logger                   = wandb_logger,
            devices                  = 1,    # single GPU para finetune de borda
            log_every_n_steps        = 10,
            enable_progress_bar      = True,
        )

        # ── 8. Treina ────────────────────────────────────────────────────────
        trainer.fit(pl_module, datamodule=dm)

        logger.info("✅ LoRA fine-tuning concluído!")
        return pl_module

    def save_adapter(
        self,
        pl_module:   AuraLightningModule,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Salva apenas os parâmetros LoRA (adapter leve < 10MB).

        Args:
            pl_module   : módulo Lightning com LoRA treinado
            output_path : caminho de saída (None = output_dir/lora_adapter.pt)

        Returns:
            path: caminho do arquivo salvo
        """
        if output_path is None:
            output_path = str(self.output_dir / "lora_adapter_final.pt")

        # Extrai apenas parâmetros LoRA (lora_A e lora_B)
        lora_state = {
            name: param
            for name, param in pl_module.model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        }

        # Metadados do adapter
        adapter_package = {
            "lora_state_dict":  lora_state,
            "lora_config":      self.config.lora,
            "n_params":         sum(p.numel() for p in lora_state.values()),
            "base_checkpoint":  self.base_ckpt,
        }

        torch.save(adapter_package, output_path)

        size_mb = Path(output_path).stat().st_size / (1024 ** 2)
        logger.info(
            f"✅ Adapter LoRA salvo em: {output_path}\n"
            f"   Parâmetros LoRA : {adapter_package['n_params']:,}\n"
            f"   Tamanho do arquivo: {size_mb:.2f} MB"
        )
        return output_path

    @staticmethod
    def load_adapter(
        base_model:   AuraLightningModule,
        adapter_path: str,
        lora_config:  LoRAConfig,
    ) -> AuraLightningModule:
        """
        Carrega e aplica adapter LoRA em modelo base para inferência.

        Args:
            base_model   : modelo base (sem LoRA)
            adapter_path : path do arquivo .pt do adapter
            lora_config  : configuração LoRA (deve coincidir com treino)

        Returns:
            modelo com adapter aplicado (pronto para inferência)
        """
        adapter_package = torch.load(adapter_path, map_location="cpu")

        # Injeta arquitetura LoRA
        base_model.model = inject_lora_into_mamba(
            base_model.model, lora_config
        )

        # Carrega pesos
        incompatible = base_model.model.load_state_dict(
            adapter_package["lora_state_dict"],
            strict = False,   # apenas LoRA params
        )
        logger.info(
            f"Adapter LoRA carregado de {adapter_path}\n"
            f"Params carregados: {adapter_package['n_params']:,}\n"
            f"Incompatíveis: {len(incompatible.missing_keys)} missing, "
            f"{len(incompatible.unexpected_keys)} unexpected"
        )
        return base_model

