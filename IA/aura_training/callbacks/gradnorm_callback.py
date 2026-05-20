"""
callbacks/gradnorm_callback.py
===============================
Callback PyTorch Lightning para integração do GradNorm no loop de treino.

O GradNorm requer acesso aos gradientes APÓS o backward e ANTES do
optimizer.step() — um momento específico do loop de treino que o
Lightning expõe via hooks on_before_optimizer_step() e
on_after_backward().

Este callback:
  1. Intercepta o momento correto no loop de treino do Lightning
  2. Chama GradNormBalancer.update_weights() com os parâmetros certos
  3. Loga todas as métricas de balanceamento no WandB
  4. Provê monitoramento de saúde do balanceamento (alertas de desequilíbrio)

Integração com AuraLightningModule:
  O callback acessa pl_module.balancer (GradNormBalancer) e
  pl_module.model.predictor (última camada compartilhada) diretamente.

Usage:
    gradnorm_cb = GradNormCallback(
        shared_module_name = "predictor",
        update_freq        = 10,
        log_grad_norms     = True,
        imbalance_alert_ratio = 10.0,
    )
    trainer = Trainer(callbacks=[gradnorm_cb, ...])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from losses.multitask_loss import GradNormBalancer

logger = logging.getLogger(__name__)


class GradNormCallback(Callback):
    """
    Callback para atualização e logging do GradNorm no loop Lightning.

    Gerencia a interação entre o GradNormBalancer e o loop de treino,
    garantindo que:
      - Os pesos são atualizados no momento correto (após backward)
      - As métricas são logadas de forma eficiente (não todo step)
      - Alertas de desequilíbrio são emitidos quando necessário

    Args:
        shared_module_name    : nome do módulo cuja última camada é usada
                                para calcular normas de gradiente
                                (ex: "predictor" → model.predictor)
        update_freq           : steps entre atualizações GradNorm
                                (deve coincidir com GradNormBalancer.update_freq)
        log_freq              : steps entre logs de métricas GradNorm
        log_grad_norms        : se True, loga norma de gradiente por tarefa
        log_weight_histogram  : se True, loga histograma dos pesos no WandB
        imbalance_alert_ratio : alerta se max(w)/min(w) > este valor
                                (indica balanceamento patológico)
        use_last_layer_only   : se True, usa apenas a última camada Linear
                                do shared_module (mais eficiente)
                                se False, usa todos os parâmetros do módulo
    """

    def __init__(
        self,
        shared_module_name:    str   = "predictor",
        update_freq:           int   = 10,
        log_freq:              int   = 20,
        log_grad_norms:        bool  = True,
        log_weight_histogram:  bool  = True,
        imbalance_alert_ratio: float = 10.0,
        use_last_layer_only:   bool  = True,
    ) -> None:
        super().__init__()

        self.shared_module_name    = shared_module_name
        self.update_freq           = update_freq
        self.log_freq              = log_freq
        self.log_grad_norms        = log_grad_norms
        self.log_weight_histogram  = log_weight_histogram
        self.imbalance_alert_ratio = imbalance_alert_ratio
        self.use_last_layer_only   = use_last_layer_only

        # Estado interno do callback
        self._step_count:      int                         = 0
        self._last_gn_metrics: Dict[str, float]            = {}
        self._weight_history:  List[torch.Tensor]          = []
        self._shared_params:   Optional[List[nn.Parameter]] = None
        self._balancer:        Optional[GradNormBalancer]  = None

        # Acumuladores de métricas para média móvel
        self._metrics_buffer:  Dict[str, List[float]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

    def setup(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
        stage:     str,
    ) -> None:
        """
        Inicializa referências ao balanceador e parâmetros compartilhados.

        Chamado uma vez antes do treino começar.
        """
        if stage != "fit":
            return

        # Verifica se o módulo tem GradNormBalancer
        balancer = getattr(pl_module, "balancer", None)
        if not isinstance(balancer, GradNormBalancer):
            logger.warning(
                f"GradNormCallback: pl_module.balancer não é GradNormBalancer "
                f"(é {type(balancer).__name__}). Callback desativado."
            )
            self._balancer = None
            return

        self._balancer = balancer

        # Obtém parâmetros da camada compartilhada
        shared_module = getattr(pl_module.model, self.shared_module_name, None)
        if shared_module is None:
            raise AttributeError(
                f"GradNormCallback: módulo '{self.shared_module_name}' "
                f"não encontrado em pl_module.model. "
                f"Módulos disponíveis: {list(pl_module.model._modules.keys())}"
            )

        self._shared_params = self._extract_shared_params(shared_module)

        logger.info(
            f"GradNormCallback configurado:\n"
            f"  Módulo compartilhado: {self.shared_module_name}\n"
            f"  Parâmetros para grad norm: {len(self._shared_params)}\n"
            f"  Total parâmetros: "
            f"{sum(p.numel() for p in self._shared_params):,}\n"
            f"  Update freq: {self.update_freq} steps"
        )

    def _extract_shared_params(
        self,
        module: nn.Module,
    ) -> List[nn.Parameter]:
        """
        Extrai parâmetros do módulo compartilhado para cálculo de grad norm.

        Se use_last_layer_only=True: usa apenas a última camada Linear,
        que é computacionalmente eficiente e geralmente representativa.

        Se use_last_layer_only=False: usa todos os parâmetros do módulo,
        mais preciso mas mais custoso.

        Args:
            module: módulo compartilhado (ex: MambaPredictor)

        Returns:
            Lista de nn.Parameter para calcular normas de gradiente
        """
        if not self.use_last_layer_only:
            # Todos os parâmetros treináveis do módulo
            params = [
                p for p in module.parameters()
                if p.requires_grad
            ]
            logger.debug(
                f"GradNorm: usando todos os {len(params)} parâmetros "
                f"de '{self.shared_module_name}'"
            )
            return params

        # Encontra a última camada Linear do módulo
        last_linear = None
        for child in module.modules():
            if isinstance(child, nn.Linear) and child.weight.requires_grad:
                last_linear = child

        if last_linear is None:
            logger.warning(
                f"GradNorm: nenhuma camada Linear encontrada em "
                f"'{self.shared_module_name}'. Usando todos os parâmetros."
            )
            return [p for p in module.parameters() if p.requires_grad]

        params = [last_linear.weight]
        if last_linear.bias is not None and last_linear.bias.requires_grad:
            params.append(last_linear.bias)

        logger.debug(
            f"GradNorm: usando última Linear de '{self.shared_module_name}': "
            f"weight={last_linear.weight.shape}"
        )
        return params

    # ─────────────────────────────────────────────────────────────────────────
    # Hook Principal: on_before_optimizer_step
    # ─────────────────────────────────────────────────────────────────────────

    def on_before_optimizer_step(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Hook chamado APÓS backward() e ANTES de optimizer.step().

        Este é o momento ideal para GradNorm porque:
          ✅ Os gradientes já foram calculados (backward completo)
          ✅ O optimizer ainda não atualizou os pesos
          ✅ Podemos ler e modificar gradientes livremente

        IMPORTANTE: O Lightning com gradient accumulation chama este hook
        apenas quando o acúmulo está completo — comportamento correto.
        """
        if self._balancer is None or self._shared_params is None:
            return

        # Verifica se estamos na fase correta (Multitask)
        current_phase = getattr(pl_module, "phase", None)
        from config.training_config import TrainingPhase
        if current_phase != TrainingPhase.MULTITASK:
            return

        self._step_count += 1

        # Recupera as perdas individuais do step atual
        # (o training_step deve armazená-las em pl_module._last_task_losses)
        task_losses = getattr(pl_module, "_last_task_losses", None)

        if task_losses is None:
            # Apenas loga aviso uma vez para não spammar
            if self._step_count == 1:
                logger.warning(
                    "GradNormCallback: pl_module._last_task_losses não encontrado. "
                    "Certifique-se que o training_step armazena as perdas individuais "
                    "em self._last_task_losses = [loss_bss, loss_doa, loss_heatmap]"
                )
            return

        # ── Atualiza pesos GradNorm ───────────────────────────────────────────
        gn_metrics = self._balancer.update_weights(
            losses        = task_losses,
            shared_params = self._shared_params,
        )

        # Armazena métricas para logging
        self._last_gn_metrics = gn_metrics
        self._accumulate_metrics(gn_metrics)

        # ── Logging ──────────────────────────────────────────────────────────
        if self._step_count % self.log_freq == 0:
            self._log_metrics(trainer, pl_module)

        # ── Alerta de Desequilíbrio ───────────────────────────────────────────
        self._check_imbalance(trainer, pl_module)

    # ─────────────────────────────────────────────────────────────────────────
    # Hook: on_train_epoch_end (logging de histograma)
    # ─────────────────────────────────────────────────────────────────────────

    @rank_zero_only
    def on_train_epoch_end(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Loga histograma dos pesos GradNorm ao final de cada epoch."""
        if self._balancer is None:
            return

        if not self.log_weight_histogram:
            return

        # Salva histórico de pesos para análise de convergência
        weights_snapshot = self._balancer.weights.detach().cpu().clone()
        self._weight_history.append(weights_snapshot)

        # Loga no WandB
        if trainer.logger is not None:
            self._log_weight_histogram(trainer, pl_module)

        # Log textual de resumo dos pesos por epoch
        weights = self._balancer.weights.tolist()
        task_names = self._balancer.task_names
        weight_str = " | ".join(
            f"{name}={w:.4f}"
            for name, w in zip(task_names, weights)
        )
        logger.info(
            f"[Epoch {trainer.current_epoch}] GradNorm weights: {weight_str}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Hook: on_train_end
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_end(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Logging final do histórico de pesos GradNorm."""
        if not self._weight_history:
            return

        # Empilha histórico [n_epochs, n_tasks]
        history = torch.stack(self._weight_history, dim=0)

        logger.info(
            f"\nGradNorm — Evolução dos pesos ao longo do treino:\n"
            f"{'Epoch':<8}" + "".join(f"{n:>12}" for n in self._balancer.task_names)
        )
        for epoch, w in enumerate(history):
            row = f"{epoch:<8}" + "".join(f"{v.item():>12.4f}" for v in w)
            logger.info(row)

    # ─────────────────────────────────────────────────────────────────────────
    # Métodos Auxiliares
    # ─────────────────────────────────────────────────────────────────────────

    def _accumulate_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Acumula métricas no buffer para calcular médias entre logs.

        Reduz overhead de logging ao fazer média de `log_freq` steps
        em vez de logar cada valor individual.
        """
        for key, value in metrics.items():
            if key not in self._metrics_buffer:
                self._metrics_buffer[key] = []
            self._metrics_buffer[key].append(value)

    @rank_zero_only
    def _log_metrics(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Loga métricas GradNorm acumuladas no logger do Lightning.

        Calcula médias dos últimos `log_freq` steps para suavizar o log.
        """
        if not self._metrics_buffer:
            return

        # Calcula médias
        avg_metrics = {
            f"gradnorm/{key}": sum(vals) / len(vals)
            for key, vals in self._metrics_buffer.items()
        }

        # Log via Lightning (que repassa ao WandB)
        pl_module.log_dict(
            avg_metrics,
            on_step  = True,
            on_epoch = False,
            prog_bar = False,
            sync_dist = True,
        )

        # Log das normas de gradiente individuais se habilitado
        if self.log_grad_norms and self._last_gn_metrics:
            grad_norm_metrics = {
                k: v for k, v in self._last_gn_metrics.items()
                if "grad_norm" in k
            }
            if grad_norm_metrics:
                pl_module.log_dict(
                    {f"gradnorm/{k}": v for k, v in grad_norm_metrics.items()},
                    on_step  = True,
                    on_epoch = False,
                    prog_bar = False,
                )

        # Limpa buffer após logging
        self._metrics_buffer.clear()

    @rank_zero_only
    def _log_weight_histogram(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Loga histograma dos pesos GradNorm no WandB."""
        try:
            import wandb
            if not isinstance(trainer.logger, pl.loggers.WandbLogger):
                return

            weights = self._balancer.weights.detach().cpu()
            task_names = self._balancer.task_names

            # Gráfico de barras dos pesos atuais
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Barras de pesos atuais
            colors = plt.cm.tab10(range(len(task_names)))
            bars   = axes[0].bar(task_names, weights.tolist(), color=colors)
            axes[0].set_title(
                f"GradNorm Weights — Epoch {trainer.current_epoch}"
            )
            axes[0].set_ylabel("Weight")
            axes[0].set_ylim(bottom=0)

            # Adiciona valor numérico acima de cada barra
            for bar, w in zip(bars, weights.tolist()):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{w:.3f}",
                    ha="center", va="bottom", fontsize=10
                )

            # Histórico de pesos ao longo do treino
            if len(self._weight_history) > 1:
                history = torch.stack(self._weight_history, dim=0).numpy()  # [E, N]
                for i, name in enumerate(task_names):
                    axes[1].plot(history[:, i], label=name, color=colors[i])
                axes[1].set_title("GradNorm Weight History")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Weight")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            wandb.log(
                {"gradnorm/weight_history": wandb.Image(fig)},
                step=trainer.global_step,
            )
            plt.close(fig)

        except Exception as e:
            logger.debug(f"Erro no log de histograma GradNorm: {e}")

    def _check_imbalance(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Verifica e alerta sobre desequilíbrios extremos nos pesos GradNorm.

        Um ratio max/min muito alto indica que uma tarefa está sendo
        praticamente ignorada, o que pode sinalizar:
          1. Alpha muito alto (dominância das tarefas que progridem rápido)
          2. Uma tarefa com perda inicial muito discrepante das outras
          3. Bug no cálculo de gradientes de uma tarefa específica
        """
        if self._balancer is None:
            return

        weights = self._balancer.weights
        w_max   = weights.max().item()
        w_min   = weights.min().item()
        ratio   = w_max / (w_min + 1e-8)

        if ratio > self.imbalance_alert_ratio:
            # Identifica a tarefa dominante e a negligenciada
            dominant_idx   = weights.argmax().item()
            neglected_idx  = weights.argmin().item()
            task_names     = self._balancer.task_names

            logger.warning(
                f"\n⚠️  GradNorm: Desequilíbrio detectado no step {self._step_count}!\n"
                f"   Ratio max/min = {ratio:.2f} (limiar: {self.imbalance_alert_ratio})\n"
                f"   Dominante : {task_names[dominant_idx]} "
                f"(w={w_max:.4f})\n"
                f"   Negligenciada: {task_names[neglected_idx]} "
                f"(w={w_min:.4f})\n"
                f"   Sugestões:\n"
                f"     - Reduza alpha (atual: {self._balancer.alpha})\n"
                f"     - Verifique se loss_{task_names[neglected_idx]} "
                f"está calculando corretamente\n"
                f"     - Considere usar UncertaintyWeighting em vez de GradNorm"
            )

            # Loga o alerta como métrica
            if trainer.logger is not None:
                try:
                    pl_module.log(
                        "gradnorm/imbalance_ratio",
                        ratio,
                        on_step=True,
                        prog_bar=True,
                    )
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Callback Auxiliar: GradientHealthMonitor
# ─────────────────────────────────────────────────────────────────────────────

class GradientHealthMonitor(Callback):
    """
    Callback complementar que monitora a saúde geral dos gradientes.

    Detecta problemas comuns como:
      - Gradient explosion: ||g|| > threshold
      - Gradient vanishing: ||g|| < 1e-7 por muitos steps consecutivos
      - NaN/Inf nos gradientes
      - Gradientes mortos (parâmetros que não recebem gradiente)

    Útil em conjunto com GradNormCallback para diagnóstico completo.

    Args:
        explosion_threshold  : ||g|| acima deste valor = explosão
        vanishing_threshold  : ||g|| abaixo deste valor = vanishing
        vanishing_patience   : steps consecutivos com vanishing para alertar
        check_freq           : frequência de verificação (em steps)
        module_names_to_check: módulos específicos para monitorar (None = todos)
    """

    def __init__(
        self,
        explosion_threshold:   float      = 100.0,
        vanishing_threshold:   float      = 1e-7,
        vanishing_patience:    int        = 50,
        check_freq:            int        = 25,
        module_names_to_check: List[str]  = None,
    ) -> None:
        super().__init__()
        self.explosion_thresh = explosion_threshold
        self.vanishing_thresh = vanishing_threshold
        self.vanishing_pat    = vanishing_patience
        self.check_freq       = check_freq
        self.module_names     = module_names_to_check

        # Estado
        self._vanishing_counter: int = 0
        self._step:              int = 0

    def on_before_optimizer_step(
        self,
        trainer:   pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._step += 1

        if self._step % self.check_freq != 0:
            return

        # Coleta gradientes dos módulos monitorados
        modules_to_check = {}
        if self.module_names:
            for name in self.module_names:
                mod = getattr(pl_module.model, name, None)
                if mod is not None:
                    modules_to_check[name] = mod
        else:
            modules_to_check = dict(pl_module.model.named_children())

        for module_name, module in modules_to_check.items():
            grad_norms = []
            nan_count  = 0
            zero_count = 0

            for param_name, param in module.named_parameters():
                if param.grad is None:
                    zero_count += 1
                    continue

                grad = param.grad.detach()

                # Verifica NaN/Inf
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    nan_count += 1
                    logger.error(
                        f"⛔ NaN/Inf no gradiente! "
                        f"Módulo: {module_name}, Param: {param_name}"
                    )
                    continue

                norm = grad.norm(2).item()
                grad_norms.append(norm)

            if not grad_norms:
                continue

            total_norm = sum(n ** 2 for n in grad_norms) ** 0.5
            mean_norm  = sum(grad_norms) / len(grad_norms)

            # Loga normas
            pl_module.log_dict({
                f"grad_health/{module_name}/total_norm": total_norm,
                f"grad_health/{module_name}/mean_norm":  mean_norm,
                f"grad_health/{module_name}/nan_count":  float(nan_count),
                f"grad_health/{module_name}/zero_params": float(zero_count),
            }, on_step=True, prog_bar=False)

            # Detecta explosão
            if total_norm > self.explosion_thresh:
                logger.warning(
                    f"💥 Gradient Explosion detectado! "
                    f"Módulo: {module_name} | "
                    f"||g||={total_norm:.2f} > {self.explosion_thresh}"
                )

            # Detecta vanishing
            if mean_norm < self.vanishing_thresh:
                self._vanishing_counter += 1
                if self._vanishing_counter >= self.vanishing_pat:
                    logger.warning(
                        f"🌊 Gradient Vanishing detectado! "
                        f"Módulo: {module_name} | "
                        f"||g||_mean={mean_norm:.2e} por "
                        f"{self._vanishing_counter} verificações consecutivas"
                    )
            else:
                self._vanishing_counter = 0

