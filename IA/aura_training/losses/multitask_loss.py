"""
losses/multitask_loss.py
========================
Balanceamento dinâmico de perdas multitarefa.

Problema: As três perdas têm ordens de magnitude muito diferentes:
  - L_bss     (SI-SDR): tipicamente ∈ [-30, 0] dB → gradientes GRANDES
  - L_doa     (Circ. MSE): tipicamente ∈ [0, 2] → gradientes MÉDIOS
  - L_heatmap (BCE): tipicamente ∈ [0, 1] → gradientes PEQUENOS

Sem balanceamento, L_bss domina completamente e as outras tarefas
não aprendem — especialmente L_doa que é mais delicada.

Dois algoritmos implementados:

┌─────────────────────────────────────────────────────────────────────┐
│ 1. Uncertainty Weighting (Kendall, Gal, Cipolla — NeurIPS 2018)    │
│                                                                     │
│    L = Σ_i  1/(2σ_i²) * L_i  +  log(σ_i)                         │
│                                                                     │
│    σ_i = desvio padrão da "observação" da tarefa i (aprendível)    │
│    Tarefas com alta incerteza → σ_i grande → peso 1/(2σ_i²) pequeno│
│                                                                     │
│    Vantagem: completamente diferenciável, otimizado com AdamW      │
│    Desvantagem: pode saturar com σ_i → ∞ para tarefas difíceis    │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│ 2. GradNorm (Chen, Badrinarayanan, Lee, Rabinovich — ICML 2018)    │
│                                                                     │
│    Objetivo: ||∇_W(w_i * L_i)|| ≈ G_mean * r_i^α                  │
│                                                                     │
│    G_mean = média das normas de gradiente entre tarefas            │
│    r_i    = relative training rate (quanto cada tarefa progrediu)  │
│    α      = assimetria (α>1 prioriza tarefas que progridem rápido) │
│                                                                     │
│    Vantagem: diretamente controla o balanceamento de gradientes    │
│    Desvantagem: requer backward parcial para calcular normas       │
└─────────────────────────────────────────────────────────────────────┘

Referências:
    Kendall et al. (2018) — Multi-Task Learning Using Uncertainty
    Chen et al. (2018) — GradNorm: Gradient Normalization for Adaptive Loss
    Liu et al. (2021)  — Conflict-Averse Gradient Descent
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Algoritmo 1: Uncertainty Weighting
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyWeighting(nn.Module):
    """
    Balanceamento por Incerteza Homoscedástica (Kendall et al. 2018).

    Aprende log(σ²) para cada tarefa como parâmetros do modelo.
    O otimizador principal (AdamW) atualiza esses parâmetros junto com
    os pesos do modelo — sem necessidade de loop de otimização separado.

    Derivação matemática:
        Assumindo likelihood Gaussiana para cada tarefa:
            p(y_i | f(x), σ_i) = N(f(x), σ_i²)

        Maximizar log-likelihood = Minimizar:
            L = Σ_i [ L_i / (2σ_i²) + log(σ_i) ]
              = Σ_i [ L_i * exp(-log_σ²_i) / 2 + log_σ²_i / 2 ]

        O termo log(σ_i) age como regularizador: impede que σ_i → ∞
        (que minimizaria L_i/(2σ_i²) trivialmente ao custo de alta incerteza)

    Args:
        n_tasks    : número de tarefas (3 = BSS, DoA, Heatmap)
        task_names : nomes das tarefas para logging
        init_log_sigma: valor inicial de log(σ²) por tarefa
                       0.0 = σ=1 → peso inicial = 0.5
                       Valores negativos → σ<1 → pesos maiores (tarefas importantes)
        clamp_log_sigma: limites para log(σ²) (evita degeneração)
    """

    def __init__(
        self,
        n_tasks:        int         = 3,
        task_names:     List[str]   = None,
        init_log_sigma: float       = 0.0,
        clamp_log_sigma: Tuple[float, float] = (-4.0, 4.0),
    ) -> None:
        super().__init__()

        self.n_tasks   = n_tasks
        self.task_names = (
            task_names if task_names is not None
            else [f"task_{i}" for i in range(n_tasks)]
        )
        self.clamp_range = clamp_log_sigma

        # log_sigma_sq: parâmetros aprendíveis [n_tasks]
        # Representamos log(σ²) = 2*log(σ) por estabilidade numérica
        self.log_sigma_sq = nn.Parameter(
            torch.full(
                (n_tasks,),
                init_log_sigma,
                dtype = torch.float32,
            )
        )

    def forward(
        self,
        losses: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combina N perdas com pesos de incerteza aprendíveis.

        Args:
            losses: Lista de escalares [L_bss, L_doa, L_heatmap]

        Returns:
            total_loss : escalar — perda combinada com regularização
            metrics    : dict com pesos efetivos e sigmas por tarefa

        Shapes intermediários:
            log_sigma_sq : [N_tasks]
            precision    : [N_tasks] = exp(-log_sigma_sq)
            weighted     : [N_tasks] = precision * L_i / 2 + log_sigma_sq / 2
            total_loss   : escalar = sum(weighted)
        """
        assert len(losses) == self.n_tasks, (
            f"Esperava {self.n_tasks} perdas, recebeu {len(losses)}"
        )

        # Clamp para evitar saturação numérica
        log_s2_clamped = self.log_sigma_sq.clamp(*self.clamp_range)

        total    = torch.zeros(1, device=self.log_sigma_sq.device)
        metrics  = {}

        for i, (loss_i, log_s2_i, name) in enumerate(
            zip(losses, log_s2_clamped, self.task_names)
        ):
            # Precisão = 1/σ² = exp(-log_σ²)
            precision_i = torch.exp(-log_s2_i)   # escalar

            # Perda ponderada: L_i/(2σ²) + log(σ) = L_i*prec/2 + log_σ²/2
            weighted_i = precision_i * loss_i * 0.5 + log_s2_i * 0.5
            total      = total + weighted_i

            # Métricas para logging
            sigma_i  = torch.exp(log_s2_i * 0.5).item()    # σ = exp(log_σ²/2)
            weight_i = (precision_i * 0.5).item()           # peso efetivo

            metrics[f"uw/sigma_{name}"]  = sigma_i
            metrics[f"uw/weight_{name}"] = weight_i
            metrics[f"uw/loss_{name}"]   = loss_i.item()
            metrics[f"uw/weighted_{name}"] = weighted_i.item()

        # Adiciona soma total e balanceamento de pesos
        metrics["uw/total_loss"]   = total.item()
        metrics["uw/weight_ratio"] = (
            max(metrics[f"uw/weight_{n}"] for n in self.task_names) /
            (min(metrics[f"uw/weight_{n}"] for n in self.task_names) + 1e-8)
        )

        return total.squeeze(), metrics

    @property
    def effective_weights(self) -> torch.Tensor:
        """
        Retorna pesos efetivos [n_tasks] = 1/(2σ²) para cada tarefa.
        Útil para inspeção e visualização do balanceamento atual.
        """
        return 0.5 * torch.exp(-self.log_sigma_sq.clamp(*self.clamp_range))

    def weight_summary(self) -> str:
        """Retorna string formatada com pesos atuais (para logging)."""
        weights = self.effective_weights.detach()
        parts   = [
            f"{name}: {w:.4f} (σ={torch.exp(s*0.5):.3f})"
            for name, w, s in zip(
                self.task_names, weights, self.log_sigma_sq
            )
        ]
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Algoritmo 2: GradNorm
# ─────────────────────────────────────────────────────────────────────────────

class GradNormBalancer(nn.Module):
    """
    Balanceamento GradNorm para equalização das normas de gradiente.

    Diferente do UncertaintyWeighting (que é um módulo nn.Module treinado
    pelo otimizador principal), o GradNorm mantém seus próprios pesos e
    os atualiza via um passo de gradiente separado a cada `update_freq` steps.

    Isso é necessário porque os pesos GradNorm dependem das normas de gradiente
    que só estão disponíveis APÓS o backward do loss total — um loop especial.

    Implementação simplificada:
        - Calcula norma de gradiente apenas na última camada compartilhada
          (em vez de todas as camadas do backbone — mais eficiente)
        - Usa learning rate fixo para os pesos (não adaptativo)

    Args:
        n_tasks      : número de tarefas
        task_names   : nomes das tarefas para logging
        alpha        : fator de assimetria (α > 1 → mais agressivo)
                       α = 0: todos gradientes igualados
                       α = 2: favorece tarefas que progridem mais rápido
        update_freq  : steps entre atualizações dos pesos GradNorm
        lr_weights   : learning rate para os pesos GradNorm
        min_weight   : peso mínimo para cada tarefa (evita tarefa ignorada)
        max_weight   : peso máximo para cada tarefa (evita dominância)
    """

    def __init__(
        self,
        n_tasks:    int        = 3,
        task_names: List[str]  = None,
        alpha:      float      = 1.5,
        update_freq: int       = 10,
        lr_weights:  float     = 1e-3,
        min_weight:  float     = 0.01,
        max_weight:  float     = 10.0,
    ) -> None:
        super().__init__()

        self.n_tasks     = n_tasks
        self.task_names  = (
            task_names if task_names is not None
            else [f"task_{i}" for i in range(n_tasks)]
        )
        self.alpha       = alpha
        self.update_freq = update_freq
        self.lr_weights  = lr_weights
        self.min_weight  = min_weight
        self.max_weight  = max_weight

        # Pesos GradNorm: atualizados manualmente (não via autograd do optimizer)
        self.weights = nn.Parameter(
            torch.ones(n_tasks, dtype=torch.float32),
            requires_grad=True,
        )
        self.register_buffer(
            "_L0",
            torch.zeros(n_tasks, dtype=torch.float32),   # perdas iniciais (L_i^0)
        )
        self.register_buffer(
            "_initialized",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "_step_count",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(
        self,
        losses: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combina perdas com pesos GradNorm atuais.

        NOTA: Este forward apenas aplica os pesos ATUAIS.
        A atualização dos pesos acontece em `update_weights()`,
        chamado separadamente após o backward.

        Args:
            losses: Lista de escalares

        Returns:
            total_loss : escalar
            metrics    : dict com pesos e perdas individuais
        """
        assert len(losses) == self.n_tasks

        total   = sum(w * l for w, l in zip(self.weights, losses))
        metrics = {}

        for i, (name, w, l) in enumerate(
            zip(self.task_names, self.weights, losses)
        ):
            metrics[f"gn/weight_{name}"]     = w.item()
            metrics[f"gn/loss_{name}"]        = l.item()
            metrics[f"gn/weighted_{name}"]    = (w * l).item()

        metrics["gn/weight_ratio"] = (
            self.weights.max() / (self.weights.min() + 1e-8)
        ).item()

        return total, metrics

    def update_weights(
        self,
        losses:        List[torch.Tensor],
        shared_params: List[nn.Parameter],
    ) -> Dict[str, float]:
        """
        Atualiza os pesos GradNorm via gradiente das normas.

        Deve ser chamado APÓS o backward da loss total e ANTES do
        `optimizer.step()`. Como os pesos são buffers (não nn.Parameter),
        o AdamW NÃO os atualiza — apenas este método o faz.

        Args:
            losses        : perdas individuais [L_bss, L_doa, L_heatmap]
            shared_params : parâmetros da última camada compartilhada
                           (ex: list(model.predictor.proj_out.parameters()))

        Returns:
            gn_metrics: dict com normas de gradiente por tarefa

        Complexidade:
            O(N_tasks * backward_cost) — N_tasks execuções de autograd
            Para minimizar overhead, use apenas a última camada compartilhada
        """
        self._step_count += 1

        # Executa apenas a cada update_freq steps
        if (self._step_count % self.update_freq).item() != 0:
            return {}

        # Inicializa L_i^0 na primeira chamada
        if not self._initialized.item():
            with torch.no_grad():
                for i, loss_i in enumerate(losses):
                    self._L0[i] = loss_i.detach().abs()
            self._initialized.fill_(True)
            logger.info(
                f"GradNorm inicializado. L0 = "
                f"{[f'{v:.4f}' for v in self._L0.tolist()]}"
            )
            return {}

        # ── Calcula normas de gradiente por tarefa ────────────────────────────
        grad_norms_per_task = []
        device = losses[0].device

        for i, (w_i, loss_i) in enumerate(zip(self.weights, losses)):
            try:
                # Gradiente de w_i * L_i em relação à última camada compartilhada
                grads = torch.autograd.grad(
                    outputs      = w_i * loss_i,
                    inputs       = shared_params,
                    retain_graph = True,          # necessário pois loss_i é reutilizada
                    allow_unused = True,          # alguns params podem não ter grad
                    create_graph = False,         # não precisamos grad de grads
                )

                # Norma L2 total dos gradientes desta tarefa
                # (ignora parâmetros sem gradiente via filtering)
                norm_sq = sum(
                    g.detach().norm(2) ** 2
                    for g in grads
                    if g is not None
                )
                grad_norm_i = torch.sqrt(norm_sq + 1e-8)
                grad_norms_per_task.append(grad_norm_i)

            except RuntimeError as e:
                logger.warning(
                    f"GradNorm: erro ao calcular gradiente da tarefa {i} "
                    f"({self.task_names[i]}): {e}. Usando norma=1."
                )
                grad_norms_per_task.append(torch.ones(1, device=device).squeeze())

        G_norms = torch.stack(grad_norms_per_task)   # [N_tasks]
        G_mean  = G_norms.mean().detach()            # escalar

        # ── Relative Training Rate r_i ────────────────────────────────────────
        # r_i = (L_i / L_i^0) / mean(L_j / L_j^0)
        # Mede quanto cada tarefa progrediu RELATIVAMENTE à inicialização
        current_L = torch.stack([l.detach().abs() for l in losses])   # [N]
        L0        = self._L0.clamp(min=1e-8)
        loss_ratio = current_L / L0                    # [N] — progresso absoluto
        r_i        = loss_ratio / loss_ratio.mean()    # [N] — progresso relativo

        # ── Targets de norma GradNorm ─────────────────────────────────────────
        # Tarefa com r_i alto (progrediu pouco) → target alto → peso maior
        target_norms = (G_mean * (r_i ** self.alpha)).detach()   # [N]

        # ── GradNorm Loss: L1 entre normas atuais e targets ───────────────────
        gn_loss = F.l1_loss(G_norms, target_norms, reduction="sum")

        # ── Gradiente da GradNorm Loss em relação aos pesos ──────────────────
        try:
            gn_grads = torch.autograd.grad(
                outputs      = gn_loss,
                inputs       = [self.weights],
                allow_unused = True,
            )[0]

            if gn_grads is not None:
                with torch.no_grad():
                    # Atualização dos pesos via gradient descent
                    self.weights -= self.lr_weights * gn_grads

                    # Clamp para [min_weight, max_weight]
                    self.weights.clamp_(self.min_weight, self.max_weight)

                    # Renormaliza: Σ w_i = N_tasks (mantém escala total)
                    self.weights.data *= (self.n_tasks / self.weights.sum())

        except RuntimeError as e:
            logger.warning(f"GradNorm: erro no update dos pesos: {e}")

        # Métricas de diagnóstico
        gn_metrics = {}
        for i, name in enumerate(self.task_names):
            gn_metrics[f"gn/grad_norm_{name}"]   = G_norms[i].item()
            gn_metrics[f"gn/target_norm_{name}"] = target_norms[i].item()
            gn_metrics[f"gn/r_{name}"]           = r_i[i].item()

        gn_metrics["gn/G_mean"]         = G_mean.item()
        gn_metrics["gn/gn_loss"]        = gn_loss.item()
        gn_metrics["gn/weight_sum"]     = self.weights.sum().item()

        return gn_metrics

    def reset(self) -> None:
        """Reinicia o estado interno (útil ao mudar de fase ou dataset)."""
        with torch.no_grad():
            self.weights.fill_(1.0)
            self._L0.zero_()
            self._initialized.fill_(False)
            self._step_count.zero_()
        logger.info("GradNormBalancer: estado reiniciado.")

