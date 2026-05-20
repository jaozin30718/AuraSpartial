"""
losses/jepa_loss.py
===================
Função de perda LeJEPA completa com regularização VICReg.

Implementa as três componentes que evitam colapso de representação
sem amostras negativas e sem EMA (Exponential Moving Average):

    L_total = γ * L_pred + α * L_var + β * L_cov

Componentes:
  ┌─────────────────────────────────────────────────────────────────┐
  │ L_pred : Erro preditivo entre z_pred e z_target                 │
  │          → Força o preditor a reconstruir representações alvo   │
  │                                                                 │
  │ L_var  : Hinge de variância mínima por dimensão                 │
  │          → Evita colapso dimensional (todas as dims = constante)│
  │                                                                 │
  │ L_cov  : Penaliza covariância off-diagonal                      │
  │          → Força decorrelação: cada dim captura info diferente   │
  └─────────────────────────────────────────────────────────────────┘

Referências:
    Bardes, Ponce, LeCun (2022) — VICReg
    Assran et al. (2023)        — I-JEPA
    Garrido et al. (2024)       — LeJEPA / V-JEPA

Estabilidade numérica garantida via:
  - Clamp antes de sqrt() para evitar sqrt(0) → NaN no backward
  - Epsilon em todas as divisões
  - Normalização da covariância por n_samples E n_dims separadamente
  - Detach seletivo do z_target para evitar colapso do encoder alvo
  - Checagem de NaN/Inf com fallback para zero loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuração da Loss
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeJEPALossConfig:
    """
    Configuração completa da LeJEPA Loss.

    Pesos (α, β, γ):
        γ = weight_pred : peso do erro preditivo
        α = weight_var  : peso da regularização de variância
        β = weight_cov  : peso da regularização de covariância

    Valores padrão seguem VICReg paper (α=25, β=25, γ=1).
    Para áudio espacial, valores menores de α/β podem ser necessários
    no início do treino para estabilidade.
    """

    # ── Pesos das componentes ─────────────────────────────────────────────────
    weight_pred: float = 1.0     # γ — peso L_pred
    weight_var:  float = 25.0    # α — peso L_var
    weight_cov:  float = 25.0    # β — peso L_cov

    # ── Limiar de variância (γ do hinge de variância) ─────────────────────────
    # L_var penaliza dimensões com std < variance_gamma
    # Valor 1.0 = força std ≥ 1.0 por dimensão (espaço isotrópico)
    variance_gamma: float = 1.0

    # ── Tipo de perda preditiva ───────────────────────────────────────────────
    # "smooth_l1" : Huber loss — robusto a outliers no início do treino
    # "l2"        : MSE — penaliza erros grandes fortemente
    # "cosine"    : distância cosseno — invariante à escala
    # "normalized_l2": L2 após L2-normalização dos vetores
    pred_loss_type: str = "smooth_l1"

    # Parâmetro beta do Smooth L1 (transição linear → quadrático)
    smooth_l1_beta: float = 1.0

    # ── Normalização antes do cálculo de covariância ──────────────────────────
    # Se True: z-score normaliza z antes de calcular covariância
    # Aumenta estabilidade mas reduz informação de escala
    normalize_before_cov: bool = False

    # ── Detach do z_target ────────────────────────────────────────────────────
    # True  = LeJEPA clássico: gradiente só em z_pred (mais estável)
    # False = gradiente flui por ambos os encoders (mais rico, mais instável)
    detach_target: bool = True

    # ── Epsilon numérico ──────────────────────────────────────────────────────
    eps: float = 1e-6

    # ── Warm-up dos pesos de regularização ───────────────────────────────────
    # Se True: α e β crescem de 0 até seu valor final durante warmup_steps
    # Evita instabilidade numérica no início do treino
    warmup_regularization: bool = True
    warmup_steps: int = 1000

    # ── Logging detalhado ─────────────────────────────────────────────────────
    # Se True: retorna métricas adicionais (std por dim, % dims colapsadas, etc.)
    detailed_metrics: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Implementações das Componentes Individuais
# ─────────────────────────────────────────────────────────────────────────────

def compute_pred_loss(
    z_pred:    torch.Tensor,    # [N, D] onde N = B*L_masked
    z_target:  torch.Tensor,    # [N, D]
    loss_type: str  = "smooth_l1",
    beta:      float = 1.0,
    eps:       float = 1e-6,
) -> torch.Tensor:
    """
    L_pred: Distância entre predição e representação alvo.

    Calcula o erro de reconstrução apenas nos patches mascarados,
    forçando o preditor Mamba a aprender a dinâmica temporal do espaço latente.

    Args:
        z_pred   : [N, D] — representações preditas (MambaPredictor output)
        z_target : [N, D] — representações alvo (E_tgt output, detachado)
        loss_type: tipo de função de distância
        beta     : parâmetro de suavidade para smooth_l1
        eps      : epsilon para estabilidade numérica

    Returns:
        loss: escalar ≥ 0

    Shapes intermediários documentados:
        - Para "l2":        MSE → escalar
        - Para "smooth_l1": Huber → escalar
        - Para "cosine":    1 - cos_sim → [N] → mean → escalar
        - Para "normalized_l2": normaliza → MSE → escalar
    """
    if loss_type == "l2":
        # MSE padrão: Σ(z_pred_i - z_tgt_i)² / (N * D)
        return F.mse_loss(z_pred, z_target, reduction="mean")

    elif loss_type == "smooth_l1":
        # Huber loss: quadrático para |e| < beta, linear para |e| >= beta
        # Mais robusto a outliers que MSE no início do treino
        return F.smooth_l1_loss(
            z_pred, z_target,
            reduction = "mean",
            beta      = beta,
        )

    elif loss_type == "cosine":
        # Distância cosseno: 1 - <z_pred, z_tgt> / (||z_pred|| * ||z_tgt||)
        # Invariante à escala — bom quando as normas variam muito
        # Resultado ∈ [0, 2]: 0 = idêntico, 2 = opostos
        cos_sim = F.cosine_similarity(
            z_pred, z_target, dim=-1, eps=eps
        )  # [N]
        return (1.0 - cos_sim).mean()

    elif loss_type == "normalized_l2":
        # L2 sobre vetores L2-normalizados
        # Equivalente a 2 * (1 - cosine_similarity) / 2
        # Mais estável numericamente que cosine direto
        z_pred_n   = F.normalize(z_pred,   p=2, dim=-1, eps=eps)
        z_target_n = F.normalize(z_target, p=2, dim=-1, eps=eps)
        return F.mse_loss(z_pred_n, z_target_n, reduction="mean")

    else:
        raise ValueError(
            f"pred_loss_type inválido: '{loss_type}'. "
            f"Opções: 'l2', 'smooth_l1', 'cosine', 'normalized_l2'"
        )


def compute_variance_loss(
    z:              torch.Tensor,    # [N, D]
    gamma:          float = 1.0,
    eps:            float = 1e-6,
    return_per_dim: bool  = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    L_var: Regularização de Variância Mínima (Hinge).

    Penaliza dimensões com desvio padrão abaixo de γ, prevenindo o colapso
    onde todas as representações convergem para um único ponto no espaço latente.

    Fórmula:
        L_var = (1/D) * Σ_j max(0, γ - std(z_j))
               = (1/D) * Σ_j max(0, γ - sqrt(Var(z_j) + ε))

    O ε dentro do sqrt garante que o gradiente não exploda quando Var → 0.
    Note: ε DENTRO do sqrt, não fora — crítico para estabilidade do backward.

    Args:
        z             : [N, D] — representações (N amostras, D dimensões)
        gamma         : limiar mínimo de std (default: 1.0)
        eps           : epsilon numérico DENTRO do sqrt
        return_per_dim: se True, retorna também o hinge por dimensão [D]

    Returns:
        l_var    : escalar — perda de variância média
        hinge_pd : [D] ou None — hinge por dimensão (para diagnóstico)

    Interpretação:
        l_var ≈ 0   : todas as dims têm std ≥ γ → espaço saudável
        l_var >> 0  : muitas dims colapsadas → representação degenerada
    """
    N, D = z.shape

    if N < 2:
        # Não é possível calcular variância com menos de 2 amostras
        logger.warning("compute_variance_loss: N < 2, retornando zero.")
        dummy = z.new_zeros(1).squeeze()
        return dummy, (z.new_zeros(D) if return_per_dim else None)

    # ── Centraliza z: subtrai média por dimensão ───────────────────────────
    # z_c: [N, D] — cada coluna tem média zero
    z_c = z - z.mean(dim=0, keepdim=True)

    # ── Variância por dimensão: [D] ────────────────────────────────────────
    # unbiased=True: divide por (N-1) — estimador não-viesado
    var_per_dim = z_c.var(dim=0, unbiased=True)   # [D] ≥ 0

    # ── Std com epsilon DENTRO do sqrt ────────────────────────────────────
    # IMPORTANTE: torch.sqrt(0) tem gradiente indefinido → NaN no backward
    # A solução é adicionar ε antes do sqrt, não depois
    std_per_dim = torch.sqrt(var_per_dim + eps)   # [D] > 0

    # ── Hinge: penaliza dims com std < γ ──────────────────────────────────
    # max(0, γ - std_j): se std_j ≥ γ, contribuição = 0
    #                    se std_j < γ, contribuição = γ - std_j > 0
    hinge_per_dim = F.relu(gamma - std_per_dim)  # [D] ≥ 0

    # Média sobre todas as dimensões
    l_var = hinge_per_dim.mean()   # escalar

    return l_var, (hinge_per_dim if return_per_dim else None)


def compute_covariance_loss(
    z:                    torch.Tensor,    # [N, D]
    eps:                  float = 1e-6,
    normalize_by_std:     bool  = True,
    return_cov_matrix:    bool  = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    L_cov: Regularização de Decorrelação (Penalidade Off-Diagonal).

    Penaliza correlações entre pares de dimensões, forçando isotropia geométrica.
    Isso garante que cada dimensão do espaço latente capture informação diferente,
    sem redundância entre features — critério essencial para representações úteis.

    Fórmula:
        C_ij = Cov(z_i, z_j) / (std(z_i) * std(z_j))   [correlação normalizada]
        L_cov = (1/D) * Σ_{i≠j} C_ij²

    A normalização por std converte covariância → correlação (∈ [-1, 1]),
    tornando a perda invariante à escala das dimensões.

    Complexidade: O(N*D²) — pode ser custoso para D grande.
    Para D=512 e N=1024: ~270M operações por step.

    Args:
        z                 : [N, D] — representações achatadas
        eps               : epsilon para divisão por std
        normalize_by_std  : se True, normaliza → correlação [-1,1]
                           se False, usa covariância bruta
        return_cov_matrix : se True, retorna a matriz de correlação [D, D]

    Returns:
        l_cov  : escalar — penalidade off-diagonal média
        cov_mat: [D, D] ou None — matriz de correlação (para visualização WandB)

    Notas de implementação:
        - Usamos matmul ao invés de torch.cov para controle do epsilon
        - A diagonal é zerada antes do sum (auto-correlação = 1, não penalizar)
        - Dividimos por D (não D²) para normalização consistente com VICReg
    """
    N, D = z.shape

    if N < 2:
        logger.warning("compute_covariance_loss: N < 2, retornando zero.")
        dummy = z.new_zeros(1).squeeze()
        return dummy, (z.new_zeros(D, D) if return_cov_matrix else None)

    # ── Centraliza z ──────────────────────────────────────────────────────────
    z_c = z - z.mean(dim=0, keepdim=True)   # [N, D]

    # ── Matriz de Covariância [D, D] ──────────────────────────────────────────
    # Cov = Z^T Z / (N-1)   onde Z está centralizado
    # Operação: [D, N] @ [N, D] = [D, D]
    cov = (z_c.T @ z_c) / float(N - 1)   # [D, D]

    if normalize_by_std:
        # ── Normaliza → Matriz de Correlação ──────────────────────────────────
        # Extrai variâncias da diagonal: [D]
        var_diag = cov.diagonal()                                    # [D]
        std_diag = torch.sqrt(var_diag.clamp(min=eps))              # [D]

        # Produto externo dos stds: [D, D]
        # std_outer[i,j] = std[i] * std[j]
        std_outer = std_diag.unsqueeze(0) * std_diag.unsqueeze(1)  # [D, D]

        # Correlação: Cov[i,j] / (std[i] * std[j]) ∈ [-1, 1]
        corr = cov / std_outer.clamp(min=eps)   # [D, D]
        matrix = corr
    else:
        matrix = cov

    # ── Máscara off-diagonal: [D, D] ─────────────────────────────────────────
    # Cria máscara booleana que é False na diagonal e True fora
    eye_mask      = torch.eye(D, dtype=torch.bool, device=z.device)
    off_diag_mask = ~eye_mask   # [D, D]

    # ── Penalidade: quadrado dos elementos off-diagonal ───────────────────────
    # Seleciona os D*(D-1) elementos fora da diagonal e eleva ao quadrado
    off_diag_values = matrix[off_diag_mask]    # [D*(D-1)]
    l_cov = off_diag_values.pow(2).mean()      # escalar

    return l_cov, (matrix.detach() if return_cov_matrix else None)


# ─────────────────────────────────────────────────────────────────────────────
# Classe Principal LeJEPALoss
# ─────────────────────────────────────────────────────────────────────────────

class LeJEPALoss(nn.Module):
    """
    Função de Perda LeJEPA Completa.

    Combina erro preditivo com regularização VICReg para treinamento
    auto-supervisionado sem amostras negativas e sem EMA.

    Design para estabilidade:
      1. Warm-up dos pesos de regularização (α e β crescem gradualmente)
      2. Checagem de NaN/Inf com logging de alerta
      3. Métricas detalhadas para monitoramento do espaço latente
      4. Suporte a detach seletivo do z_target

    Args:
        config: LeJEPALossConfig com todos os hiperparâmetros

    Usage:
        loss_fn = LeJEPALoss(config)
        loss, metrics = loss_fn(z_pred, z_target, mask, step=global_step)
        loss.backward()
    """

    def __init__(self, config: LeJEPALossConfig) -> None:
        super().__init__()
        self.cfg = config

        # Registra o step atual para warm-up (não é um parâmetro — não treina)
        self.register_buffer(
            "_step", torch.tensor(0, dtype=torch.long)
        )

    def _get_warmup_scale(self) -> float:
        """
        Calcula o fator de escala para warm-up dos pesos de regularização.

        Durante warmup_steps: fator cresce linearmente de 0 → 1
        Após warmup_steps: fator = 1 (pesos completos)

        Isso evita que L_var e L_cov dominam o início do treino antes
        do preditor ter chance de aprender algo significativo.
        """
        if not self.cfg.warmup_regularization:
            return 1.0

        step = self._step.item()
        if step >= self.cfg.warmup_steps:
            return 1.0

        # Warm-up linear: 0 → 1
        return float(step) / float(self.cfg.warmup_steps)

    def forward(
        self,
        z_pred:   torch.Tensor,
        z_target: torch.Tensor,
        mask:     Optional[torch.Tensor] = None,
        step:     Optional[int]          = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        cfg = self.cfg
        
        # 🟢 OTIMIZAÇÃO CRÍTICA: Desabilita o FP16 (16-mixed) APENAS na Loss.
        # Como o modelo ficou muito inteligente, a matriz de covariância passou do 
        # limite de 65504. Isso garante que a matemática pesada rode em 32-bits (Seguro).
        device_type = 'cuda' if z_pred.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            
            # Força tudo para float32 puro
            z_pred = z_pred.float()
            z_target = z_target.float()

            # ── Atualiza contador de step para warm-up ────────────────────────────
            if step is not None:
                self._step.fill_(step)
            else:
                self._step.add_(1)

            warmup_scale = self._get_warmup_scale()

            # ── Detach seletivo do z_target ───────────────────────────────────────
            if cfg.detach_target:
                z_target = z_target.detach()

            # ── Seleção dos tokens para L_pred ────────────────────────────────────
            if mask is not None:
                z_pred_masked   = z_pred[mask]    
                z_target_masked = z_target[mask]  

                if z_pred_masked.shape[0] == 0:
                    z_pred_masked   = rearrange(z_pred,   "b l d -> (b l) d")
                    z_target_masked = rearrange(z_target, "b l d -> (b l) d")
            else:
                z_pred_masked   = rearrange(z_pred,   "b l d -> (b l) d")
                z_target_masked = rearrange(z_target, "b l d -> (b l) d")

            z_flat = rearrange(z_pred, "b l d -> (b l) d")  # [B*L, D]

            # ── L_pred: Erro Preditivo ────────────────────────────────────────────
            l_pred = compute_pred_loss(
                z_pred    = z_pred_masked,
                z_target  = z_target_masked,
                loss_type = cfg.pred_loss_type,
                beta      = cfg.smooth_l1_beta,
                eps       = cfg.eps,
            )

            # ── L_var: Regularização de Variância ────────────────────────────────
            l_var, hinge_per_dim = compute_variance_loss(
                z              = z_flat,
                gamma          = cfg.variance_gamma,
                eps            = cfg.eps,
                return_per_dim = cfg.detailed_metrics,
            )

            # ── L_cov: Regularização de Covariância ───────────────────────────────
            l_cov, cov_matrix = compute_covariance_loss(
                z                 = z_flat,
                eps               = cfg.eps,
                normalize_by_std  = True,
                return_cov_matrix = cfg.detailed_metrics,
            )

            effective_weight_var = cfg.weight_var * warmup_scale
            effective_weight_cov = cfg.weight_cov * warmup_scale

            total_loss = (
                cfg.weight_pred        * l_pred +
                effective_weight_var   * l_var  +
                effective_weight_cov   * l_cov
            )

            # ── Verificação de Saúde Numérica ─────────────────────────────────────
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error(
                    f"Loss inválida detectada!\n"
                    f"  l_pred={l_pred.item():.6f}  "
                    f"l_var={l_var.item():.6f}  "
                    f"l_cov={l_cov.item():.6f}\n"
                    f"  z_pred: mean={z_flat.mean().item():.4f} "
                    f"std={z_flat.std().item():.4f}\n"
                    f"  Substituindo por zero loss para evitar crash."
                )
                total_loss = total_loss.new_zeros(1).squeeze().requires_grad_(True)

            # ── Empacota Métricas para Logging ─────────────────────────────────────
            metrics: Dict[str, float] = {
                "loss/total":  total_loss.item(),
                "loss/pred":   l_pred.item(),
                "loss/var":    l_var.item(),
                "loss/cov":    l_cov.item(),
                "weights/pred": cfg.weight_pred,
                "weights/var":  effective_weight_var,
                "weights/cov":  effective_weight_cov,
                "warmup/scale":       warmup_scale,
                "warmup/step":        self._step.item(),
                "warmup/completed":   float(warmup_scale >= 1.0),
                "latent/z_pred_mean":    z_flat.mean().item(),
                "latent/z_pred_std":     z_flat.std().item(),
                "latent/z_pred_norm":    z_flat.norm(dim=-1).mean().item(),
            }

            if cfg.detailed_metrics:
                with torch.no_grad():
                    std_per_dim = z_flat.std(dim=0)   
                    dead_dims   = (std_per_dim < 0.1).sum().item()

                    metrics.update({
                        "latent/std_min":  std_per_dim.min().item(),
                        "latent/std_max":  std_per_dim.max().item(),
                        "latent/std_mean": std_per_dim.mean().item(),
                        "latent/dead_dims_count":   dead_dims,
                        "latent/dead_dims_fraction": dead_dims / z_flat.shape[1],
                        "mask/n_masked_patches": (mask.sum().item() if mask is not None else z_flat.shape[0]),
                        "mask/mask_ratio": (mask.float().mean().item() if mask is not None else 1.0),
                    })

                    if hinge_per_dim is not None:
                        metrics.update({
                            "latent/hinge_max":  hinge_per_dim.max().item(),
                            "latent/hinge_mean": hinge_per_dim.mean().item(),
                            "latent/n_collapsed_dims": (hinge_per_dim > 0).sum().item(),
                        })

        return total_loss, metrics

    def get_cov_matrix_for_logging(
        self,
        z: torch.Tensor,   # [B, L, D]
    ) -> torch.Tensor:
        """
        Calcula e retorna a matriz de correlação [D, D] para logging WandB.

        Método separado para evitar overhead no loop de treino principal.
        Chamar apenas periodicamente (ex: a cada 100 steps).

        Args:
            z: [B, L, D] — representações latentes

        Returns:
            cov_matrix: [D, D] — matriz de correlação (detachada, em CPU)
        """
        z_flat = rearrange(z.detach().float(), "b l d -> (b l) d")
        _, cov_matrix = compute_covariance_loss(
            z_flat,
            return_cov_matrix = True,
        )
        return cov_matrix.cpu() if cov_matrix is not None else torch.eye(z.shape[-1])

    def extra_repr(self) -> str:
        """Representação legível para print(model)."""
        return (
            f"weight_pred={self.cfg.weight_pred}, "
            f"weight_var={self.cfg.weight_var}, "
            f"weight_cov={self.cfg.weight_cov}, "
            f"variance_gamma={self.cfg.variance_gamma}, "
            f"pred_loss_type={self.cfg.pred_loss_type!r}"
        )

