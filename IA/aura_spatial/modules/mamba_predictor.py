"""
modules/mamba_predictor.py
==========================
MÓDULO 3: Núcleo Preditivo Mamba (SSM)

Substitui a atenção quadrática O(L²) do Transformer por
State Space Models (SSM) O(L) do Mamba, capturando dependências
temporais de longo prazo de forma eficiente em hardware.

Fluxo do preditor:
    z_ctx [B, L, D]
      → proj_in    → [B, L, D']          (redução de dimensão)
      → N × Mamba  → [B, L, D']          (dinâmica SSM temporal)
      → proj_out   → [B, L, D]           (projeção de volta)
      → z_pred     → [B, L, D]           (target das representações E_tgt)

O estado oculto h_t do Mamba captura:
  - Padrões TDoA variantes no tempo
  - Dinâmicas de reverberação
  - Movimentos de fonte (Doppler)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from ..config import MambaPredictorConfig

# ── Import condicional do mamba-ssm ──────────────────────────────────────────
# Tenta importar mamba_ssm; se não disponível, usa implementação fallback
# para permitir desenvolvimento em CPU sem CUDA.
try:
    from mamba_ssm import Mamba as MambaBlock
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    import warnings
    warnings.warn(
        "mamba_ssm não encontrado. Usando SSM simplificado como fallback.\n"
        "Para performance máxima, instale: pip install mamba-ssm",
        ImportWarning,
        stacklevel=2,
    )


class SimplifiedSSM(nn.Module):
    """
    Implementação SSM simplificada para fallback CPU/teste.

    Aproxima o comportamento do Mamba usando LSTM + projeção,
    mantendo a mesma interface de input/output.

    ⚠️ Apenas para desenvolvimento/teste. Use mamba_ssm em produção.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = d_inner,
            batch_first = True,
        )
        self.proj_out = nn.Linear(d_inner, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args/Returns: [B, L, D] → [B, L, D]
        """
        residual = x
        out, _   = self.lstm(x)           # [B, L, d_inner]
        out      = self.proj_out(out)      # [B, L, D]
        return self.norm(out + residual)   # [B, L, D]


def _build_mamba_block(config: MambaPredictorConfig) -> nn.Module:
    """
    Factory que instancia o bloco Mamba correto baseado na disponibilidade.

    Args:
        config: MambaPredictorConfig

    Returns:
        Módulo com interface [B, L, D] → [B, L, D]
    """
    if MAMBA_AVAILABLE:
        return MambaBlock(
            d_model = config.predictor_dim,
            d_state = config.d_state,
            d_conv  = config.d_conv,
            expand  = config.expand,
        )
    else:
        return SimplifiedSSM(
            d_model = config.predictor_dim,
            d_state = config.d_state,
            expand  = config.expand,
        )


class MambaLayer(nn.Module):
    """
    Bloco Mamba Bidirecional para dados não-causais (Áudio Espacial).
    Processa a sequência para frente e para trás com Gradient Checkpointing para economizar VRAM.
    """
    def __init__(self, config: MambaPredictorConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.predictor_dim)
        
        # Instancia dois SSMs para bidirecionalidade
        self.mamba_fwd = _build_mamba_block(config)
        self.mamba_bwd = _build_mamba_block(config)
        
        # Projeção para fundir as direções
        self.proj_fusion = nn.Linear(config.predictor_dim * 2, config.predictor_dim)

        # 🟢 Habilita Gradient Checkpointing (configurável)
        self.use_checkpointing = config.use_checkpointing

    def _inner_forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Cálculo denso que será apagado da VRAM e recomputado no backward.
        """
        # Forward pass causal
        out_fwd = self.mamba_fwd(x_norm)
        
        # Backward pass (inverte a sequência temporal L e garante contiguidade)
        x_bwd_in = torch.flip(x_norm, dims=[1]).contiguous()
        out_bwd = self.mamba_bwd(x_bwd_in)
        out_bwd = torch.flip(out_bwd, dims=[1]).contiguous()
        
        # Fusão
        fused = self.proj_fusion(torch.cat([out_fwd, out_bwd], dim=-1))
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, L, D'] """
        x_norm = self.norm(x)
        
        # Se estiver em modo treino E o checkpointing estiver ligado
        if self.training and self.use_checkpointing:
            # use_reentrant=False é o padrão seguro do PyTorch 2.0+
            fused = checkpoint(self._inner_forward, x_norm, use_reentrant=False)
        else:
            fused = self._inner_forward(x_norm)
            
        return x + fused  # Residual


class MambaPredictor(nn.Module):
    """
    Preditor baseado em SSM (Mamba) para o framework LeJEPA.

    Recebe as representações de contexto Z_ctx e aprende a predizer
    as representações alvo Z_tgt nos patches mascarados.

    O design de dimensão reduzida (predictor_dim < embed_dim) segue
    I-JEPA: o preditor intencionalmente tem capacidade menor que os
    codificadores, forçando-o a aprender representações semânticas
    de alto nível em vez de detalhes de baixo nível.

    Fluxo completo:
        z_ctx      [B, L, D]   ← representação do contexto (E_ctx output)
        + pos_tgt  [B, L, D]   ← pos. embed. dos patches ALVO (opcional)
          ↓
        proj_in    [B, L, D']  ← compressão para dim. menor do preditor
          ↓
        N×MambaLayer [B, L, D'] ← captura dinâmicas sequenciais
          ↓
        proj_out   [B, L, D]   ← expansão de volta para D
          ↓
        z_pred     [B, L, D]   ← predição das representações alvo
    """

    def __init__(self, config: MambaPredictorConfig) -> None:
        super().__init__()
        self.config = config

        D  = config.embed_dim
        D_ = config.predictor_dim

        # ── Projeção de entrada: D → D' ───────────────────────────────────────
        self.proj_in = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D_, bias=True),
        )

        # ── Pilha de blocos Mamba ─────────────────────────────────────────────
        self.mamba_layers = nn.ModuleList([
            MambaLayer(config) for _ in range(config.n_mamba_layers)
        ])

        # ── Projeção de saída: D' → D ─────────────────────────────────────────
        self.proj_out = nn.Sequential(
            nn.LayerNorm(D_),
            nn.Linear(D_, D, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialização de pesos com escala √(2/n_layers) para estabilidade."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z_ctx: torch.Tensor,
        pos_embed_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_ctx            : [B, L, D] — representação de contexto do E_ctx
            pos_embed_target : [B, L, D] — pos. embed. dos patches alvo (opcional)
                               Quando fornecido, o preditor recebe informação
                               ONDE deve predizer (patches mascarados).

        Returns:
            z_pred: [B, L, D] — predições para comparar com z_tgt

        Notas:
            A injeção de pos_embed_target é análoga ao design I-JEPA onde
            o preditor recebe as posições alvo como "query" condicional,
            permitindo predição direcionada de regiões específicas.
        """
        x = z_ctx  # [B, L, D]

        # Soma posição dos patches alvo se fornecida
        if pos_embed_target is not None:
            x = x + pos_embed_target   # [B, L, D]

        # ── Projeção D → D' ───────────────────────────────────────────────────
        x = self.proj_in(x)            # [B, L, D']

        # ── Blocos Mamba (processamento SSM sequencial) ───────────────────────
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)         # [B, L, D']

        # ── Projeção D' → D ───────────────────────────────────────────────────
        z_pred = self.proj_out(x)      # [B, L, D]

        return z_pred  # [B, L, D]