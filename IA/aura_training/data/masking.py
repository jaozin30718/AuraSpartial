"""
data/masking.py
===============
Implementação do Block Masking para treinamento LeJEPA.

Estratégia: Seleciona blocos contíguos 2D (tempo × frequência) para mascarar,
em vez de patches aleatórios independentes. Isso força o modelo a aprender
representações de longo alcance e é mais desafiador que random masking.

Inspirado em:
  - BEiT v2 (block masking)
  - I-JEPA (multi-block target masking)
  - AudioMAE (spectral block masking)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import numpy as np


class BlockMaskGenerator:
    """
    Gerador de máscaras em bloco 2D para patches espectrais.

    Gera N blocos contíguos aleatórios no grid (nt, nf) de patches,
    onde nt = n_patches_time e nf = n_patches_freq.

    A máscara resultante é um tensor bool flat [L] onde L = nt * nf,
    com True indicando "este patch está mascarado".

    Args:
        n_patches_time  : número de patches na dimensão temporal
        n_patches_freq  : número de patches na dimensão de frequência
        mask_ratio      : fração alvo de patches mascarados
        min_block_time  : tamanho mínimo do bloco temporal (em patches)
        max_block_time  : tamanho máximo do bloco temporal (em patches)
        min_block_freq  : tamanho mínimo do bloco de frequência (em patches)
        max_block_freq  : tamanho máximo do bloco de frequência (em patches)
        max_attempts    : tentativas máximas para atingir mask_ratio
    """

    def __init__(
        self,
        n_patches_time: int,
        n_patches_freq: int,
        mask_ratio:     float = 0.75,
        min_block_time: int   = 4,
        max_block_time: int   = 16,
        min_block_freq: int   = 2,
        max_block_freq: int   = 8,
        max_attempts:   int   = 20,
    ) -> None:
        self.nt = n_patches_time
        self.nf = n_patches_freq
        self.L  = n_patches_time * n_patches_freq
        self.mask_ratio    = mask_ratio
        self.min_bt        = min_block_time
        self.max_bt        = max_block_time
        self.min_bf        = min_block_freq
        self.max_bf        = max_block_freq
        self.max_attempts  = max_attempts
        self.target_masked = int(self.L * mask_ratio)

    def __call__(
        self,
        batch_size: int,
        device:     torch.device = torch.device("cpu"),
        generator:  torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gera máscaras de contexto e alvo para um batch.

        Returns:
            mask_ctx : [B, L] bool — True = mascarado (contexto recebe mask_token)
            mask_tgt : [B, L] bool — True = patch alvo (calcula loss apenas aqui)

        Nota: mask_tgt ⊆ mask_ctx tipicamente, mas podem diferir.
        Aqui usamos mask_ctx == mask_tgt por simplicidade.
        """
        masks = []
        for _ in range(batch_size):
            mask_2d = self._generate_single_mask()
            masks.append(mask_2d.flatten())

        mask_flat = torch.stack(masks, dim=0).to(device)   # [B, L]
        return mask_flat, mask_flat  # ctx e tgt idênticos nesta impl.

    def _generate_single_mask(self) -> torch.Tensor:
        """
        Gera máscara 2D [nt, nf] para uma única amostra.

        Estratégia iterativa:
          1. Inicializa grid não-mascarado
          2. Seleciona ponto de ancoragem aleatório
          3. Cresce bloco aleatório a partir do ponto
          4. Repete até atingir mask_ratio alvo
        """
        mask_2d = torch.zeros(self.nt, self.nf, dtype=torch.bool)
        n_masked = 0

        for _ in range(self.max_attempts):
            if n_masked >= self.target_masked:
                break

            # Tamanho do bloco aleatório
            bt = np.random.randint(self.min_bt, min(self.max_bt, self.nt) + 1)
            bf = np.random.randint(self.min_bf, min(self.max_bf, self.nf) + 1)

            # Ponto de ancoragem (canto superior esquerdo do bloco)
            t0 = np.random.randint(0, max(1, self.nt - bt + 1))
            f0 = np.random.randint(0, max(1, self.nf - bf + 1))

            # Aplica bloco
            t1 = min(t0 + bt, self.nt)
            f1 = min(f0 + bf, self.nf)
            mask_2d[t0:t1, f0:f1] = True
            n_masked = mask_2d.sum().item()

        return mask_2d   # [nt, nf] bool


class ContextTargetSplitter:
    """
    Divide tokens em contexto (mascarado) e alvo (limpo)
    para o loop LeJEPA, respeitando a máscara em bloco.

    Também implementa o protocolo I-JEPA de múltiplos blocos alvo:
    Contexto = patches não-mascarados
    Alvo     = patches mascarados (o que o preditor deve prever)
    """

    @staticmethod
    def split(
        tokens: torch.Tensor,
        mask:   torch.Tensor,
        mask_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aplica a máscara aos tokens de entrada.

        Args:
            tokens    : [B, L, D] — tokens embedados
            mask      : [B, L] bool — True = mascarado
            mask_token: [1, 1, D] — token de máscara aprendível

        Returns:
            tokens_ctx : [B, L, D] — tokens com mascarados → mask_token
            tokens_tgt : [B, L, D] — tokens originais (sem alteração)
            mask       : [B, L]    — máscara (passthrough)
        """
        B, L, D = tokens.shape

        # Tokens alvo: cópia limpa para E_tgt
        tokens_tgt = tokens.clone()   # [B, L, D]

        # Tokens contexto: substitui mascarados pelo mask_token
        mask_expanded = mask.unsqueeze(-1).expand_as(tokens)   # [B, L, D]
        mask_tok_expanded = mask_token.expand(B, L, D)
        tokens_ctx = torch.where(mask_expanded, mask_tok_expanded, tokens)

        return tokens_ctx, tokens_tgt, mask

