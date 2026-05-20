"""
losses/bss_loss.py
==================
SI-SDR Loss com Permutation Invariant Training (PIT).

SI-SDR = Scale-Invariant Signal-to-Distortion Ratio (Le Roux et al. 2019)
PIT    = Permutation Invariant Training (Kolbæk et al. 2017)

O PIT é essencial porque a rede não sabe a priori qual saída corresponde
a qual fonte. Resolve isso encontrando a permutação ótima via Hungarian.
"""

from __future__ import annotations

import itertools
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_sdr(
    estimate:  torch.Tensor,
    reference: torch.Tensor,
    eps:       float = 1e-8,
) -> torch.Tensor:
    """
    Scale-Invariant SDR entre estimativa e referência.

    Args:
        estimate  : [..., S] — sinal estimado
        reference : [..., S] — sinal de referência
        eps       : epsilon numérico

    Returns:
        si_sdr_val: [...] — SI-SDR em dB (maior é melhor)
    """
    # Centraliza (remove componente DC)
    estimate  = estimate  - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    # Projeção ótima de escala
    dot       = (estimate * reference).sum(dim=-1, keepdim=True)   # [..., 1]
    ref_power = (reference ** 2).sum(dim=-1, keepdim=True) + eps    # [..., 1]
    alpha     = dot / ref_power                                      # [..., 1]

    # Componentes target e noise
    target = alpha * reference
    noise  = estimate - target

    # SI-SDR em dB
    si_sdr_val = 10.0 * torch.log10(
        ((target ** 2).sum(dim=-1) + eps) /
        ((noise  ** 2).sum(dim=-1) + eps)
    )
    return si_sdr_val   # [...]


class PIT_SI_SDR_Loss(nn.Module):
    """
    SI-SDR Loss com Permutation Invariant Training.

    Encontra a permutação das saídas que maximiza SI-SDR total,
    depois calcula o gradiente apenas dessa permutação.

    Input shapes:
        estimates  : [B, N_src, S] — N fontes estimadas
        references : [B, N_src, S] — N fontes de referência

    Returns:
        loss : escalar — -SI-SDR médio (queremos minimizar → maximizar SDR)
    """

    def __init__(self, pit: bool = True) -> None:
        super().__init__()
        self.pit = pit

    def forward(
        self,
        estimates:  torch.Tensor,   # [B, N, S]
        references: torch.Tensor,   #[B, N, S]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, S = estimates.shape

        if not self.pit or N == 1:
            sdr = si_sdr(estimates, references)   
            loss = -sdr.mean()
            return loss, sdr.mean(dim=1)

        # Computa SI-SDR Par-a-Par[B, N_est, N_ref] vetorizado (N² cálculos)
        est_exp = estimates.unsqueeze(2)  #[B, N, 1, S]
        ref_exp = references.unsqueeze(1) #[B, 1, N, S]
        pairwise_sdr = si_sdr(est_exp, ref_exp)  # [B, N, N]

        perms = list(itertools.permutations(range(N)))
        perms_t = torch.tensor(perms, device=estimates.device, dtype=torch.long)  # [P, N]

        sdr_matrix = torch.zeros(B, len(perms), device=estimates.device)

        # Soma escalares pré-calculados em vez de rodar o SI-SDR inteiro iterativamente
        for perm_idx, perm in enumerate(perms):
            sdr_sum = 0
            for i in range(N):
                sdr_sum += pairwise_sdr[:, i, perm[i]]
            sdr_matrix[:, perm_idx] = sdr_sum / N

        best_perm_idx = sdr_matrix.argmax(dim=1)                 # [B]
        best_sdr      = sdr_matrix.gather(1, best_perm_idx.unsqueeze(1)).squeeze(1)  # [B]

        # Reconstrói gradiente
        best_perms = perms_t[best_perm_idx]  # [B, N]
        batch_idx = torch.arange(B, device=estimates.device).unsqueeze(1).expand_as(best_perms)
        perm_estimates = estimates[batch_idx, best_perms]  # [B, N, S]
        
        sdr_best = si_sdr(perm_estimates, references)       #[B, N]
        loss = -sdr_best.mean()

        return loss, best_sdr


class BSSLoss(nn.Module):
    """
    Loss completa de BSS: aplica máscara cIRM ao espectrograma
    e calcula SI-SDR no domínio do tempo.

    Args:
        n_fft      : tamanho da FFT (deve coincidir com o collate)
        hop_length : hop da STFT
        pit        : usar PIT
    """

    def __init__(
        self,
        n_fft:      int = 512,
        hop_length: int = 128,
        pit:        bool = True,
    ) -> None:
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.pit_loss   = PIT_SI_SDR_Loss(pit=pit)
        # Registra janela como buffer (evita recriar a cada forward)
        self.register_buffer("window", torch.hann_window(n_fft))

    def apply_cirm(
        self,
        mask_real:     torch.Tensor,   
        mask_imag:     torch.Tensor,   
        mixture_stft:  torch.Tensor,   # [B, F, T] complexo
    ) -> torch.Tensor:
        """
        Aplica cIRM ao STFT da mistura para obter STFT estimado por fonte.
        """
        # [B, F, T] → [B, 1, F, T] para broadcast com N fontes
        X = mixture_stft.unsqueeze(1)   # [B, 1, F, T]

        # Garante alinhamento de dimensões F e T com a mistura
        if mask_real.shape[-1] != mixture_stft.shape[-1]:
            M_r = mask_real.transpose(2, 3)
            M_i = mask_imag.transpose(2, 3)
        else:
            M_r = mask_real
            M_i = mask_imag

        # Máscara complexa: [B, N, F, T]
        M_complex = torch.complex(M_r, M_i)

        # Aplica máscara
        Y = M_complex * X   # [B, N, F, T]
        return Y

    def forward(
        self,
        masks:      torch.Tensor,    # [B, N*2, T, F] ou [B, N*2, F, T]
        stereo:     torch.Tensor,    # [B, 2, S]
        references: torch.Tensor,   # [B, N, S]
        n_fft:      Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            masks      : [B, N*2, T, F] — real e imag concatenados (cIRM)
            stereo     : [B, 2, S]      — mistura estéreo
            references : [B, N, S]      — fontes isoladas de referência

        Returns:
            loss   : escalar SI-SDR
            si_sdrs: [B]
        """
        # Desliga a precisão mista (FP16) para garantir precisão e evitar erros
        # em operações com números complexos (ComplexHalf) durante a Transformada de Fourier.
        device_type = 'cuda' if stereo.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            
            # Força tensores para float32
            masks      = masks.float()
            stereo     = stereo.float()
            references = references.float()
            window_f32 = self.window.float()

            B, _, S = stereo.shape
            N = masks.shape[1] // 2

            # Split real/imag
            mask_real = masks[:, :N, :, :]   
            mask_imag = masks[:, N:, :, :]   

            # STFT da mistura (canal L) — usa janela registrada como buffer
            mix_stft   = torch.stft(
                stereo[:, 0],
                n_fft      = self.n_fft,
                hop_length = self.hop_length,
                window     = window_f32,
                return_complex = True,
            )  # [B, F, T]

            # 🟢 IDENTIFICAÇÃO DINÂMICA DE SHAPE (T e F)
            # Áudio real sempre tem T >> F.
            dim2, dim3 = mask_real.shape[2], mask_real.shape[3]
            if dim2 > dim3:
                T_mask, F_mask = dim2, dim3
            else:
                F_mask, T_mask = dim2, dim3

            F_stft, T_stft = mix_stft.shape[1], mix_stft.shape[2]

            # Ajusta mix_stft para ter o mesmo shape (F, T) das máscaras preditas
            if F_stft > F_mask:
                mix_stft = mix_stft[:, :F_mask, :]
            elif F_stft < F_mask:
                mix_stft = F.pad(mix_stft, (0, 0, 0, F_mask - F_stft))

            if T_stft > T_mask:
                mix_stft = mix_stft[:, :, :T_mask]
            elif T_stft < T_mask:
                mix_stft = F.pad(mix_stft, (0, T_mask - T_stft))

            # Aplica cIRM
            est_stft = self.apply_cirm(mask_real, mask_imag, mix_stft)  # [B, N, F_mask, T_mask]

            # ISTFT batched: [B, N, F_mask, T_mask] → [B*N, F_mask, T_mask]
            est_flat = est_stft.reshape(B * N, *est_stft.shape[2:])  # [B*N, F_mask, T_mask]
            
            # O ISTFT do PyTorch requer que o número de bins seja n_fft // 2 + 1
            expected_F = self.n_fft // 2 + 1
            if F_mask < expected_F:
                est_flat = F.pad(est_flat, (0, 0, 0, expected_F - F_mask))
            elif F_mask > expected_F:
                est_flat = est_flat[:, :expected_F, :]

            # O comprimento temporal realçado pela máscara
            S_trunc = min(S, (T_mask - 1) * self.hop_length)

            est_time = torch.istft(
                est_flat,
                n_fft      = self.n_fft,
                hop_length = self.hop_length,
                window     = window_f32,
                length     = S_trunc,
            )  # [B*N, S_trunc]
            estimates_t = est_time.reshape(B, N, S_trunc)  # [B, N, S_trunc]

            return self.pit_loss(estimates_t, references[:, :, :S_trunc])

