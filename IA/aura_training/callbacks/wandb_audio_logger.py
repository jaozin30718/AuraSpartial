"""
callbacks/wandb_audio_logger.py
================================
Logger WandB para visualizações de áudio e representações latentes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB não disponível. Logging visual desabilitado.")


class WandBAudioLogger(Callback):
    """
    Callback para logging visual de qualidade do modelo no WandB:
      - Espectrogramas de mistura vs. separados
      - Matriz de covariância do espaço latente
      - Heatmaps de DoA preditos vs. reais
      - Amostras de áudio reconstruído

    Args:
        log_every_n_epochs : frequência de logging pesado
        n_samples_to_log   : número de amostras a visualizar
        sample_rate        : taxa de amostragem para áudio WandB
    """

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        n_samples_to_log:   int = 4,
        sample_rate:        int = 16_000,
    ) -> None:
        super().__init__()
        self.log_every_n = log_every_n_epochs
        self.n_samples   = n_samples_to_log
        self.sr          = sample_rate
        self._val_batch_cache: Optional[dict] = None

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Cacheia o primeiro batch de validação para visualização."""
        if batch_idx == 0 and self._val_batch_cache is None:
            self._val_batch_cache = {
                k: v[:self.n_samples].detach().cpu()
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Loga visualizações ao final de epochs selecionados."""
        if not WANDB_AVAILABLE:
            return
        if trainer.current_epoch % self.log_every_n != 0:
            return
        if self._val_batch_cache is None:
            return
        if not isinstance(trainer.logger, pl.loggers.WandbLogger):
            return

        batch = self._val_batch_cache

        with torch.no_grad():
            pl_module.eval()
            features = batch["features"].to(pl_module.device)

            # Forward inference
            outputs = pl_module.model(features)

            self._log_spectrograms(trainer, features, outputs)
            self._log_covariance_matrix(trainer, pl_module, features)
            self._log_doa_heatmap(trainer, batch, outputs)
            self._log_audio_samples(trainer, pl_module, batch, outputs)

            pl_module.train()

    def _log_spectrograms(
        self,
        trainer,
        features: torch.Tensor,    # [B, 5, T, F]
        outputs:  dict,
    ) -> None:
        """Loga espectrogramas de entrada e heatmaps de saída."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Magnitude L (entrada)
            axes[0, 0].imshow(
                features[0, 0].cpu().numpy().T,
                aspect="auto", origin="lower", cmap="viridis"
            )
            axes[0, 0].set_title("Input: Mag_L")

            # GCC-PHAT
            axes[0, 1].imshow(
                features[0, 4].cpu().numpy().T,
                aspect="auto", origin="lower", cmap="RdBu"
            )
            axes[0, 1].set_title("Input: GCC-PHAT")

            # Heatmap predito
            if "heatmap" in outputs:
                axes[1, 0].imshow(
                    outputs["heatmap"][0, 0].cpu().numpy(),
                    aspect="auto", origin="lower", cmap="hot"
                )
                axes[1, 0].set_title("Pred: Spatial Heatmap")

            # Máscara BSS (fonte 0, parte real)
            if "masks" in outputs:
                axes[1, 1].imshow(
                    outputs["masks"][0, 0].cpu().numpy().T,
                    aspect="auto", origin="lower", cmap="Blues"
                )
                axes[1, 1].set_title("Pred: BSS Mask Src 0 (Re)")

            plt.tight_layout()
            wandb.log(
                {"val/spectrograms": wandb.Image(fig)},
                step=trainer.global_step
            )
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Erro no log de espectrogramas: {e}")

    def _log_covariance_matrix(
        self,
        trainer,
        pl_module,
        features: torch.Tensor,
    ) -> None:
        """Loga matriz de covariância do espaço latente para monitorar colapso."""
        try:
            import matplotlib.pyplot as plt

            # Extrai representações latentes
            tokens, _ = pl_module.model.patch_embedder(features, mask=None)
            z = pl_module.model.encoder.forward_target(tokens)  # [B, L, D]

            # Achata: [B*L, D]
            z_flat = z.reshape(-1, z.shape[-1]).float()
            z_c    = z_flat - z_flat.mean(dim=0)

            # Covariância [D, D] — visualiza submatriz 64x64
            cov = (z_c.T @ z_c) / (z_c.shape[0] - 1)
            cov_sub = cov[:64, :64].cpu().numpy()

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cov_sub, cmap="RdBu", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Latent Covariance Matrix (first 64 dims)")

            wandb.log(
                {"val/covariance_matrix": wandb.Image(fig)},
                step=trainer.global_step
            )
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Erro no log da covariância: {e}")

    def _log_doa_heatmap(
        self,
        trainer,
        batch:   dict,
        outputs: dict,
    ) -> None:
        """Loga scatter plot de DoA predito vs. real."""
        if "doa" not in outputs or "doa" not in batch:
            return

        try:
            import matplotlib.pyplot as plt

            pred_doa   = outputs["doa"][0].cpu().numpy()    # [N, 2]
            target_doa = batch["doa"][0].cpu().numpy()      # [N, 2]
            N = pred_doa.shape[0]

            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "polar"})
            colors = plt.cm.tab10(range(N))

            for n in range(N):
                az_pred = np.radians(pred_doa[n, 0])
                az_true = np.radians(target_doa[n, 0])
                ax.scatter(az_true, 1.0, c=[colors[n]], marker="o",
                           s=100, label=f"Src {n} True")
                ax.scatter(az_pred, 1.0, c=[colors[n]], marker="x",
                           s=100, label=f"Src {n} Pred")

            ax.set_title("DoA: True (o) vs Predicted (x)")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

            wandb.log(
                {"val/doa_comparison": wandb.Image(fig)},
                step=trainer.global_step
            )
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Erro no log DoA: {e}")

    def _log_audio_samples(
        self,
        trainer,
        pl_module,
        batch:   dict,
        outputs: dict,
    ) -> None:
        """Reconstrói áudio usando as máscaras preditas e loga no WandB."""
        if "masks" not in outputs or "stereo" not in batch:
            return

        try:
            # 🟢 OTIMIZAÇÃO: Forçamos a CPU/GPU a rodar o áudio em FP32 
            # para evitar o bug do ComplexHalf e ruídos de conversão.
            device_type = 'cuda' if pl_module.device.type == 'cuda' else 'cpu'
            with torch.autocast(device_type=device_type, enabled=False):
                
                stereo = batch["stereo"][0].float().to(pl_module.device) # [2, S]
                masks  = outputs["masks"][0].float()                     # [N*2, T, F]
                
                n_fft = 512
                hop_length = 128
                window = torch.hann_window(n_fft).to(stereo.device)
                
                # STFT do canal esquerdo (Mix)
                stft_mix = torch.stft(
                    stereo[0], 
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    return_complex=True
                ) # [F_stft, T_stft]
                
                # 🟢 CORREÇÃO DA MATEMÁTICA cIRM
                N = masks.shape[0] // 2
                mask_real = masks[0].T # [F_mask, T_mask] (Fonte 0 Real)
                mask_imag = masks[N].T # [F_mask, T_mask] (Fonte 0 Imag)
                
                F_mask, T_mask = mask_real.shape
                F_stft, T_stft = stft_mix.shape
                
                # Alinhamento de Shapes (Paddings/Crops)
                if F_stft > F_mask:
                    stft_mix = stft_mix[:F_mask, :]
                elif F_stft < F_mask:
                    stft_mix = torch.nn.functional.pad(stft_mix, (0, 0, 0, F_mask - F_stft))

                if T_stft > T_mask:
                    stft_mix = stft_mix[:, :T_mask]
                elif T_stft < T_mask:
                    stft_mix = torch.nn.functional.pad(stft_mix, (0, T_mask - T_stft))
                
                # Máscara complexa ideal
                M_complex = torch.complex(mask_real, mask_imag)
                
                # Separação!
                stft_est = stft_mix * M_complex
                
                # Ajuste de bins para a ISTFT
                expected_F = n_fft // 2 + 1
                if F_mask < expected_F:
                    stft_est = torch.nn.functional.pad(stft_est, (0, 0, 0, expected_F - F_mask))
                elif F_mask > expected_F:
                    stft_est = stft_est[:expected_F, :]

                # Reconstrução para o tempo
                audio_est = torch.istft(
                    stft_est,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    length=stereo.shape[1]
                )
                
                # Normalização de segurança para o player do WandB não estourar
                audio_est_np = audio_est.cpu().numpy()
                max_val = np.max(np.abs(audio_est_np)) + 1e-8
                if max_val > 1.0:
                    audio_est_np = audio_est_np / max_val
                    
                stereo_np = stereo[0].cpu().numpy()
                
                wandb.log({
                    "val/audio_mix":  wandb.Audio(stereo_np, sample_rate=self.sr),
                    "val/audio_est_0": wandb.Audio(audio_est_np, sample_rate=self.sr)
                }, step=trainer.global_step)
            
        except Exception as e:
            logger.debug(f"Erro no log de áudio: {e}")


