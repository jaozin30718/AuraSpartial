"""
config.py
=========
Dataclasses de configuração para toda a arquitetura AuraSpatial.
Centraliza hiperparâmetros com tipagem estrita e valores padrão validados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PatchEmbedderConfig:
    """Configuração do módulo de patching e embedding de entrada."""

    # Shape de entrada: [B, C, T, F]
    in_channels: int = 5               # [Mag_L, Mag_R, cos(IPD), sin(IPD), GCC-PHAT]
    n_time_frames: int = 128           # T — frames temporais do espectrograma
    n_freq_bins: int = 256             # F — bins de frequência (ex: STFT com n_fft=512)

    # Tamanho do patch (stride = kernel_size → patches não sobrepostos)
    patch_time: int = 8                # px na dimensão temporal
    patch_freq: int = 16               # px na dimensão de frequência

    # Dimensão do espaço latente
    embed_dim: int = 512               # D

    # Positional encoding 2D aprendível
    learnable_pos_embed: bool = True

    @property
    def n_patches_time(self) -> int:
        return self.n_time_frames // self.patch_time

    @property
    def n_patches_freq(self) -> int:
        return self.n_freq_bins // self.patch_freq

    @property
    def n_patches_total(self) -> int:    # L
        return self.n_patches_time * self.n_patches_freq

    @property
    def patch_dim(self) -> int:          # dimensão bruta de um patch achatado
        return self.in_channels * self.patch_time * self.patch_freq


@dataclass
class EncoderConfig:
    """Configuração dos codificadores LeJEPA (contexto e alvo)."""

    embed_dim: int = 512               # D — deve coincidir com PatchEmbedderConfig
    num_layers: int = 6                # profundidade do Transformer
    num_heads: int = 8                 # cabeças de atenção multi-head
    mlp_ratio: float = 4.0            # razão da dimensão oculta do MLP
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # Mascaramento
    mask_ratio: float = 0.75          # fração de patches mascarados


@dataclass
class MambaPredictorConfig:
    """Configuração do preditor Mamba (SSM)."""

    embed_dim: int = 512               # D de entrada
    predictor_dim: int = 256           # D' — dim interna do preditor (tipicamente < D)
    n_mamba_layers: int = 4            # blocos Mamba empilhados

    # Parâmetros do bloco Mamba
    d_state: int = 16                  # dimensão do estado oculto SSM
    d_conv: int = 4                    # kernel da convolução causal interna
    expand: int = 2                    # fator de expansão do SSM

    dropout: float = 0.0
    use_checkpointing: bool = True


@dataclass
class LossConfig:
    """Configuração da função de perda LeJEPA."""

    # Pesos das três componentes da perda
    weight_pred: float = 1.0
    weight_var: float = 25.0
    weight_cov: float = 25.0

    # Limiar de variância mínima (γ)
    variance_gamma: float = 1.0

    # Epsilon numérico para estabilidade
    eps: float = 1e-6

    # Tipo de perda preditiva
    pred_loss_type: str = "smooth_l1"   # "l2" | "smooth_l1" | "cosine"
    smooth_l1_beta: float = 1.0

    # Estabilidade e regularização
    normalize_before_cov:  bool = False
    detach_target:         bool = True
    warmup_regularization: bool = True
    warmup_steps:          int = 1000
    detailed_metrics:      bool = True


@dataclass
class OutputHeadsConfig:
    """Configuração dos cabeçotes de saída multitarefa."""

    embed_dim: int = 512
    n_sources: int = 2                 # número de fontes a separar

    # BSS Head
    n_time_frames: int = 128
    n_freq_bins: int = 256
    mask_activation: str = "sigmoid"   # "sigmoid" | "tanh" (para cIRM)

    # DoA Head
    # output shape: [B, n_sources, 2] → (azimute, elevação)
    doa_hidden_dim: int = 256

    # Heatmap Head
    heatmap_res_x: int = 36           # bins de azimute (ex: 36 → 10°/bin)
    heatmap_res_y: int = 18           # bins de elevação (ex: 18 → 10°/bin)


@dataclass
class AuraSpatialConfig:
    """Configuração raiz do modelo completo."""

    patch_embedder: PatchEmbedderConfig = field(default_factory=PatchEmbedderConfig)
    encoder:        EncoderConfig       = field(default_factory=EncoderConfig)
    predictor:      MambaPredictorConfig = field(default_factory=MambaPredictorConfig)
    loss:           LossConfig          = field(default_factory=LossConfig)
    heads:          OutputHeadsConfig   = field(default_factory=OutputHeadsConfig)

    def __post_init__(self):
        """Valida consistência entre módulos."""
        assert self.patch_embedder.embed_dim == self.encoder.embed_dim, (
            "embed_dim deve ser igual entre PatchEmbedder e Encoder"
        )
        assert self.encoder.embed_dim == self.predictor.embed_dim, (
            "embed_dim deve ser igual entre Encoder e MambaPredictor"
        )
        assert self.encoder.embed_dim == self.heads.embed_dim, (
            "embed_dim deve ser igual entre Encoder e OutputHeads"
        )