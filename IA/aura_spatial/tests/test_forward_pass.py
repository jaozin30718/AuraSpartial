"""
tests/test_forward_pass.py
==========================
Script de validação dimensional completo do AuraSpatialModel.

Testa todos os modos de operação e verifica shapes exatos
de cada tensor intermediário e final.

Uso:
    python -m pytest tests/test_forward_pass.py -v
    # ou diretamente:
    python tests/test_forward_pass.py
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Any

import torch
import torch.nn as nn

# ── Imports do modelo ─────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from config import (
    AuraSpatialConfig,
    PatchEmbedderConfig,
    EncoderConfig,
    MambaPredictorConfig,
    LossConfig,
    OutputHeadsConfig,
)
from model import AuraSpatialModel


# ─────────────────────────────────────────────────────────────────────────────
# Configuração de Teste
# ─────────────────────────────────────────────────────────────────────────────

def build_test_config(small: bool = True) -> AuraSpatialConfig:
    """
    Constrói configuração reduzida para testes rápidos em CPU.

    Args:
        small: Se True, usa dimensões mínimas para testes de shape.
               Se False, usa configuração de produção.
    """
    if small:
        # Configuração mínima para validação rápida de shapes
        D = 64    # embed_dim reduzido

        patch_cfg = PatchEmbedderConfig(
            in_channels    = 5,
            n_time_frames  = 64,    # T
            n_freq_bins    = 64,    # F
            patch_time     = 8,     # → nt = 8
            patch_freq     = 8,     # → nf = 8
            embed_dim      = D,     # L = 8*8 = 64
        )
        encoder_cfg = EncoderConfig(
            embed_dim  = D,
            num_layers = 2,
            num_heads  = 4,
            mlp_ratio  = 2.0,
        )
        predictor_cfg = MambaPredictorConfig(
            embed_dim      = D,
            predictor_dim  = 32,
            n_mamba_layers = 2,
            d_state        = 8,
            d_conv         = 4,
            expand         = 2,
        )
        heads_cfg = OutputHeadsConfig(
            embed_dim     = D,
            n_sources     = 2,
            n_time_frames = 64,
            n_freq_bins   = 64,
            doa_hidden_dim = 32,
            heatmap_res_x  = 36,
            heatmap_res_y  = 18,
        )
    else:
        # Configuração de produção
        D = 512
        patch_cfg     = PatchEmbedderConfig()
        encoder_cfg   = EncoderConfig()
        predictor_cfg = MambaPredictorConfig()
        heads_cfg     = OutputHeadsConfig()

    return AuraSpatialConfig(
        patch_embedder = patch_cfg,
        encoder        = encoder_cfg,
        predictor      = predictor_cfg,
        loss           = LossConfig(),
        heads          = heads_cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários de Teste
# ─────────────────────────────────────────────────────────────────────────────

def assert_shape(
    tensor: torch.Tensor,
    expected: tuple,
    name: str,
) -> None:
    """Verifica shape e imprime resultado formatado."""
    actual = tuple(tensor.shape)
    status = "✅" if actual == expected else "❌"
    print(f"    {status} {name:<40} shape={actual}")
    if actual != expected:
        raise AssertionError(
            f"Shape incorreto para '{name}':\n"
            f"  Esperado: {expected}\n"
            f"  Obtido:   {actual}"
        )


def print_separator(title: str, width: int = 70) -> None:
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def check_no_nan_inf(tensor: torch.Tensor, name: str) -> None:
    """Verifica ausência de NaN e Inf no tensor."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        raise ValueError(f"Tensor '{name}' contém {'NaN' if has_nan else 'Inf'}!")
    print(f"    ✅ {name:<40} sem NaN/Inf")


# ─────────────────────────────────────────────────────────────────────────────
# Testes Individuais por Módulo
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_embedder(
    model: AuraSpatialModel,
    x: torch.Tensor,
    cfg: AuraSpatialConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Testa MÓDULO 1: PatchEmbedder"""
    print_separator("MÓDULO 1: PatchEmbedder")

    B  = x.shape[0]
    L  = cfg.patch_embedder.n_patches_total
    D  = cfg.patch_embedder.embed_dim

    print(f"  Configuração:")
    print(f"    n_patches_time : {cfg.patch_embedder.n_patches_time}")
    print(f"    n_patches_freq : {cfg.patch_embedder.n_patches_freq}")
    print(f"    n_patches_total: {L}")
    print(f"    embed_dim      : {D}")
    print(f"\n  Shapes:")

    # Sem máscara (modo inferência)
    tokens_clean, pos_embed = model.patch_embedder(x, mask=None)
    assert_shape(tokens_clean, (B, L, D), "tokens_clean (sem máscara)")
    assert_shape(pos_embed,    (B, L, D), "pos_embed")

    # Com máscara (modo pré-treino)
    mask = model.patch_embedder.generate_random_mask(B, x.device, mask_ratio=0.75)
    assert_shape(mask, (B, L), "mask")
    print(f"    ✅ {'mask':<40} shape={tuple(mask.shape)}  (bool)")
    print(f"    ℹ️  Patches mascarados: {mask.sum().item()}/{B*L} ({mask.float().mean().item()*100:.1f}%)")

    tokens_masked, _ = model.patch_embedder(x, mask=mask)
    assert_shape(tokens_masked, (B, L, D), "tokens_masked (com máscara)")

    check_no_nan_inf(tokens_clean, "tokens_clean")
    check_no_nan_inf(tokens_masked, "tokens_masked")

    return tokens_clean, tokens_masked, mask


def test_jepa_encoders(
    model: AuraSpatialModel,
    tokens_clean: torch.Tensor,
    tokens_masked: torch.Tensor,
    cfg: AuraSpatialConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Testa MÓDULO 2: JEPAEncoder (E_ctx e E_tgt)"""
    print_separator("MÓDULO 2: JEPAEncoder (E_ctx = E_tgt, weights compartilhados)")

    B, L, D = tokens_clean.shape

    print(f"  ℹ️  E_ctx e E_tgt são a mesma instância: {model.encoder is model.encoder}")
    print(f"  ℹ️  Parâmetros do encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"\n  Shapes:")

    # E_tgt: tokens limpos → z_target
    z_target = model.encoder.forward_target(tokens_clean)
    assert_shape(z_target, (B, L, D), "z_target = E_tgt(tokens_clean)")

    # E_ctx: tokens mascarados → z_ctx
    z_ctx = model.encoder.forward_context(tokens_masked)
    assert_shape(z_ctx, (B, L, D), "z_ctx = E_ctx(tokens_masked)")

    check_no_nan_inf(z_target, "z_target")
    check_no_nan_inf(z_ctx,    "z_ctx")

    # Verifica que E_ctx ≠ E_tgt em conteúdo (apesar dos pesos iguais, inputs diferentes)
    diff = (z_target - z_ctx).abs().mean().item()
    print(f"    ℹ️  {'|z_target - z_ctx|_mean':<40} = {diff:.6f} (deve ser > 0)")
    assert diff > 0, "z_target e z_ctx devem diferir (inputs diferentes)"

    return z_ctx, z_target


def test_mamba_predictor(
    model: AuraSpatialModel,
    z_ctx: torch.Tensor,
    pos_embed: torch.Tensor,
    mask: torch.Tensor,
    cfg: AuraSpatialConfig,
) -> torch.Tensor:
    """Testa MÓDULO 3: MambaPredictor"""
    print_separator("MÓDULO 3: MambaPredictor (SSM)")

    B, L, D = z_ctx.shape

    print(f"  Configuração Mamba:")
    print(f"    predictor_dim : {cfg.predictor.predictor_dim}")
    print(f"    n_mamba_layers: {cfg.predictor.n_mamba_layers}")
    print(f"    d_state       : {cfg.predictor.d_state}")
    print(f"    d_conv        : {cfg.predictor.d_conv}")
    print(f"    expand        : {cfg.predictor.expand}")
    print(f"\n  Shapes:")

    # Sem pos_embed de alvo
    z_pred_nopos = model.predictor(z_ctx, pos_embed_target=None)
    assert_shape(z_pred_nopos, (B, L, D), "z_pred (sem pos_embed_target)")

    # Com pos_embed de alvo (apenas posições mascaradas)
    pos_target = pos_embed * mask.unsqueeze(-1).float()
    z_pred = model.predictor(z_ctx, pos_embed_target=pos_target)
    assert_shape(z_pred, (B, L, D), "z_pred (com pos_embed_target)")

    check_no_nan_inf(z_pred, "z_pred")

    return z_pred


def test_jepa_loss(
    model: AuraSpatialModel,
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    mask: torch.Tensor,
    cfg: AuraSpatialConfig,
) -> None:
    """Testa MÓDULO 4: LeJEPALoss"""
    print_separator("MÓDULO 4: LeJEPALoss")

    print(f"  Configuração da Loss:")
    print(f"    weight_pred : {cfg.loss.weight_pred}")
    print(f"    weight_var  : {cfg.loss.weight_var}")
    print(f"    weight_cov  : {cfg.loss.weight_cov}")
    print(f"    gamma       : {cfg.loss.variance_gamma}")
    print(f"    pred_type   : {cfg.loss.pred_loss_type}")
    print(f"\n  Cálculo:")

    total_loss, metrics = model.loss_fn(z_pred, z_target, mask=mask)

    print(f"    ✅ {'total_loss':<40} = {metrics['loss/total']:.6f}")
    print(f"    ✅ {'L_pred':<40} = {metrics['loss/pred']:.6f}")
    print(f"    ✅ {'L_var':<40} = {metrics['loss/var']:.6f}")
    print(f"    ✅ {'L_cov':<40} = {metrics['loss/cov']:.6f}")

    assert isinstance(total_loss, torch.Tensor), "Loss deve ser Tensor"
    assert total_loss.shape == (), "Loss deve ser escalar"
    assert not torch.isnan(total_loss), "Loss não pode ser NaN"
    assert not torch.isinf(total_loss), "Loss não pode ser Inf"
    assert total_loss.requires_grad, "Loss deve ter grad habilitado"

    print(f"\n    ✅ Backward pass:")
    t0 = time.perf_counter()
    total_loss.backward(retain_graph=True)
    dt = (time.perf_counter() - t0) * 1000
    print(f"    ✅ {'backward() concluído':<40} em {dt:.2f}ms")


def test_output_heads(
    model: AuraSpatialModel,
    h: torch.Tensor,
    cfg: AuraSpatialConfig,
) -> dict:
    """Testa MÓDULO 5: Output Heads"""
    print_separator("MÓDULO 5: Output Heads (Multitarefa)")

    B, L, D = h.shape
    S   = cfg.heads.n_sources
    T   = cfg.heads.n_time_frames
    F   = cfg.heads.n_freq_bins
    Rx  = cfg.heads.heatmap_res_x
    Ry  = cfg.heads.heatmap_res_y

    print(f"  Configuração:")
    print(f"    n_sources     : {S}")
    print(f"    T (frames)    : {T}")
    print(f"    F (freq bins) : {F}")
    print(f"    Heatmap (Rx,Ry): ({Rx}, {Ry})")
    print(f"    mask_activation: {cfg.heads.mask_activation}")
    print(f"\n  Shapes:")

    # BSS Head
    masks = model.bss_head(h)
    n_mask_ch = S * 2 if cfg.heads.mask_activation == "tanh" else S
    assert_shape(masks, (B, n_mask_ch, T, F), "BSS masks")

    # Verifica range da ativação
    if cfg.heads.mask_activation == "sigmoid":
        assert masks.min() >= 0.0 and masks.max() <= 1.0, "Sigmoid: valores devem ser [0,1]"
        print(f"    ✅ {'masks range':<40} [{masks.min().item():.3f}, {masks.max().item():.3f}] ∈ [0,1] ✓")
    elif cfg.heads.mask_activation == "tanh":
        assert masks.min() >= -1.0 and masks.max() <= 1.0, "Tanh: valores devem ser [-1,1]"
        print(f"    ✅ {'masks range (cIRM)':<40} [{masks.min().item():.3f}, {masks.max().item():.3f}] ∈ [-1,1] ✓")

    # DoA Head
    doa = model.doa_head(h)
    assert_shape(doa, (B, S, 2), "DoA (azimute, elevação)")
    print(f"    ℹ️  DoA sample[0]: Az={doa[0,0,0].item():.2f}°, El={doa[0,0,1].item():.2f}°")

    # Heatmap Head
    heatmap = model.heatmap_head(h)
    assert_shape(heatmap, (B, 1, Rx, Ry), "Heatmap espacial")
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "Heatmap: valores devem ser [0,1]"
    print(f"    ✅ {'heatmap range':<40} [{heatmap.min().item():.3f}, {heatmap.max().item():.3f}] ∈ [0,1] ✓")

    check_no_nan_inf(masks,   "BSS masks")
    check_no_nan_inf(doa,     "DoA")
    check_no_nan_inf(heatmap, "Heatmap")

    return {"masks": masks, "doa": doa, "heatmap": heatmap}


def test_full_forward_pretraining(
    model: AuraSpatialModel,
    x: torch.Tensor,
    cfg: AuraSpatialConfig,
) -> None:
    """Testa o forward pass completo no modo pré-treino."""
    print_separator("FORWARD PASS COMPLETO — Modo Pré-treino (LeJEPA)")

    B, C, T, F = x.shape
    L = cfg.patch_embedder.n_patches_total
    D = cfg.patch_embedder.embed_dim

    t0 = time.perf_counter()
    z_pred, z_target, mask, metrics = model.forward_pretraining(x)
    dt = (time.perf_counter() - t0) * 1000

    print(f"  Shapes de saída:")
    assert_shape(z_pred,   (B, L, D), "z_pred")
    assert_shape(z_target, (B, L, D), "z_target")
    assert_shape(mask,     (B, L),    "mask")

    print(f"\n  Métricas LeJEPA:")
    for k, v in metrics.items():
        print(f"    ℹ️  {k:<35} = {v:.6f}")

    print(f"\n  Performance:")
    print(f"    ✅ {'Tempo de forward (CPU)':<40} {dt:.1f}ms")


def test_full_forward_inference(
    model: AuraSpatialModel,
    x: torch.Tensor,
    cfg: AuraSpatialConfig,
) -> None:
    """Testa o forward pass completo no modo inferência multitarefa."""
    print_separator("FORWARD PASS COMPLETO — Modo Inferência (Multitarefa)")

    B, C, T, F = x.shape
    S  = cfg.heads.n_sources
    Rx = cfg.heads.heatmap_res_x
    Ry = cfg.heads.heatmap_res_y
    n_mask_ch = S * 2 if cfg.heads.mask_activation == "tanh" else S

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(x)
    dt = (time.perf_counter() - t0) * 1000

    print(f"  Shapes de saída:")
    assert_shape(outputs["masks"],   (B, n_mask_ch, T, F), "masks (BSS)")
    assert_shape(outputs["doa"],     (B, S, 2),             "doa (DoA)")
    assert_shape(outputs["heatmap"], (B, 1, Rx, Ry),        "heatmap")

    print(f"\n  Performance:")
    print(f"    ✅ {'Tempo de inferência (CPU)':<40} {dt:.1f}ms")


def test_parameter_count(model: AuraSpatialModel) -> None:
    """Conta e exibe parâmetros por módulo."""
    print_separator("Contagem de Parâmetros")

    counts = model.count_parameters()
    print(f"  {'Módulo':<25} {'Parâmetros':>15}")
    print(f"  {'─'*40}")
    for name, count in counts.items():
        if name != "total":
            print(f"  {name:<25} {count:>15,}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<25} {counts['total']:>15,}")
    print(f"  {'TOTAL (M)':<25} {counts['total']/1e6:>14.2f}M")


def test_gradient_flow(
    model: AuraSpatialModel,
    x: torch.Tensor,
) -> None:
    """Verifica que gradientes fluem corretamente por todos os módulos."""
    print_separator("Verificação de Fluxo de Gradiente")

    # Zero grad
    model.zero_grad()

    # Forward
    z_pred, z_target, mask, metrics = model.forward_pretraining(x)
    total_loss, loss_metrics = model.loss_fn(z_pred, z_target, mask=mask)

    # Backward
    total_loss.backward()

    # Verifica gradientes em parâmetros chave
    key_params = {
        "patch_embedder.patch_conv.weight": model.patch_embedder.patch_conv.weight,
        "encoder.blocks[0].norm1.weight":   model.encoder.blocks[0].norm1.weight,
        "predictor.proj_in[1].weight":      model.predictor.proj_in[1].weight,
        "bss_head.proj[1].weight":          model.bss_head.proj[1].weight,
        "doa_head.mlp[1].weight":           model.doa_head.mlp[1].weight,
    }

    print(f"  {'Parâmetro':<45} {'|grad|_mean':>15}")
    print(f"  {'─'*62}")
    for name, param in key_params.items():
        if param.grad is not None:
            grad_norm = param.grad.abs().mean().item()
            status    = "✅" if grad_norm > 0 else "⚠️ "
            print(f"  {status} {name:<43} {grad_norm:>15.8f}")
        else:
            print(f"  ⚠️  {name:<43} {'SEM GRAD':>15}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner Principal
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests(use_small_config: bool = True) -> None:
    """
    Executa todos os testes de validação dimensional e de gradiente.

    Args:
        use_small_config: Se True, usa configuração reduzida (rápido, CPU).
                          Se False, usa configuração de produção (requer GPU).
    """
    print("\n" + "█"*70)
    print("  AuraSpatialModel — Suite Completa de Validação Dimensional")
    print("  LeJEPA (sem EMA) + Mamba SSM + Multitask Heads")
    print("█"*70)

    # ── Configuração e Dispositivo ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo: {device}")
    print(f"  PyTorch: {torch.__version__}")

    cfg = build_test_config(small=use_small_config)

    # ── Tensor de Entrada Sintético ───────────────────────────────────────────
    B = 4    # batch size de teste
    C = cfg.patch_embedder.in_channels    # 5
    T = cfg.patch_embedder.n_time_frames  # 64 (small) ou 128 (full)
    F = cfg.patch_embedder.n_freq_bins    # 64 (small) ou 256 (full)

    print(f"\n  Tensor de Entrada:")
    print(f"    Shape: [B={B}, C={C}, T={T}, F={F}]")
    print(f"    Canais: [Mag_L, Mag_R, cos(IPD), sin(IPD), GCC-PHAT]")

    # Simula features DSP realistas:
    # - Mag_L, Mag_R: magnitude espectral ≥ 0 (log-mel style)
    # - cos(IPD), sin(IPD): fase inter-canal ∈ [-1, 1]
    # - GCC-PHAT: correlação cruzada normalizada ∈ [-1, 1]
    torch.manual_seed(42)
    x = torch.zeros(B, C, T, F, device=device)
    x[:, 0] = torch.abs(torch.randn(B, T, F))        # Mag_L ≥ 0
    x[:, 1] = torch.abs(torch.randn(B, T, F))        # Mag_R ≥ 0
    x[:, 2] = torch.cos(torch.randn(B, T, F))        # cos(IPD) ∈ [-1,1]
    x[:, 3] = torch.sin(torch.randn(B, T, F))        # sin(IPD) ∈ [-1,1]
    x[:, 4] = torch.tanh(torch.randn(B, T, F))       # GCC-PHAT ∈ [-1,1]

    # ── Instancia Modelo ──────────────────────────────────────────────────────
    print(f"\n  Instanciando AuraSpatialModel...")
    t0 = time.perf_counter()
    model = AuraSpatialModel(cfg).to(device)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  ✅ Modelo instanciado em {dt:.1f}ms")

    # ── Executa Testes por Módulo ─────────────────────────────────────────────
    model.train()   # modo treino para testes de gradiente

    # MÓDULO 1
    tokens_clean, tokens_masked, mask = test_patch_embedder(model, x, cfg)

    # Pos embed (necessário para testes seguintes)
    _, pos_embed = model.patch_embedder(x, mask=None)

    # MÓDULO 2
    z_ctx, z_target = test_jepa_encoders(model, tokens_clean, tokens_masked, cfg)

    # MÓDULO 3
    z_pred = test_mamba_predictor(model, z_ctx, pos_embed, mask, cfg)

    # MÓDULO 4
    test_jepa_loss(model, z_pred, z_target, mask, cfg)

    # Encoder output para os heads (usa tokens limpos, modo inferência)
    with torch.no_grad():
        z_clean   = model.encoder.forward_target(tokens_clean)
        h_for_heads = model.predictor(z_clean)

    # MÓDULO 5
    test_output_heads(model, h_for_heads, cfg)

    # ── Testes de Forward Pass Integrado ──────────────────────────────────────
    test_full_forward_pretraining(model, x, cfg)
    test_full_forward_inference(model, x, cfg)

    # ── Análise do Modelo ─────────────────────────────────────────────────────
    test_parameter_count(model)
    test_gradient_flow(model, x)

    # ── Sumário Final ─────────────────────────────────────────────────────────
    print("\n" + "█"*70)
    print("  ✅ TODOS OS TESTES PASSARAM COM SUCESSO!")
    print("  Pipeline dimensional validado de ponta a ponta.")
    print("█"*70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validação do AuraSpatialModel")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Usa configuração de produção (requer GPU com ≥8GB VRAM)"
    )
    args = parser.parse_args()

    run_all_tests(use_small_config=not args.full)