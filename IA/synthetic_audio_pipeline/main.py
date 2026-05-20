"""
main.py
=======
Entry point do pipeline de geração de dados sintéticos.

Uso:
  python main.py                          # configuração padrão
  python main.py --total 100000           # 100k amostras
  python main.py --workers 16 --format webdataset
  python main.py --validate              # valida 1 amostra e mostra stats
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from config import PipelineConfig, RoomConfig, AugmentationConfig
from batch_generator import BatchGenerator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de Geração de Dados de Áudio Sintético para BSS/DoA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--total",        type=int,   default=15000,    help="Total de amostras a gerar")
    parser.add_argument("--workers",      type=int,   default=6,        help="Número de workers paralelos")
    parser.add_argument("--output",       type=str,   default="./dataset", help="Diretório de saída")
    parser.add_argument("--sources",      type=str,   default="./audio_sources", help="Dir de fontes de voz/áudio")
    parser.add_argument("--noise",        type=str,   default="./noise_sources", help="Dir de ruídos ambientes")
    parser.add_argument("--sample-rate",  type=int,   default=16_000,      help="Taxa de amostragem (Hz)")
    parser.add_argument("--duration",     type=float, default=4.0,         help="Duração por amostra (s)")
    parser.add_argument("--format",       type=str,   default="hdf5",      choices=["hdf5", "webdataset"])
    parser.add_argument("--seed",         type=int,   default=None,        help="Seed global (None=aleatório)")
    parser.add_argument("--validate",     action="store_true",             help="Gera 1 amostra e valida stats")
    parser.add_argument("--verbose",      action="store_true")

    return parser.parse_args()


def validate_single_sample(config: PipelineConfig) -> None:
    """
    Gera uma única amostra de forma síncrona e imprime estatísticas.
    Útil para verificar instalação e configuração sem iniciar pipeline completo.
    """
    import os
    from dataset_loaders import AudioFileIndex, AudioLoader
    from room_simulator import RoomSimulator
    from noise_injector import NoiseInjector
    from ground_truth_extractor import GroundTruthExtractor

    logger.info("=== MODO VALIDAÇÃO — Gerando 1 amostra ===")

    rng = np.random.default_rng(42)

    src_index    = AudioFileIndex(config.audio_source_dir)
    audio_loader = AudioLoader(target_sr=config.room.sample_rate)
    room_sim     = RoomSimulator(config.room, audio_loader, src_index)
    noise_inj    = NoiseInjector(config.augmentation)
    gt_ext       = GroundTruthExtractor()

    logger.info("Simulando sala...")
    sim = room_sim.simulate(rng)

    logger.info("Aplicando augmentation...")
    aug = noise_inj.apply(sim, rng)

    logger.info("Extraindo ground truth...")
    gt  = gt_ext.extract(aug)

    # Imprime estatísticas
    print("\n" + "="*60)
    print("RESULTADO DA VALIDAÇÃO")
    print("="*60)
    print(f"Sala:               {sim.room_dims} m")
    print(f"RT60 alvo/medido:   {sim.rt60_target:.2f}s / {sim.rt60_measured or '?'}")
    print(f"Nº fontes:          {gt.n_sources}")
    print(f"Array estéreo:")
    print(f"  Mic 0:            {sim.mic_positions[0]}")
    print(f"  Mic 1:            {sim.mic_positions[1]}")
    print(f"  Baseline:         {np.linalg.norm(sim.mic_positions[0]-sim.mic_positions[1]):.4f}m")
    print()

    for i, meta in enumerate(sim.sources_meta):
        print(f"  Fonte {i}:")
        print(f"    Posição:        {meta.position}")
        print(f"    Azimute:        {meta.azimuth_deg:.1f}°")
        print(f"    Elevação:       {meta.elevation_deg:.1f}°")
        print(f"    Distância:      {meta.distance_m:.2f}m")
        print(f"    TDoA:           {meta.tdoa_samples:.3f} amostras")
        print(f"    SI-SDR:         {gt.si_sdr_per_source[i]:.1f} dB")

    print()
    print(f"Áudio estéreo:      {aug.stereo_mix_augmented.shape} float32")
    print(f"Heatmap espacial:   {gt.spatial_heatmap.shape}")
    print(f"Augmentation log:   {aug.augmentation_log}")
    print("="*60)
    print("[OK] Validação bem-sucedida!")


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level   = logging.DEBUG if args.verbose else logging.INFO,
        format  = "%(asctime)s | %(levelname)7s | %(message)s",
        datefmt = "%H:%M:%S",
        stream  = sys.stdout,
    )

    # Monta configuração
    room_config = RoomConfig(
        sample_rate              = args.sample_rate,
        audio_duration_seconds   = args.duration,
    )

    aug_config = AugmentationConfig()

    config = PipelineConfig(
        room             = room_config,
        augmentation     = aug_config,
        output_dir       = args.output,
        audio_source_dir = args.sources,
        noise_source_dir = args.noise,
        total_samples    = args.total,
        n_workers        = args.workers or max(1, (__import__("os").cpu_count() or 4) - 2),
        use_webdataset   = (args.format == "webdataset"),
        global_seed      = args.seed,
        verbose          = args.verbose,
    )

    if args.validate:
        validate_single_sample(config)
        return 0

    # Pipeline completo
    logger.info("🚀 Iniciando pipeline de geração paralela...")
    generator = BatchGenerator(config)
    generator.generate()

    return 0


if __name__ == "__main__":
    sys.exit(main())