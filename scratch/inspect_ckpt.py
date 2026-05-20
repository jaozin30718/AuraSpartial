import torch
import sys
from pathlib import Path

sys.path.append(r"c:\Users\Leonan\Documents\Ruido\IA")
sys.path.append(r"c:\Users\Leonan\Documents\Ruido\IA\aura_training")

torch.serialization.add_safe_globals(["TrainingConfig", "TrainingPhase", "OptimizerConfig", "SchedulerConfig", "JEPAPhaseConfig", "MultitaskPhaseConfig", "LoRAConfig"])
from config.training_config import TrainingConfig, TrainingPhase
from aura_spatial.model import AuraSpatialModel
from aura_spatial.config import AuraSpatialConfig
from lightning_module import AuraLightningModule

# 1. Load checkpoint optimizer state
ckpt_path = r"C:\Users\Leonan\Documents\Ruido\outputs\checkpoints\last-v9.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("--- SAVED GROUPS ---")
saved_groups = ckpt["optimizer_states"][0]["param_groups"]
for i, group in enumerate(saved_groups):
    print(f"Group {i}: {len(group['params'])} params, name: {group.get('name', 'N/A')}")

# 2. Initialize current model & lightning module
model_config = AuraSpatialConfig()
model = AuraSpatialModel(model_config)
config = TrainingConfig(phase=TrainingPhase.JEPA, train_hdf5_dir="", val_hdf5_dir="")

pl_module = AuraLightningModule(model=model, config=config, total_steps=100)
pl_module._freeze_for_phase(TrainingPhase.JEPA)

print("\n--- CURRENT GROUPS ---")
current_groups = pl_module._build_param_groups()
for i, group in enumerate(current_groups):
    print(f"Group {i}: {len(group['params'])} params, name: {group.get('name', 'N/A')}")
