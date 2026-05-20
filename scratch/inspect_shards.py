import h5py
import glob
from pathlib import Path

dataset_dir = "dataset"
shards = sorted(glob.glob(str(Path(dataset_dir) / "shard_*.h5")))

total_samples = 0
missing_heatmap = []

for shard_path in shards:
    try:
        with h5py.File(shard_path, "r") as f:
            sample_keys = list(f.keys())
            total_samples += len(sample_keys)
            
            for sk in sample_keys:
                if "labels/spatial_heatmap" not in f[sk]:
                    missing_heatmap.append((shard_path, sk))
    except Exception as e:
        print(f"Error reading {shard_path}: {e}")

print(f"Total samples found: {total_samples}")
print(f"Samples missing 'spatial_heatmap': {len(missing_heatmap)}")
if missing_heatmap:
    print("Example missing samples:")
    for sm in missing_heatmap[:10]:
        print(f"  {sm}")
