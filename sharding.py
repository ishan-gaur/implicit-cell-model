from pathlib import Path
import torch
from multiprocessing import Pool
import sys

dataset_dir = Path("/home/ishang/cross-modal-autoencoders/FUCCI-dev-data/")
channel_names = ["dapi", "tubulin", "geminin", "cdt1"]

shard_size = 1000
for dataset in channel_names:
    print(f"loading {dataset}")
    data = torch.load(dataset_dir / f"{dataset}.pt")
    print(f"{dataset} loaded, shape: {data.shape}")
    print("saving shards")

    def save_shard(i):
        print(f"Saving {dataset}_{int(i / shard_size)}.pt")
        sys.stdout.flush()
        torch.save(data[i:min(i+shard_size, len(data))].clone(), dataset_dir / f"{dataset}_{int(i / shard_size)}.pt")
    
    print(list(range(0, len(data), shard_size)))
    with Pool(32) as p:
        p.map(save_shard, list(range(0, len(data), shard_size)))