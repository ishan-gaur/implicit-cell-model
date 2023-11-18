import os
import time
from pathlib import Path
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from lightning_modules import FUCCIModel, FUCCIDataModule
from metrics import ReconstructionVisualization, EmbeddingLogger, FUCCIPredictionLogger

chkpt = Path("/data/ishang/fucci_vae/FUCCI_total_VAE_2023_06_28_07_08/lightning_logs/277-963202.12.ckpt")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="Train a model to predict FUCCI channels from reference.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data", required=True, help="path to dataset")
parser.add_argument("-r", "--augment", action="store_true", help="augment the dataset (random rotations)")
parser.add_argument("-e", "--epochs", type=int, default=100, help="maximum number of epochs to train for")
parser.add_argument("-m", "--checkpoint", help="path to checkpoint for common pretrained autoencoder")
parser.add_argument("-p", "--pretrained", help="path to pretrained model to load from")
parser.add_argument("-a", "--all", action="store_true", help="train all of the model parameters")
parser.add_argument("-f", "--featurizer", action="store_true", help="train the featurizer model as well")
parser.add_argument("-g", "--generation", action="store_true", help="train the generation model as well")
parser.add_argument("-w", "--warmup", action="store_true", help="warmup the model learning rate over 10 epochs")
parser.add_argument("-c", "--cpu", action="store_true", help="run on CPU")

args = parser.parse_args()

if args.checkpoint is not None:
    if not Path(args.checkpoint).exists():
        raise ValueError("Checkpoint path does not exist.")
    else:
        chkpt = Path(args.checkpoint)
    
config = {
    "imsize": 256,
    "latent_dim": 512,
    "batch_size": 8,
    # "devices": [7],
    # "devices": list(range(0, 4)),
    "devices": list(range(4, torch.cuda.device_count())),
    # "devices": list(range(0, torch.cuda.device_count())),
    # "devices": list(range(2, torch.cuda.device_count())),
    # "devices": [2, 3, 4, 5, 6, 7],
    "num_workers": 1,
    # "num_workers": 4,
    # "num_workers": 5,
    # "num_workers": 8,
    "split": (0.80, 0.10, 0.10),
    "lr": 5e-5,
    "grad_clip": 1e3,
    # "eps": 1e-12,
    "epochs": args.epochs,
    "mapper_mults": (3, 3),
    "warmup": args.warmup,
    "augmentation": args.augment,
}

def print_with_time(msg):
    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] {msg}")

print_with_time("Setting up run file folders...")
fucci_path = Path(args.data)
project_name = f"FUCCI_fucci_pred_VAE"
log_folder = Path(f"/data/ishang/fucci_vae/{project_name}_{time.strftime('%Y_%m_%d_%H_%M')}")
if not log_folder.exists():
    os.makedirs(log_folder, exist_ok=True)
lightning_dir = log_folder / "lightning_logs"
wandb_dir = log_folder


print_with_time("Setting up data module...")
dm = FUCCIDataModule(
    data_dir=fucci_path,
    dataset="all",
    imsize=config["imsize"],
    split=config["split"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"]
)

print_with_time("Setting up model...")
if args.pretrained is not None:
    model = FUCCIModel.load_from_checkpoint(args.pretrained)
    model.train_all = args.all
    model.train_generation = args.generation
    model.lr = config["lr"]
    model.warmup = config["warmup"]
    model.augmentation = config["augmentation"]
else:
    model = FUCCIModel(
        ae_checkpoint=chkpt,
        latent_dim=config["latent_dim"],
        lr=config["lr"],
        map_widths=config["mapper_mults"],
        train_all=args.all,
        train_encoder=args.featurizer,
        train_decoder=args.generation,
        warmup=config["warmup"],
        augmentation=config["augmentation"],
    )

wandb_logger = WandbLogger(
    project=project_name,
    log_model=True,
    save_dir=wandb_dir,
    config=config
)

wandb_logger.watch(model, log="all", log_freq=10)

val_checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="validate/loss",
    mode="min",
    dirpath=lightning_dir,
    filename="{epoch:02d}-{validate/loss:.2f}",
    auto_insert_metric_name=False,
)

latest_checkpoint_callback = ModelCheckpoint(dirpath=lightning_dir, save_last=True)

trainer = pl.Trainer(
    default_root_dir=lightning_dir,
    accelerator="gpu" if not args.cpu else "cpu",
    devices=config["devices"] if not args.cpu else "auto",
    strategy='ddp_find_unused_parameters_true',
    logger=wandb_logger,
    max_epochs=config["epochs"],
    gradient_clip_val=config["grad_clip"],
    deterministic=True,
    callbacks=[
        val_checkpoint_callback,
        latest_checkpoint_callback,
        LearningRateMonitor(logging_interval='step'),
        ReconstructionVisualization(channels=None, mode="fucci", every_n_epochs=1),
        FUCCIPredictionLogger(every_n_epochs=1),
    ]
)

print_with_time("Training model...")
trainer.fit(model, dm)