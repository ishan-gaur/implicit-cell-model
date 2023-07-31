import os
import time
from pathlib import Path
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from LightningModules import FUCCIDataModule, CrossModalAutoencoder
from LightningModules import ReconstructionVisualization, EmbeddingLogger
from Dataset import MultiModalDataModule, ImageChannelDataset
from models import Encoder, Decoder

# TODO check
# are the dataloaders for these deterministic? Is there a way to share the random shuffle if I separate them?
# maybe better to do paired vs unpaired from inside the model class
# Need to make this easy to extend to when I want to plug in CCNB1 for alignment
# Sooo maybe I should have models trained with the dataset in mind and keep track of the random seed and everything
# If they come from the same dataset then it will be easy to pair them?
# maybe there should be another way to come up with the pairing score
# should probably start by reading the notebook


##########################################################################################
# Set up environment and parser
##########################################################################################

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="Train a model to align the FUCCI dataset reference channels with the FUCCI channels",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data", required=True, help="path to dataset")
parser.add_argument("-r", "--reason", required=True, help="reason for this run")
parser.add_argument("-e", "--epochs", type=int, default=100, help="maximum number of epochs to train for")
parser.add_argument("-c", "--checkpoint", help="path to checkpoint to load from")
parser.add_argument("-n", "--name", default=time.strftime('%Y_%m_%d_%H_%M'), help="Name to help lookup logging directory")

args = parser.parse_args()

if args.checkpoint is not None:
    if not Path(args.checkpoint).exists():
        raise ValueError("Checkpoint path does not exist.")

##########################################################################################
# Experiment parameters and logging
##########################################################################################
config = {
    "imsize": 256,
    "nf": 128,
    "batch_size": 8,
    # "devices": [4],
    # "devices": list(range(4, torch.cuda.device_count())),
    "devices": list(range(0, torch.cuda.device_count())),
    "num_workers": 1,
    # "num_workers": 4,
    # "num_workers": 8,
    "split": (0.64, 0.16, 0.2),
    "lr": 5e-5,
    "epochs": args.epochs,
    "latent_dim": 512,
    "lambda": 5e6,
}

fucci_path = Path(args.data)
project_name = f"FUCCI_cross_VAE"
log_folder = Path(f"/data/ishang/fucci_vae/{project_name}_{args.name}")
if not log_folder.exists():
    os.makedirs(log_folder, exist_ok=True)
    with open(log_folder / "reason.txt", "w") as f:
        f.write(args.reason)
lightning_dir = log_folder / "lightning_logs"
wandb_dir = log_folder

def print_with_time(msg):
    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] {msg}")

wandb_logger = WandbLogger(
    project=project_name,
    log_model=True,
    save_dir=wandb_dir,
    config=config
)

val_checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="validate/loss",
    mode="min",
    dirpath=lightning_dir,
    filename="{epoch:02d}-{validate/loss:.2f}",
    auto_insert_metric_name=False,
)

latest_checkpoint_callback = ModelCheckpoint(dirpath=lightning_dir, save_last=True)

##########################################################################################
# Set up data, model, and trainer
##########################################################################################

print_with_time("Setting up data module...")
dm = FUCCIDataModule(
    data_dir=fucci_path,
    dataset="all",
    imsize=config["imsize"],
    split=config["split"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"]
)

if args.checkpoint is None:
    print("Using MultiModalAutoencoder")
    model = MultiModalAutoencoder(
        nc=2,
        nf=config["nf"],
        ch_mult=(1, 2, 4, 8, 8, 8),
        imsize=config["imsize"],
        latent_dim=config["latent_dim"],
        lr=config["lr"],
        lr_eps=config["eps"],
        patience=config["patience"],
        channels=dm.get_channels(),
        map_widths=(1,),
        lambda_div=config["lambda"],
    )
else:
    print("Loading MultiModalAutoEncoder from checkpoint")
    model = MultiModalAutoencoder.load_from_checkpoint(args.checkpoint, strict=False)
    model.lr = config["lr"]
    model.lr_eps = config["eps"]
    model.patience = config["patience"]
    model.lambda_div = config["lambda"]

wandb_logger.watch(model, log="all", log_freq=10)

print_with_time("Setting up trainer...")

reconstuction_mode = "single"
if args.model == "multi":
    reconstuction_mode = "multi"
elif args.model == "all":
    reconstuction_mode = "perm"

trainer = pl.Trainer(
    default_root_dir=lightning_dir,
    accelerator="gpu" if not args.cpu else "cpu",
    devices=config["devices"] if not args.cpu else "auto",
    strategy=DDPStrategy(find_unused_parameters=True),
    limit_train_batches=0.1 if args.dev else None,
    limit_val_batches=0.1 if args.dev else None,
    # fast_dev_run=10,
    # detect_anomaly=True,
    # num_sanity_val_steps=2,
    # overfit_batches=32,
    # log_every_n_steps=1,
    logger=wandb_logger,
    max_epochs=config["epochs"],
    gradient_clip_val=5e5 if args.model != "all" else None,
    # gradient_clip_algorithm="norm",
    callbacks=[
        val_checkpoint_callback,
        latest_checkpoint_callback,
        # stopping_callback,
        LearningRateMonitor(logging_interval='step'),
        ReconstructionVisualization(channels=None if args.model == "total" else dm.get_channels(),
                                    mode=reconstuction_mode),
        EmbeddingLogger(every_n_epochs=1, mode=args.model, channels=dm.get_channels() if args.model == "all" else None),
    ]
)

##########################################################################################
# Train and test model
##########################################################################################

print_with_time("Training model...")
trainer.fit(model, dm)

print_with_time("Testing model...")
trainer = pl.Trainer(
    accelerator="gpu" if not args.cpu else "cpu",
    devices=1,
    num_nodes=1,
    limit_test_batches=0.1 if args.dev else None,
)
trainer.test(model, dm)