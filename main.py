import os
import time
from pathlib import Path
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from LightningModules import AutoEncoder, FUCCIDataModule, ReconstructionVisualization
from models import Encoder, Decoder

##########################################################################################
# Experiment parameters
##########################################################################################
config = {
    "imsize": 256,
    "batch_size": 48,
    "num_workers": 16,
    "split": (0.64, 0.16, 0.2),
    "lr": 1e-4,
}

##########################################################################################
# Set up environment and logging
##########################################################################################

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="Train a model on the FUCCI dataset.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--single", action="store_true", help="train single model")
parser.add_argument("-m", "--model", help="specify which model to train: reference, fucci, or total")
parser.add_argument("-d", "--data", required=True, help="path to dataset")

args = parser.parse_args()

if not args.single:
    raise NotImplementedError("Only single model training is supported at this time.")

if args.model not in ["reference", "fucci", "total"]:
    raise ValueError("Model must be one of: reference, fucci, total")

fucci_path = Path(args.data)

def print_with_time(msg):
    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] {msg}")

project_name = f"FUCCI_{args.model}_VAE"
wandb_logger = WandbLogger(
    project=project_name,
    log_model=True,
    save_dir="/data/ishang/fucci_vae/wandb_logs",
    config=config
)

##########################################################################################
# Set up data, model, and trainer
##########################################################################################

print_with_time("Setting up data module...")
dm = FUCCIDataModule(
    data_dir=fucci_path,
    dataset=args.model,
    imsize=config["imsize"],
    split=config["split"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"]
)
# dm.setup(args.model)

print_with_time("Setting up Autoencoder...")
encoder = Encoder(nc=2 if args.model in ["reference", "fucci"] else 4, imsize=config["imsize"])
decoder = Decoder(nc=2 if args.model in ["reference", "fucci"] else 4, imsize=config["imsize"])
model = AutoEncoder(encoder, decoder, lr=config["lr"])

print_with_time("Setting up trainer...")
lightning_dir = f"/data/ishang/fucci_vae/lightning_logs/{project_name}"

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val/loss",
    mode="min",
    dirpath=lightning_dir,
    filename="epoch-{epoch:02d}-val-loss-{val_loss:.2f}",
)

trainer = pl.Trainer(
    default_root_dir=lightning_dir,
    accelerator="gpu",
    devices=8,
    # accelerator="cpu",
    # fast_dev_run=10,
    # detect_anomaly=True,
    num_sanity_val_steps=2,
    # overfit_batches=5,
    # log_every_n_steps=10,
    logger=wandb_logger,
    max_epochs=100,
    callbacks=[
        checkpoint_callback,
        # EarlyStopping(monitor="val/loss", min_delta=config["min_delta"], mode="min"),
        LearningRateMonitor(logging_interval='step'),
        ReconstructionVisualization()
    ]
)

##########################################################################################
# Train and test model
##########################################################################################

print_with_time("Training model...")
trainer.fit(model, dm)

print_with_time("Testing model...")
trainer = pl.Trainer(devices=1, num_nodes=1)
trainer.test(model, dm)