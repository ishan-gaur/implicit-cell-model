import os
import time
from pathlib import Path
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from LightningModules import AutoEncoder, FUCCIDataModule
from LightningModules import ReconstructionVisualization, EmbeddingLogger
from models import Encoder, Decoder

##########################################################################################
# Set up environment and parser
##########################################################################################

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="Train a model on the FUCCI dataset.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-s", "--single", action="store_true", help="train single model")
parser.add_argument("-m", "--model", help="specify which model to train: reference, fucci, or total")
parser.add_argument("-f", "--data", required=True, help="path to dataset")
parser.add_argument("-r", "--reason", required=True, help="reason for this run")
parser.add_argument("-d", "--dev", action="store_true", help="run in development mode uses 10 percent of data")
parser.add_argument("-c", "--cpu", action="store_true", help="run on CPU")
parser.add_argument("-e", "--epochs", type=int, default=100, help="maximum number of epochs to train for")
parser.add_argument("-l", "--checkpoint", help="path to checkpoint to load from")

args = parser.parse_args()

# if not args.single:
#     raise NotImplementedError("Only single model training is supported at this time.")

if args.model not in ["reference", "fucci", "total"]:
    raise ValueError("Model must be one of: reference, fucci, total")

if args.checkpoint is not None:
    if not Path(args.checkpoint).exists():
        raise ValueError("Checkpoint path does not exist.")

##########################################################################################
# Experiment parameters and logging
##########################################################################################
config = {
    "imsize": 256,
    "nf": 128,
    "batch_size": 16,
    "num_devices": 4,
    "num_workers": 8,
    "split": (0.64, 0.16, 0.2),
    "lr": 1e-5,
    "eps": 1e-12,
    "factor": 0.5,
    "patience": 10,
    # "min_delta": 1e3,
    "stopping_patience": 10,
    "epochs": args.epochs,
    "model": args.model,
    "latent_dim": 512,
    # "lambda": 1e-6,
}

fucci_path = Path(args.data)
project_name = f"FUCCI_{args.model}_VAE"
log_folder = Path(f"/data/ishang/fucci_vae/{project_name}_{time.strftime('%Y_%m_%d_%H_%M')}")
if not log_folder.exists():
    os.mkdir(log_folder)
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

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="validate/loss",
    mode="min",
    dirpath=lightning_dir,
    filename="{epoch:02d}-{validate/loss:.2f}",
    auto_insert_metric_name=False,
)

if config["patience"] > config["stopping_patience"]:
    raise ValueError("Patience must be less than stopping patience. LR will never get adjusted.")

# stopping_callback = EarlyStopping(
#     monitor="val/loss",
#     min_delta=config["min_delta"],
#     mode="min",
#     patience=config["stopping_patience"],
# )

stopping_callback = EarlyStopping(
    monitor="lr",
    stopping_threshold=1e-7,
    mode="min",
    patience=config["stopping_patience"],
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

print_with_time("Setting up Autoencoder...")
if args.checkpoint is None:
    model = AutoEncoder(
        nc=2 if args.model in ["reference", "fucci"] else 4,
        nf=config["nf"],
        imsize=config["imsize"],
        lr=config["lr"],
        patience=config["patience"],
        channels=dm.get_channels(),
        latent_dim=config["latent_dim"],
        eps=config["eps"],
        factor=config["factor"],
        # lambda_kl=config["lambda"],
    )
else:
    model = AutoEncoder.load_from_checkpoint(args.checkpoint)

model = torch.compile(model)

wandb_logger.watch(model, log="all", log_freq=10)

print_with_time("Setting up trainer...")

trainer = pl.Trainer(
    default_root_dir=lightning_dir,
    accelerator="gpu" if not args.cpu else "cpu",
    devices=config["num_devices"] if not args.cpu else "auto",
    # devices=[4, 5, 6, 7],
    limit_train_batches=0.1 if args.dev else None,
    limit_val_batches=0.1 if args.dev else None,
    # fast_dev_run=10,
    # detect_anomaly=True,
    # num_sanity_val_steps=2,
    # overfit_batches=32,
    # log_every_n_steps=1,
    logger=wandb_logger,
    max_epochs=config["epochs"],
    callbacks=[
        checkpoint_callback,
        # stopping_callback,
        LearningRateMonitor(logging_interval='step'),
        ReconstructionVisualization(channels=dm.get_channels()),
        EmbeddingLogger(every_n_epochs=1)
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