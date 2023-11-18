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

from lightning_modules import AutoEncoder, FUCCIDataModule, CrossModalAutoencoder, MultiModalAutoencoder
from lightning_modules import ReconstructionVisualization, EmbeddingLogger
from dataset import MultiModalDataModule, ImageChannelDataset
from models import Encoder, Decoder

##########################################################################################
# Set up environment and parser
##########################################################################################

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="Train a model on the FUCCI dataset.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", help="specify which model to train: reference, fucci, total, or multi")
parser.add_argument("-f", "--data", required=True, help="path to dataset")
parser.add_argument("-r", "--reason", required=True, help="reason for this run")
parser.add_argument("-d", "--dev", action="store_true", help="run in development mode uses 10 percent of data")
parser.add_argument("-c", "--cpu", action="store_true", help="run on CPU")
parser.add_argument("-e", "--epochs", type=int, default=100, help="maximum number of epochs to train for")
parser.add_argument("-l", "--checkpoint", help="path to checkpoint to load from")

args = parser.parse_args()

model_options = ["reference", "fucci", "total", "all", "multi"]
if args.model not in model_options:
    raise ValueError(f"Model must be one of: {model_options}. Got {args.model} instead.")

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
    # "devices": [0, 1, 3, 4, 5, 6, 7],
    "num_workers": 1,
    # "devices": list(range(0, 4)),
    # "devices": list(range(4, torch.cuda.device_count())),
    "devices": list(range(1, torch.cuda.device_count())),
    # "devices": list(range(0, torch.cuda.device_count())),
    # "num_workers": 4,
    # "num_workers": 8,
    "split": (0.64, 0.16, 0.2),
    "lr": 5e-5,
    "eps": 1e-12,
    "factor": 0.5,
    "patience": 10,
    # "min_delta": 1e3,
    "stopping_patience": 20,
    "epochs": args.epochs,
    "model": args.model,
    "latent_dim": 512,
    "lambda": 5e6,
}

fucci_path = Path(args.data)
project_name = f"FUCCI_{args.model}_VAE"
log_folder = Path(f"/data/ishang/fucci_vae/{project_name}_{time.strftime('%Y_%m_%d_%H_%M')}")
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
if args.model == "multi":
    channel_names = ["dapi", "tubulin", "geminin", "cdt1"]
    dataset_dirs = [fucci_path for _ in range(len(channel_names))]
    colors = ["blue", "yellow", "green", "red"]
    dm = MultiModalDataModule(
        dataset_dirs,
        channel_names,
        colors,
        "paired",
        (0.64, 0.16, 0.2),
        config["batch_size"],
        config["num_workers"]
    )
else:
    dm = FUCCIDataModule(
        data_dir=fucci_path,
        dataset=args.model,
        imsize=config["imsize"],
        split=config["split"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

print_with_time("Setting up Autoencoder...")
if args.model == "multi":
    print("Using CrossModalAutoencoder")
    model = CrossModalAutoencoder(
        nc=1,
        nf=config["nf"],
        ch_mult=(1, 2, 4, 8, 8, 8),
        imsize=config["imsize"],
        latent_dim=config["latent_dim"],
        lr=config["lr"],
        lr_eps=config["eps"],
        patience=config["patience"],
        channels=dm.get_channels(),
        # map_widths=(2, 2),
        map_widths=(1,),
    )
elif args.model == "all":
    if args.checkpoint is None:
        print("Using MultiModalAutoencoder")
        model = MultiModalAutoencoder(
            nc=1,
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
else:
    if args.model == "reference":
        nc = 2
    elif args.model == "fucci":
        nc = 2
    elif args.model == "total":
        nc = 1
    else:
        nc = 4
        nf = 32
    if args.checkpoint is None:
        print("Using AutoEncoder")
        model = AutoEncoder(
            nc=nc,
            nf=config["nf"] if args.model != "all" else nf,
            ch_mult=(1, 2, 4, 8, 8, 8),
            lr=config["lr"],
            patience=config["patience"],
            channels=None if args.model == "total" else dm.get_channels(),
            latent_dim=config["latent_dim"],
            eps=config["eps"],
            factor=config["factor"],
            lambda_kl=config["lambda"],
        )
    else:
        print("Loading AutoEncoder from checkpoint")
        model = AutoEncoder.load_from_checkpoint(args.checkpoint)

# model = torch.compile(model)

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