import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from torchvision.utils import make_grid
from kornia import tensor_to_image
from microfilm.colorify import multichannel_to_rgb
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ReconstructionVisualization(Callback):
    def __init__(self, num_images=8, every_n_epochs=5, channels=None, mode="single"):
        super().__init__()
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.channels = channels
        self.mode = mode

    def __shared_reconstruction_step(self, input_imgs, pl_module, cmap, target_imgs=None):
        # note that input, output, and target images need the same number of channels
        input_imgs = input_imgs.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            reconst_imgs = pl_module(input_imgs)
            pl_module.train()
        if self.mode == "fucci":
            pred_imgs = torch.cat([input_imgs, reconst_imgs], dim=1) # 4-channel prediction
            input_imgs = torch.cat([input_imgs, torch.zeros_like(input_imgs)], dim=1) # two channel input
            reconst_imgs = torch.cat([torch.zeros_like(reconst_imgs), reconst_imgs], dim=1) # two channel reconstruction
            assert target_imgs is not None, "Must provide target images for fucci mode"
            assert input_imgs.shape == reconst_imgs.shape == target_imgs.shape, f"Input, output, and target images must have the same shape. Got {input_imgs.shape}, {reconst_imgs.shape}, and {target_imgs.shape}"
            grid = ReconstructionVisualization.make_reconstruction_grid(input_imgs, reconst_imgs, target_imgs=[pred_imgs, target_imgs])
        else:
            grid = ReconstructionVisualization.make_reconstruction_grid(input_imgs, reconst_imgs)
        grid = tensor_to_image(grid)
        grid = np.moveaxis(grid, -1, 0)
        rgb_grid, _, _, _ = multichannel_to_rgb(grid, cmaps=cmap)
        return rgb_grid, grid

    def __perm_reconstruction_step(self, batch, pl_module, cmap):
        batch = torch.clone(batch)
        batch = batch.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            embeddings = {channel: None for channel in range(len(batch))} # because you can't directly access the channel list here
            for channel, channel_batch in enumerate(batch):
                mu, logvar = pl_module.encode(channel_batch, channel)
                embeddings[channel] = pl_module.reparameterized_sampling(mu, logvar)
            embeddings = [embeddings[channel] for channel in embeddings]
            embeddings = torch.stack(embeddings)
            loss, x_hat, x = pl_module.perm_mse_loss(embeddings, batch)
            pl_module.train()
        x = x.transpose(0, 1)
        x_hat = x_hat.transpose(0, 1)
        x = x.squeeze(2)
        x_hat = x_hat.squeeze(2)
        grid = ReconstructionVisualization.make_reconstruction_grid(x, x_hat)
        grid = tensor_to_image(grid)
        grid = np.moveaxis(grid, -1, 0)
        rgb_grid, _, _, _ = multichannel_to_rgb(grid, cmaps=cmap)
        return rgb_grid, grid

    def __one_to_all_reconstruction_step(self, batch, pl_module, cmap):
        batch = torch.clone(batch)
        batch = batch.to(pl_module.device) # 4 x N x 1 x H x W
        batch.transpose_(0, 1) # N x 4 x 1 x H x W
        batch = batch[:batch.shape[1]] # one prediction per channel
        batch.transpose_(0, 1) # 4 x N(4) x 1 x H x W
        with torch.no_grad():
            pl_module.eval()
            embeddings = {channel: None for channel in range(len(batch))}
            for channel, channel_batch in enumerate(batch):
                mu, logvar = pl_module.encode(channel_batch, channel)
                embeddings[channel] = pl_module.reparameterized_sampling(mu, logvar)
            embeddings = [embeddings[channel] for channel in embeddings]
            embeddings = torch.stack(embeddings) # 4 x N(4) x 512
            embeddings.transpose_(0, 1) # N x 4 x 512
            batch.transpose_(0, 1) # N x 4 x 1 x H x W
            x_hat = torch.zeros_like(batch)
            for channel, image_embed in enumerate(embeddings):
                for output_channel in range(batch.shape[1]):
                    x_hat[channel, output_channel] = pl_module.decode(image_embed[channel, None, ...], output_channel)
            pl_module.train()
        x = batch.squeeze(2)
        x_hat = x_hat.squeeze(2)
        grid = ReconstructionVisualization.make_reconstruction_grid(x, x_hat)
        grid = tensor_to_image(grid)
        grid = np.moveaxis(grid, -1, 0)
        rgb_grid, _, _, _ = multichannel_to_rgb(grid, cmaps=cmap)
        return rgb_grid, grid

    def __many_to_one_reconstruction_step(self, batch, pl_module, cmap):
        batch = torch.clone(batch)
        batch = batch.to(pl_module.device) # 4 x N x 1 x H x W
        batch.transpose_(0, 1) # N x 4 x 1 x H x W
        batch = batch[:batch.shape[1]] # one prediction per channel
        batch.transpose_(0, 1) # 4 x N(4) x 1 x H x W
        with torch.no_grad():
            pl_module.eval()

            # get embeddings per channel in the batch (each one has all images)
            embeddings = {channel: None for channel in range(len(batch))}
            for channel, channel_batch in enumerate(batch):
                mu, logvar = pl_module.encode(channel_batch, channel)
                embeddings[channel] = pl_module.reparameterized_sampling(mu, logvar)
            embeddings = [embeddings[channel] for channel in embeddings]
            embeddings = torch.stack(embeddings) # 4 x N(4) x 512
            embeddings.transpose_(0, 1) # N x 4 x 512
            batch.transpose_(0, 1) # N x 4 x 1 x H x W

            # inference by using all embeddings other than the channel of interest to predict the channel
            x_hat = torch.zeros_like(batch)
            for channel, image_embed in enumerate(embeddings):
                for output_channel in range(batch.shape[1]):
                    if channel != output_channel:
                        x_hat[channel, output_channel] = batch[channel, output_channel]
                    else:
                        embed_pred = torch.zeros_like(image_embed[0])
                        for c in range(batch.shape[1]):
                            if c != channel:
                                embed_pred += image_embed[c]
                        embed_pred /= (batch.shape[1] - 1)
                        x_hat[channel, output_channel] = pl_module.decode(embed_pred, output_channel)
            pl_module.train()
        x = batch.squeeze(2)
        x_hat = x_hat.squeeze(2)
        grid = ReconstructionVisualization.make_reconstruction_grid(x, x_hat)
        grid = tensor_to_image(grid)
        grid = np.moveaxis(grid, -1, 0)
        rgb_grid, _, _, _ = multichannel_to_rgb(grid, cmaps=cmap)
        return rgb_grid, grid

    @staticmethod
    def image_list_normalization(image_list):
        for i in range(len(image_list)):
            image_list[i] = image_list[i] - torch.min(image_list[i])
            image_list[i] = image_list[i] / (torch.max(image_list[i]) - torch.min(image_list[i]))
            image_list[i] = image_list[i] * 2 - 1
        return image_list

    @staticmethod
    def make_reconstruction_grid(input_imgs, reconst_imgs, target_imgs=None):
        if target_imgs is not None:
            input_imgs = input_imgs.to(reconst_imgs.device)
            images = [t.to(reconst_imgs.device) for t in target_imgs]
            images.insert(0, reconst_imgs)
            images.insert(0, input_imgs)
            imgs = torch.stack(images, dim=1).flatten(0, 1)
            grid = make_grid(imgs, nrow=len(images), normalize=True)
        else:
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = make_grid(imgs, nrow=2, normalize=True)
        return grid

    def __shared_logging_step(self, input_imgs, pl_module, cmap, trainer, channel=""):
        if self.mode != "fucci":
            rgb_grid, grid = self.__shared_reconstruction_step(input_imgs, pl_module, cmap)
        else:
            rgb_grid, grid = self.__shared_reconstruction_step(input_imgs[:, :2], pl_module, cmap, target_imgs=input_imgs)

        if self.mode != "multi":
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/reconstruction_samples": wandb.Image(rgb_grid,
                    caption="Original and reconstructed images")
            })

        if self.channels is not None:
            if len(self.channels) != grid.shape[0]:
                raise ValueError(f"Number of channels ({len(self.channels)}) does not match number of images ({grid.shape[0]})")
            for i, channel in enumerate(self.channels):
                trainer.logger.experiment.log({
                    f"{trainer.state.stage}/reconstruction_samples_{channel}": wandb.Image(grid[i],
                        caption=f"Original and reconstructed {channel} images")
                })

        if self.mode == "perm":
            input_imgs = input_imgs.transpose(0, 1) # 4 x N x 256 x 256
            input_imgs = input_imgs[:, :, None, ...] # 4 x N x 1 x 256 x 256
            # rgb_grid, grid = self.__perm_reconstruction_step(input_imgs, pl_module, cmap)
            rgb_grid, grid = self.__perm_reconstruction_step(input_imgs, pl_module, cmap)
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/perm_reconstruction_samples": wandb.Image(rgb_grid,
                    caption="Original and reconstructed images")
            })
            rgb_grid, grid = self.__one_to_all_reconstruction_step(input_imgs, pl_module, cmap)
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/one_to_many_reconstruction_samples": wandb.Image(rgb_grid,
                    caption="Original and reconstructed images")
            })
            rgb_grid, grid = self.__many_to_one_reconstruction_step(input_imgs, pl_module, cmap)
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/many_to_one_reconstruction_samples": wandb.Image(rgb_grid,
                    caption="Original and reconstructed images")
            })

    def __shared_logging_dispatch(self, dataloader, pl_module, trainer):
        if self.mode == "multi":
            for channel, data in dataloader.iterables.items():
                input_imgs = data[:self.num_images]
                cmap = None
                self.__shared_logging_step(input_imgs, pl_module, cmap, trainer, channel)
        else:
            input_imgs = dataloader.dataset[:self.num_images]
            cmap = trainer.datamodule.channel_colors() if input_imgs.shape[-3] > 1 else None
            self.__shared_logging_step(input_imgs, pl_module, cmap, trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.__shared_logging_dispatch(trainer.datamodule.train_dataloader(), pl_module, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.__shared_logging_dispatch(trainer.datamodule.val_dataloader(), pl_module, trainer)
    def on_test_end(self, trainer, pl_module):
        self.__shared_logging_dispatch(trainer.datamodule.test_dataloader(), pl_module, trainer)

class EmbeddingLogger(Callback):
    def __init__(self, num_images=200, every_n_epochs=5, mode="single", channels=[]):
        super().__init__()
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.mode = mode
        self.channels = channels

    def __x_logging_step(self, x, trainer, name, channel=""):
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/{channel}_embedding_{name}_mean": x.mean()
        })
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/{channel}_embedding_{name}_min": x.min()
        })
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/{channel}_embedding_{name}_max": x.max()
        })

    def __shared_logging_step(self, input_imgs, pl_module, trainer, channel=""):
        input_imgs = input_imgs.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            mu, logvar = pl_module.forward_embedding(input_imgs)
            pl_module.train()
        return mu, logvar

    def __shared_logging_dispatch(self, dataloader, pl_module, trainer):
        if self.mode == "multi":
            for channel, data in dataloader.iterables.items():
                input_imgs = data[:self.num_images]
                mu, logvar = self.__shared_logging_step(input_imgs, pl_module, trainer, channel)
                self.__x_logging_step(mu, trainer, "mu", channel)
                self.__x_logging_step(logvar, trainer, "logvar", channel)
        elif self.mode == "all":
            input_imgs = dataloader.dataset[:self.num_images]
            mu, logvar = self.__shared_logging_step(input_imgs, pl_module, trainer)
            for m, l, c in zip(mu, logvar, self.channels):
                self.__x_logging_step(m, trainer, "mu", c)
                self.__x_logging_step(l, trainer, "logvar", c)
        else:
            input_imgs = dataloader.dataset[:self.num_images]
            mu, logvar = self.__shared_logging_step(input_imgs, pl_module, trainer)
            self.__x_logging_step(mu, trainer, "mu")
            self.__x_logging_step(logvar, trainer, "logvar")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.__shared_logging_dispatch(trainer.datamodule.train_dataloader(), pl_module, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.__shared_logging_dispatch(trainer.datamodule.val_dataloader(), pl_module, trainer)
            
    def on_test_end(self, trainer, pl_module):
        self.__shared_logging_dispatch(trainer.datamodule.test_dataloader(), pl_module, trainer)

class FUCCIPredictionLogger(Callback):
    def __init__(self, num_images=1000, every_n_epochs=1):
        super().__init__()
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs

    def __get_predictions(self, input_imgs, pl_module, trainer):
        input_imgs = input_imgs.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            pred = pl_module(input_imgs)
            pl_module.train()
        return pred

    def __fucci_level_from_image(self, img):
        # TODO: this is a problem if the denominator is small although this only happens for mitotic cells
        return torch.sum(img[:, 1] - img[:, 0], dim=(1,2)) / torch.sum(img, dim=(1,2,3))

    def __fucci_states_from_level(self, level):
        with_G2 = torch.where(level > 0.5, 2 * torch.ones_like(level), torch.zeros_like(level))
        states = with_G2 + torch.where((level > -0.5) & (level < 0.5), torch.ones_like(level), torch.zeros_like(level))
        return states
    
    def __shared_logging(self, sample_imgs, pl_module, trainer):
            pred = self.__get_predictions(sample_imgs[:, :2], pl_module, trainer)

            fucci_level_pred = self.__fucci_level_from_image(pred).cpu()
            fucci_level_target = self.__fucci_level_from_image(sample_imgs[:, 2:]).cpu()
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/fucci_level_error_hist": wandb.Histogram(
                    (fucci_level_pred - fucci_level_target).cpu().numpy()
                )
            })

            geminin_target = fucci_level_target[fucci_level_target > 0.5].cpu()
            geminin_pred = fucci_level_pred[fucci_level_target > 0.5].cpu()
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/fucci_level_geminin_error_hist": wandb.Histogram(
                    (geminin_pred - geminin_target).cpu().numpy()
                )
            })

            cdt1_target = fucci_level_target[fucci_level_target < -0.5].cpu()
            cdt1_pred = fucci_level_pred[fucci_level_target < -0.5].cpu()
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/fucci_level_cdt1_error_hist": wandb.Histogram(
                    (cdt1_pred - cdt1_target).cpu().numpy()
                )
            })

            mixed_target = fucci_level_target[(fucci_level_target < 0.5) & (fucci_level_target > -0.5)]
            mixed_pred = fucci_level_pred[(fucci_level_target < 0.5) & (fucci_level_target > -0.5)]
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/fucci_level_mixed_error_hist": wandb.Histogram(
                    (mixed_pred - mixed_target).cpu().numpy()
                )
            })

            fucci_states_target = self.__fucci_states_from_level(fucci_level_target).cpu()
            fucci_states_pred = self.__fucci_states_from_level(fucci_level_pred).cpu()
            fucci_states_target = torch.cat([fucci_states_target, torch.Tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])])
            fucci_states_pred = torch.cat([fucci_states_pred, torch.Tensor([0, 1, 2, 1, 2, 0, 2, 0, 1])])
            conf_matrix = confusion_matrix(fucci_states_target, fucci_states_pred)
            print(torch.unique(fucci_states_target))
            print(torch.unique(fucci_states_target, return_counts=True))
            print(torch.unique(fucci_states_pred))
            print(torch.unique(fucci_states_pred, return_counts=True))
            print(conf_matrix)
            plt.clf()
            ax = sns.heatmap(conf_matrix, annot=True, fmt='d')
            ax.set_xlabel("Predicted")
            # ax.xaxis.set_ticklabels(["G1", "S", "G2"])
            ax.set_ylabel("Target")
            # ax.yaxis.set_ticklabels(["G1", "S", "G2"])
            trainer.logger.experiment.log({
                f"{trainer.state.stage}/fucci_states_confusion_matrix":
                    wandb.Image(ax.figure, caption="Confusion matrix for FUCCI states")
            }, rank_zero_only=True)
            plt.clf()

    # @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = trainer.datamodule.train_dataloader().dataset[:self.num_images]
            print(input_imgs.shape)
            self.__shared_logging(input_imgs, pl_module, trainer)
    
    # @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = trainer.datamodule.val_dataloader().dataset[:self.num_images]
            self.__shared_logging(input_imgs, pl_module, trainer)

    def on_test_end(self, trainer, pl_module):
        input_imgs = trainer.datamodule.test_dataloader().dataset[:self.num_images]
        self.__shared_logging(input_imgs, pl_module, trainer)