import os
import numpy as np
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, OneCycleLR
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.collections import MetricCollection
from torchvision.transforms import Normalize

from UNet import UNet


class DepthModelHandler(object):
    def __init__(self,
                 config,
                 train_loader,
                 val_loader,
                 summary_writer):
        """
        Training-evaluation class for depth model.
        :param config: dict, with config.json values
        :param train_loader: DataLoader, training data
        :param val_loader: DataLoader, validation data
        :param summary_writer: SummaryWriter, tensorboard writer
        """
        self.config = config
        train_params = config['hyper_parameters']
        self.epochs = train_params['epochs']
        self.batch_size = train_params['batch_size']
        self.learning_rate = train_params['learning_rate']
        self.optimizer_name = train_params['optimizer']
        self.weight_decay = train_params['weight_decay']
        self.momentum = train_params['momentum']

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.summary_writer = summary_writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Callbacks dict
        self.callbacks = config['callbacks']

        # Fine-tuning file to continue training
        model_params = config['model']
        self.finetuning_file = model_params['fine-tuning-file']
        self.exclude_layers = model_params['exclude_layers']
        self.load_weights_only = model_params['load_weights_only']
        self.train_loaded_weights = model_params['train_loaded_weights']
        self.epochs_per_save = model_params['epochs_per_save']
        self.results_dir = model_params['results_dir']

        # Steps to print results
        self.steps_per_log = model_params['steps_per_log']

        # Model, loss and optimizer
        self.model = UNet().to(self.device)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        # Flag in case of retraining
        self.start_epoch = 0

        # Metrics
        metrics = MetricCollection([SSIM(data_range=(0, 1))]).to(self.device)
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

        self.logs = pd.DataFrame()
        self.logs[['loss_train', 'loss_val', 'ssim_train', 'ssim_val']] = None

    # The default optimizer is adam, because it is used in most UNets.
    def get_optimizer(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adadelta':
            return torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        elif self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)
        else:
            raise ValueError('Supported optimizers: adam, adamw, sgd, adadelta, adagrad, rmsprop')

    def train_step(self, img, mask):
        """
        Performs a single training step.
        :param img: Tensor, input image
        :param mask: Tensor, ground truth mask
        :return: float loss value and Tensor predictions
        """
        img, mask = img.to(self.device), mask.to(self.device)
        preds = self.model(img)
        loss = self.loss_function(preds, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), preds

    @torch.no_grad()
    def evaluation_step(self, img, mask):
        """
        Performs a single evaluation step.
        :param img: Tensor, input image
        :param mask: Tensor, ground truth mask
        :return: float loss value and Tensor predictions
        """
        img, mask = img.to(self.device), mask.to(self.device)
        preds = self.model(img)
        loss = self.loss_function(preds, mask)
        return loss.item(), preds

    def run_epoch(self, scheduler_oc, epoch, mode='train'):
        """
        Runs one epoch of training or evaluation.
        :param scheduler_oc: OneCycleLR, scheduler for OneCycleLR (if is activated)
        :param epoch: int, current epoch
        :param mode: str, train or val
        :return: float ssim value and float loss value
        """

        if mode == 'train':
            self.model.train()
            data_loader = self.train_loader
        else:
            self.model.eval()
            data_loader = self.val_loader

        step = 0
        print('{} mode'.format(mode))
        progress_data = tqdm(data_loader, total=len(data_loader))
        epoch_loss = 0
        for img, mask in progress_data:
            if mode == 'train':
                batch_loss, preds = self.train_step(img, mask)
                self.train_metrics(preds, mask)
            else:
                batch_loss, preds = self.evaluation_step(img, mask)
                self.val_metrics(preds, mask)
            progress_data.set_description(f'loss: {batch_loss:.3f}')

            epoch_loss += batch_loss

            if mode == 'train' and self.callbacks['one_cycle_lr'] and epoch >= self.callbacks['epoch_begin']:
                scheduler_oc.step()

            step += 1

            if step % self.steps_per_log == 0:
                print(f'batch mse: {batch_loss} \taverage mse: {epoch_loss / step} \t')
            del img, mask, preds, batch_loss

        avg_loss = epoch_loss / len(data_loader)
        m = self.train_metrics.compute() if mode == 'train' else self.val_metrics.compute()
        _ssim = m['StructuralSimilarityIndexMeasure'].cpu().item()
        self.logs.loc[epoch, [f'loss_{mode}', f'ssim_{mode}']] = (avg_loss, _ssim)
        self.train_metrics.reset() if mode == 'train' else self.val_metrics.reset()

        return _ssim, avg_loss

    def get_lr(self, optimizer):
        """
        Get current learning rate.
        :param optimizer: Optimizer obj, optimizer
        :return: float, learning rate
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def remove_module(self, layers_dict):
        """
        Removes module from layers dict, because a model trained on GPU has the module prefix
        :param layers_dict: dict, model layers dict
        :return: dict, model layers dict without module prefix
        """
        new_state_dict = OrderedDict()
        for name, v in layers_dict.items():
            if 'module' in name:
                name = name[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    def load_layers(self, checkpoint, model_dict):
        """
        Loads model layers from checkpoint
        :param checkpoint: str, path to checkpoint
        :param model_dict: dict, model layers dict
        """
        new_state_dict = self.remove_module(checkpoint)
        # Keep matching layers
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        # Exclude layers
        for layer in self.exclude_layers:
            if layer in pretrained_dict.keys():
                print(f'Layer {layer} excluded')
                del pretrained_dict[layer]
            else:
                raise ValueError(f'Layer: {layer} to exclude is not in model')

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict, strict=False)

        if not self.train_loaded_weights:
            for name, param in self.model.named_parameters():
                if name in self.exclude_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def check_ckpt_params(self, checkpoint):
        """
        Checks checkpoint parameters
        :param checkpoint: dict, checkpoint parameters
        """
        if self.load_weights_only:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0
        else:
            self.start_epoch = checkpoint['epoch']
            self.optimizer = checkpoint['optimizer']

    def check_pretrained(self):
        """
        Checks if a pre-trained model is loaded
        """
        if self.finetuning_file:
            print(f'Pre-trained model loaded from: {self.finetuning_file}')
            checkpoint = torch.load(self.finetuning_file, map_location='cpu')
            model_dict = self.model.state_dict()
            self.load_layers(checkpoint['model_weights'], model_dict)
            self.check_ckpt_params(checkpoint)
        else:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0

    def save_model(self, epoch, results_dir, best=False):
        """
        Saves model
        :param epoch: int, current epoch
        :param results_dir: str, path to results directory
        :param best: bool, if True saves best model
        """
        model_dir = f'{results_dir}/model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if best:
            ckpt_file = f'{model_dir}/best_model.pth.tar'
            print('Best model saved\n')
        else:
            ckpt_file = f'{model_dir}/model_{epoch}.pth.tar'
            print('Model saved\n')
        torch.save({'epoch': epoch,
                    'model_weights': self.model.state_dict(),
                    'optimizer': self.optimizer}, ckpt_file)

    def run(self):
        """
        Runs the training-evaluation loop
        :return: dict metrics, float best ssim
        """
        print(f'Num model parameters: {self.model._num_params()}')
        self.check_pretrained()
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
        scheduler = self.get_callbacks()
        best_ssim = -1e9
        epoch_max_ssim = self.start_epoch
        for epoch in range(self.start_epoch + 1, self.epochs + self.start_epoch + 1):
            lr = self.get_lr(self.optimizer)
            print('Epoch: {} \t Learning rate {}'.format(epoch, lr))
            train_ssim, train_mse = self.run_epoch(scheduler, epoch)
            val_ssim, val_mse = self.run_epoch(scheduler, epoch, mode='val')
            if epoch >= self.callbacks['epoch_begin']:
                self.activate_callbacks(scheduler, val_mse)

            print(f'Epoch train ssim: {train_ssim} \tEpoch train mse: {train_mse}\n')
            print(f'Epoch val ssim: {val_ssim} \tEpoch val mse: {val_mse}\n')

            if self.epochs_per_save and (epoch % self.epochs_per_save) == 0:
                self.save_model(epoch, self.results_dir)

            if val_ssim > best_ssim:
                self.save_model(epoch, self.results_dir, best=True)
                best_ssim = val_ssim
                epoch_max_ssim = epoch
            self.summary_writer.add_scalar(f'Per_epoch/train_ssim', train_ssim, epoch)
            self.summary_writer.add_scalar(f'Per_epoch/train_mse', train_mse, epoch)
            self.summary_writer.add_scalar(f'Per_epoch/val_ssim', val_ssim, epoch)
            self.summary_writer.add_scalar(f'Per_epoch/val_mse', val_mse, epoch)

        print(f'Best ssim = {best_ssim}, epoch = {epoch_max_ssim}')

        return self.logs.to_dict(), best_ssim

    def get_callbacks(self):
        """
        Returns callbacks
        """
        if self.callbacks['exponential_lr']:
            return StepLR(optimizer=self.optimizer,
                          step_size=self.callbacks['num_epochs_per_decay'],
                          gamma=self.callbacks['lr_decay_factor'])
        elif self.callbacks['plateau_learning_rate']:
            return ReduceLROnPlateau(self.optimizer,
                                     factor=self.callbacks['plateau_decay'],
                                     patience=self.callbacks['plateau_patience_epochs'],
                                     min_lr=self.callbacks['plateau_min_lr'])
        elif self.callbacks['one_cycle_lr']:
            return OneCycleLR(self.optimizer, max_lr=self.learning_rate*25,
                              epochs=self.epochs, steps_per_epoch=len(self.train_loader))
        else:
            return None

    def activate_callbacks(self, scheduler, epoch_test_loss):
        """
        Activates callbacks
        :param scheduler: Scheduler obj, for learning rate
        :param epoch_test_loss: float, test loss
        """
        if self.callbacks['exponential_lr']:
            scheduler.step()
        elif self.callbacks['plateau_learning_rate']:
            scheduler.step(epoch_test_loss)


def colored_depthmap(depth, d_min=None, d_max=None,cmap=plt.cm.inferno):
    """
    Converts depth map to RGB for plotting
    :param depth: numpy array, depth map
    :param d_min: float or int, min value
    :param d_max: float or int, max value
    :param cmap: plotting colormap
    """
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3]


class UnNormalize(Normalize):
    def __init__(self,*args,**kwargs):
        """
        Unnormalizes image
        """
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


@torch.no_grad()
def plot_vals(imgs, targets, preds, results_dir, n=4,figsize=(6, 2),title=''):
    """
    Plots n random images from dataset
    :param imgs: Tensor, images
    :param targets: Tensor, targets
    :param preds: Tensor, predictions
    :param results_dir: str, path to results directory
    :param n: int, number of images to plot
    :param figsize: tuple, figure size
    :param title: str, title
    """
    plt.figure(figsize=figsize,dpi=150)
    r = 2 if n == 4 else 8
    c = 2
    for i,idx in enumerate(np.random.randint(0,imgs.size(0),(n,))):
        ax = plt.subplot(r,c,i + 1)
        img,pred,gt = imgs[idx], preds[idx], targets[idx]
        img = UnNormalize()(img)*255.
        img,pred,gt = img.permute(1,2,0).numpy(), pred.permute(1,2,0).numpy(), gt.permute(1,2,0).numpy()
        pred = colored_depthmap(np.squeeze(pred))
        gt = colored_depthmap(np.squeeze(gt))
        image_viz = np.hstack([img, gt, pred])
        plt.imshow(image_viz.astype("uint8"))
        plt.axis("off")
        plt.imsave(f'{results_dir}/image_{i}.jpg', image_viz.astype("uint8"))
    title = f'{title}\nimage/target/prediction' if len(title)!=0 else 'image/target/prediction'
    plt.suptitle(title)
    plt.show()


def run_test(model, test_loader, results_dir):
    """
    Runs best model on test data and plots results
    :param model: UNet, model
    :param test_loader: DataLoader, test data
    :param results_dir: str, results directory of best experiment
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_imgs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for img, mask in tqdm(test_loader, total=len(test_loader)):
            img, mask = img.to(device), mask.to(device)
            preds = model(img)
            all_imgs.append(img)
            all_preds.append(preds)
            all_targets.append(mask)

    metrics = MetricCollection([SSIM(data_range=(0, 1))]).to(device)
    test_metrics = metrics.clone()
    test_metrics(
        torch.vstack(all_preds),
        torch.vstack(all_targets)
    )
    m = test_metrics.compute()
    title = f"SSIM: {m['StructuralSimilarityIndexMeasure'].cpu().item():.3f}"
    plot_vals(
        torch.vstack(all_imgs).cpu(),
        torch.vstack(all_targets).cpu(),
        torch.vstack(all_preds).cpu(),
        results_dir,
        n=16,
        figsize=(10, 15),
        title=title
    )
