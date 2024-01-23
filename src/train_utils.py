import os
from collections import OrderedDict
import pandas as pd
from tqdm.auto import tqdm
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, OneCycleLR
from torch.utils.data import DataLoader
from torch import Tensor
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.regression import MeanSquaredError as MSE
from torchmetrics.collections import MetricCollection

from UNet import UNet


class DepthModelHandler(object):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        self.config = config
        train_params = config['hyper_parameters']
        self.epochs = train_params['epochs']
        self.batch_size = train_params['batch_size']
        self.learning_rate = train_params['learning_rate']
        self.optimizer_name = train_params['optimizer']
        self.grad_clip = train_params['grad_clip']
        self.weight_decay = train_params['weight_decay']
        self.momentum = train_params['momentum']

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.summary_writer = summary_writer
        self.num_classes = 1
        self.classes = []
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

        # Values of each module
        self.model = UNet().to(self.device)
        self.model.trainable_encoder(trainable=False)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate / 25.,
                                           weight_decay=self.weight_decay)
        self.start_epoch = 0
        self.train_metrics = {}
        self.test_metrics = {}

        metrics = MetricCollection([
            SSIM(data_range=(0, 1)),
            MSE()
        ]).to(self.device)
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

        self.logs = pd.DataFrame()
        self.logs[['loss_train', 'loss_val', 'ssim_train', 'ssim_val', 'mse_train', 'mse_val']] = None
        self.scaler = GradScaler()


    # The default optimizer is adam, because it is used in most UNets.
    def get_optimizer(self) -> torch.optim:
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate / 25.,
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

    def get_metric(self, metric, output, target):
        if metric == 'loss':
            return self.loss_function(output, target)
        else:
            return self.accuracy(output, target)

    def train_step(self, img: Tensor, mask: Tensor) -> tuple:
        with autocast():
            img, mask = img.to(self.device), mask.to(self.device)
            preds = self.model(img)

            loss = self.loss_function(preds, mask)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return loss.item()

    @torch.no_grad()
    def evaluation_step(self, batch: Tensor, target: Tensor, epoch: int) -> tuple:
        batch = batch.to(self.device)
        target = target.to(self.device)
        output = self.model(batch)
        # calculate-the-batch-metrics
        batch_metrics = {}
        for metric in self.test_metrics.keys():
            batch_metrics[metric] = self.get_metric(metric, output, target).item()

        if self.num_classes > 1:
            self.evaluate_per_class(output, target, epoch)
        return batch_metrics, output, torch.mean(batch)

    def log_per_class(self):
        for index, row in self.per_class_metrics.iterrows():
            for label in self.classes:
                self.summary_writer.add_scalar(f'Per_class_accuracy/{label}',
                                               row[label],
                                               index - 1)

    def run_epoch(self, scheduler_oc: torch.optim.lr_scheduler,
                  epoch: int, mode='train') -> dict:

        epoch_metrics = {}
        for metric in self.train_metrics.keys():
            epoch_metrics[metric] = 0.0

        if mode == 'train':
            self.model.train()
            data_loader = self.train_loader
        else:
            self.model.eval()
            data_loader = self.test_loader

        step = 0
        print('{} mode'.format(mode))
        train_data = tqdm(self.train_loader, total=len(self.train_loader))
        epoch_loss = 0
        for img, mask in train_data:
            if mode == 'train':
                batch_loss = self.train_step(img, mask)
            else:
                batch_metrics, outputs, batch_mean = self.evaluation_step(batch, target, epoch)

            for metric in epoch_metrics.keys():
                epoch_metrics[metric] += batch_metrics[metric]

            if self.callbacks['one_cycle_lr'] and epoch >= self.callbacks['epoch_begin']:
                scheduler_oc.step()

            step += 1
            if step % self.steps_per_log == 0:
                log = 'Step {}/{} \tbatch mean: {}\t'.format(step, len(data_loader), batch_mean)
                for metric in epoch_metrics.keys():
                    log += f'batch {metric}: {batch_metrics[metric]} \taverage {metric}: {epoch_metrics[metric] / step} \t'
                print(log)

        avg_metrics = {}
        for metric in epoch_metrics.keys():
            avg_metrics[metric] = epoch_metrics[metric] / step

        if mode == 'test' and 'accuracy' in self.train_metrics:
            for label in self.classes:
                if self.per_class_correct.loc['total', label] != 0.0:
                    self.per_class_metrics.loc[epoch, label] = self.per_class_correct.loc['correct', label] / \
                                                               self.per_class_correct.loc['total', label]
                else:
                    self.per_class_metrics.loc[epoch, label] = 0.0

        return avg_metrics

    def get_lr(self, optimizer: torch.optim) -> float:
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def remove_module(self, layers_dict):
        new_state_dict = OrderedDict()
        for name, v in layers_dict.items():
            if 'module' in name:
                name = name[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    def load_layers(self, checkpoint, model_dict):
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
        if self.load_weights_only:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0
        else:
            self.start_epoch = checkpoint['epoch']
            self.optimizer = checkpoint['optimizer']
            if 'train_metrics' in checkpoint:
                self.train_metrics = checkpoint['train_metrics']
            if 'test_metrics' in checkpoint:
                self.test_metrics = checkpoint['test_metrics']
            if 'per_class_metrics' in checkpoint:
                self.per_class_metrics = checkpoint['per_class_metrics']

    def check_pretrained(self):
        if self.finetuning_file:
            print(f'Pre-trained model loaded from: {self.finetuning_file}')
            checkpoint = torch.load(self.finetuning_file, map_location='cpu')
            model_dict = self.model.state_dict()
            self.load_layers(checkpoint['model_weights'], model_dict)
            self.check_ckpt_params(checkpoint)
        else:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0

    def save_model(self, epoch: int, results_dir: str, best=False):
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
                    'optimizer': self.optimizer,
                    'train_metrics': self.train_metrics,
                    'test_metrics': self.test_metrics}, ckpt_file)

    def get_model_summary(self):
        summary(self.model, (1, 48, 48))

    def run(self) -> dict:
        self.check_pretrained()
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
        # self.get_model_summary()
        scheduler = self.get_callbacks()
        best_ssim = -1e9
        epoch_best_ssim = self.start_epoch
        for epoch in range(self.start_epoch + 1, self.epochs + self.start_epoch + 1):
            lr = self.get_lr(self.optimizer)
            print('Epoch: {} \t Learning rate {}'.format(epoch, lr))
            avg_train_metrics = self.run_epoch(scheduler, epoch)
            avg_test_metrics = self.run_epoch(scheduler, epoch, mode='test')

            if epoch >= self.callbacks['epoch_begin']:
                self.activate_callbacks(scheduler, avg_test_metrics['loss'])

            log = ''
            for metric in self.train_metrics.keys():
                self.train_metrics[metric].append(avg_train_metrics[metric])
                self.test_metrics[metric].append(avg_test_metrics[metric])
                log += f'Epoch train {metric}: {avg_train_metrics[metric]} \tEpoch test {metric}: {avg_test_metrics[metric]}\n'

            print(log)

            if self.epochs_per_save and (epoch % self.epochs_per_save) == 0:
                self.save_model(epoch, self.results_dir)

            if avg_test_metrics['loss'] <= min_loss:
                self.save_model(epoch, self.results_dir, best=True)
                min_loss = avg_test_metrics['loss']
                epoch_min_loss = epoch

        print(f'Best loss = {min_loss}, epoch = {epoch_min_loss}')
        # Add metrics to tensorboard
        for metric in self.train_metrics.keys():
            for i, (train_value, test_value) in enumerate(zip(self.train_metrics[metric], self.test_metrics[metric])):
                self.summary_writer.add_scalar(f'Per_epoch/train_{metric}', train_value, i)
                self.summary_writer.add_scalar(f'Per_epoch/test_{metric}', test_value, i)

        # Per class metrics to tensorboard
        if self.num_classes > 1:
            self.log_per_class()

        return self.test_metrics

    def get_callbacks(self) -> torch.optim.lr_scheduler:
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
            return OneCycleLR(self.optimizer, self.learning_rate,
                              epochs=self.epochs, steps_per_epoch=len(self.train_loader))
        else:
            return None

    def activate_callbacks(self, scheduler: torch.optim.lr_scheduler,
                           epoch_test_loss: float):
        if self.callbacks['exponential_lr']:
            scheduler.step()
        elif self.callbacks['plateau_learning_rate']:
            scheduler.step(epoch_test_loss)

    @torch.no_grad()
    def tensorboard_predictions(self, best_exp, results_dir):
        model = best_exp['model'].values[0]
        test_loader = best_exp['test_loader'].values[0]
        class_names = best_exp['class_names'].values[0]
        batch_size = best_exp['batch_size'].values[0]
        # Tensorboard object
        logdir = f'{results_dir}/log_predictions'
        os.makedirs(logdir)
        summary_writer = SummaryWriter(logdir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Get predictions
        for step, (batch, targets) in enumerate(test_loader):
            batch = batch.to(device)
            targets = targets.to(device)
            output = model(batch)
            preds = torch.sigmoid(output)
            for i, label in enumerate(class_names):
                targets_per_label = targets[:, i]
                preds_per_label = preds[:, i]
                sample = 0
                for pred, target in zip(preds_per_label, targets_per_label):
                    summary_writer.add_scalars(f'Best_model_predictions/{label}',
                                               {'target': target.item(), 'prediction': pred.item()},
                                               global_step=step * batch_size + sample)
                    sample += 1
