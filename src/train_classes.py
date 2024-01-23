import numpy as np
import pandas as pd
import torch

from ml_pipeline.train_utils import ModelHandler
from ml_pipeline.networks.Vgg import VGG
from ml_pipeline.networks.InceptionV3 import Inception3
from ml_pipeline.networks.Resnets import resnet50, resnet18
from ml_pipeline.networks.EfficientNet import EfficientNet
from ml_pipeline.networks.CustomNN import ArkitCNN
from ml_pipeline.networks.LandmarksCNN import KeypointsNet
from ml_pipeline.networks.LightCNN import LightCNN_9Layers
from ml_pipeline.networks.mobilenetv2 import MobileNetV2
from ml_pipeline.networks.mobilenetv3 import MobileNetV3
from ml_pipeline.networks.mobilenetv3 import _mobilenet_v3_conf
from ml_pipeline.networks.MoGA import MoGaA
from ml_pipeline.networks.inception_resnet_v1 import InceptionResnetV1
from ml_pipeline.networks.shufflenet2 import ShuffleNet
from torch.utils.data import DataLoader
from torch import Tensor
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn


class VggFerModelHandler(ModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)
        self.classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.num_classes = len(self.classes)
        self.model = VGG('VGG19', self.num_classes).to(self.device)
        self.loss_function = F.cross_entropy
        self.train_metrics = {'loss': [], 'accuracy': []}
        self.test_metrics = {'loss': [], 'accuracy': []}
        self.per_class_metrics = pd.DataFrame(columns=self.classes)
        self.per_class_correct = pd.DataFrame(0.0, columns=self.classes, index=['correct', 'total'])

    @torch.no_grad()
    def evaluation_step(self, batch: Tensor, target: Tensor, epoch: int) -> tuple:

        batch = batch.to(self.device)
        target = target.to(self.device)
        bs, ncrops, c, h, w = np.shape(batch)
        batch = batch.view(-1, c, h, w)
        output = self.model(batch)
        output = output.view(bs, ncrops, -1).mean(1)
        # calculate-the-batch-metrics
        batch_metrics = {}
        for metric in self.test_metrics.keys():
            batch_metrics[metric] = self.get_metric(metric, output, target).item()

        self.evaluate_per_class(output, target, epoch)

        return batch_metrics, output

    def get_model_summary(self):
        summary(self.model, (1, 44, 44))


class VggAUModelHandler(ModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)
        self.classes = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9',
                        'AU12', 'AU15', 'AU16', 'AU20', 'AU23', 'AU26',
                        'AUNeutral']

        self.num_classes = len(self.classes)
        self.model = VGG('VGG19', self.num_classes).to(self.device)
        self.loss_function = self.sigmoid_cross_entropy_with_logits
        self.train_metrics = {'loss': [], 'accuracy': []}
        self.test_metrics = {'loss': [], 'accuracy': []}

    def sigmoid_cross_entropy_with_logits(self, output: Tensor, target: Tensor) -> Tensor:
        sigmoid = nn.Sigmoid()
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(sigmoid(output), target)

    def accuracy(self, outputs: Tensor, labels: Tensor) -> Tensor:
        preds = torch.sigmoid(outputs)
        preds[preds >= 0.5] = 1
        return torch.tensor(torch.sum(preds == labels).item() / (preds.shape[0]*preds.shape[1]))

    def evaluate_per_class(self, output: Tensor, target: Tensor, epoch: int):
        preds = torch.sigmoid(output)
        preds[preds >= 0.5] = 1
        for label in range(self.num_classes):
            preds_label = preds[:, label]
            target_label = target[:, label]
            c = (preds_label == target_label).squeeze()
            class_name = self.classes[label]
            for i in range(len(target_label)):
                if target_label[i] == 1:
                    self.per_class_correct.loc['correct', class_name] += c[i].item()
                    self.per_class_correct.loc['total', class_name] += 1

    @torch.no_grad()
    def evaluation_step(self, batch: Tensor, target: Tensor, epoch: int) -> tuple:
        batch = batch.to(self.device)
        target = target.to(self.device)
        bs, ncrops, c, h, w = np.shape(batch)
        batch = batch.view(-1, c, h, w)
        output = self.model(batch)
        output = output.view(bs, ncrops, -1).mean(1)
        # calculate-the-batch-metrics
        batch_metrics = {}
        for metric in self.test_metrics.keys():
            batch_metrics[metric] = self.get_metric(metric, output, target).item()

        self.evaluate_per_class(output, target, epoch)

        return batch_metrics, output

    def get_model_summary(self):
        summary(self.model, (1, 44, 44))


class VggBlendModelHandler(ModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)
        self.classes = ['BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight',
                        'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'EyeBlinkLeft', 'EyeBlinkRight',
                        'EyeLookDownLeft', 'EyeLookDownRight', 'EyeLookInLeft', 'EyeLookInRight', 'EyeLookOutLeft',
                        'EyeLookOutRight', 'EyeLookUpLeft', 'EyeLookUpRight', 'EyeSquintLeft', 'EyeSquintRight',
                        'EyeWideLeft', 'EyeWideRight', 'JawForward', 'JawLeft', 'JawOpen', 'JawRight', 'MouthClose',
                        'MouthDimpleLeft', 'MouthDimpleRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthFunnel',
                        'MouthLeft', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthPressLeft', 'MouthPressRight',
                        'MouthPucker', 'MouthRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower',
                        'MouthShrugUpper', 'MouthSmileLeft', 'MouthSmileRight', 'MouthStretchLeft', 'MouthStretchRight',
                        'MouthUpperUpLeft', 'MouthUpperUpRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut']

        self.num_classes = len(self.classes)
        self.model = VGG('VGG19', self.num_classes).to(self.device)
        self.loss_function = self.sigmoid_mse
        self.train_metrics = {'loss': []}
        self.test_metrics = {'loss': []}

    def sigmoid_mse(self, output: Tensor, target: Tensor) -> Tensor:
        sigmoid = nn.Sigmoid()
        loss_fn = nn.MSELoss()
        return loss_fn(sigmoid(output), target)

    def evaluate_per_class(self, output: Tensor, target: Tensor, epoch: int):
        preds = torch.sigmoid(output)
        for label in range(self.num_classes):
            preds_label = preds[:, label]
            target_label = target[:, label]
            c = nn.MSELoss()(preds_label, target_label)
            class_name = self.classes[label]
            self.per_class_metrics.loc[epoch, class_name] = c.item()

    def log_per_class(self):
        for index, row in self.per_class_metrics.iterrows():
            for label in self.classes:
                self.summary_writer.add_scalar(f'Per_class_loss/{label}', row[label], index - 1)

    def get_model_summary(self):
        summary(self.model, (1, 48, 48))


class Inceptionv3BlendModelHandler(VggBlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)

        self.model = Inception3(num_classes=self.num_classes, aux_logits=False).to(self.device)

    def check_pretrained(self):
        if self.finetuning_file:
            print(f'Pre-trained model loaded from: {self.finetuning_file}')
            checkpoint = torch.load(self.finetuning_file, map_location=self.device)
            model_dict = self.model.state_dict()
            if 'model_weights' in checkpoint:
                self.load_layers(checkpoint['model_weights'], model_dict)
            else:
                self.load_layers(checkpoint, model_dict)
            self.check_ckpt_params(checkpoint)
        else:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0

    def get_model_summary(self):
        summary(self.model, (3, 299, 299))


class ResnetBlendModelHandler(Inceptionv3BlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter,
                 model_name: str):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)
        if model_name == 'resnet_50':
            self.model = resnet50(num_classes=self.num_classes).to(self.device)
        else:
            self.model = resnet18(num_classes=self.num_classes).to(self.device)

    def get_model_summary(self):
        summary(self.model, (3, 224, 224))


class EfficientnetBlendModelHandler(Inceptionv3BlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter,
                 model_name: str):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)
        if model_name == 'efficientnet-b0':
            self.model = EfficientNet.from_name(model_name=model_name,
                                                num_classes=self.num_classes).to(self.device)

    def get_model_summary(self):
        summary(self.model, (3, 224, 224))


class ArkitBlendModelHandler(VggBlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)

        self.model = ArkitCNN(cnn_name='cnn_2', num_classes=self.num_classes).to(self.device)

    def get_model_summary(self):
        summary(self.model, (1, 128, 128))


class LandmarksBlendModelHandler(VggBlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)

        self.model = KeypointsNet(num_classes=self.num_classes).to(self.device)

    def check_pretrained(self):
        if self.finetuning_file:
            print(f'Pre-trained model loaded from: {self.finetuning_file}')
            checkpoint = torch.load(self.finetuning_file, map_location=self.device)
            model_dict = self.model.state_dict()
            if 'model_weights' in checkpoint:
                self.load_layers(checkpoint['model_weights'], model_dict)
            else:
                self.load_layers(checkpoint, model_dict)
            self.check_ckpt_params(checkpoint)
        else:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0

    def get_model_summary(self):
        summary(self.model, (1, 224, 224))


class LightcnnBlendModelHandler(VggBlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)

        self.model = LightCNN_9Layers(num_classes=self.num_classes).to(self.device)

    def check_pretrained(self):
        if self.finetuning_file:
            print(f'Pre-trained model loaded from: {self.finetuning_file}')
            checkpoint = torch.load(self.finetuning_file, map_location=self.device)
            model_dict = self.model.state_dict()
            if 'model_weights' in checkpoint:
                self.load_layers(checkpoint['model_weights'], model_dict)
            else:
                self.load_layers(checkpoint, model_dict)
            self.check_ckpt_params(checkpoint)
        else:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0

    def get_model_summary(self):
        summary(self.model, (1, 128, 128))


class ArkitFerModelHandler(ModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)
        self.classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.num_classes = len(self.classes)
        self.model = ArkitCNN(cnn_name='cnn_2', num_classes=self.num_classes).to(self.device)
        self.loss_function = F.cross_entropy
        self.train_metrics = {'loss': [], 'accuracy': []}
        self.test_metrics = {'loss': [], 'accuracy': []}
        self.per_class_metrics = pd.DataFrame(columns=self.classes)
        self.per_class_correct = pd.DataFrame(0.0, columns=self.classes, index=['correct', 'total'])

    @torch.no_grad()
    def evaluation_step(self, batch: Tensor, target: Tensor, epoch: int) -> tuple:

        batch = batch.to(self.device)
        target = target.to(self.device)
        output = self.model(batch)
        # calculate-the-batch-metrics
        batch_metrics = {}
        for metric in self.test_metrics.keys():
            batch_metrics[metric] = self.get_metric(metric, output, target).item()

        self.evaluate_per_class(output, target, epoch)

        return batch_metrics, output, torch.mean(batch)

    def get_model_summary(self):
        summary(self.model, (1, 128, 128))


class MobilenetModelHandler(Inceptionv3BlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter,
                 model_name: str):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)

        if model_name == 'mobilenet_v2':
            self.model = MobileNetV2(num_classes=self.num_classes).to(self.device)
        else:
            inverted_residual_setting, last_channel = _mobilenet_v3_conf(model_name)
            self.model = MobileNetV3(inverted_residual_setting, last_channel, num_classes=self.num_classes).to(self.device)

    def get_model_summary(self):
        summary(self.model, (3, 224, 224))


class MoGAModelHandler(Inceptionv3BlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter,
                 model_name: str):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)

        self.model = MoGaA(n_class=self.num_classes).to(self.device)

    def get_model_summary(self):
        summary(self.model, (3, 224, 224))


class InceptionResnetv1ModelHandler(Inceptionv3BlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)

        self.model = InceptionResnetV1(num_classes=self.num_classes, classify=True).to(self.device)

    def get_model_summary(self):
        summary(self.model, (3, 224, 224))


class ShuffleNetv2ModelHandler(Inceptionv3BlendModelHandler):
    def __init__(self,
                 config: dict,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 summary_writer: SummaryWriter):
        super().__init__(config,
                         train_loader,
                         test_loader,
                         summary_writer)

        self.model = ShuffleNet(num_classes=self.num_classes).to(self.device)

    def get_model_summary(self):
        summary(self.model, (3, 224, 224))
