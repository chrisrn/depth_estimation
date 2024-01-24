import torch.nn as nn
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='resnet18',
            in_channels=3,
            classes=1
        )

    def trainable_encoder(self, trainable=True):
        for p in self.model.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

    def _num_params(self, ):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])