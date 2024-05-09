import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

from src.modeling.base import BaseModel


class BaseNoMaxPoolAutoEncoderv2(BaseModel):
    def __init__(self, model_name='base_ae_no_maxpool_v2'):
        super(BaseNoMaxPoolAutoEncoderv2, self).__init__(model_name)
        self.encoder = nn.Sequential(
            # 128x128x3
            nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            # 64x64x128
            nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=2),
            # 32x32x64
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),
            # 16x16x32
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2),
            # 8x8x16
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),
        )


    def forward(self, x, b_t=None):
        x = self.encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.decoder(x)
        return x