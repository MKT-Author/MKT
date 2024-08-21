
import torch
import torch.nn as nn

from options import Option
from conditional_batchnorm import CategoricalConditionalBatchNorm2d

class Generator(nn.Module):
    def __init__(self, options=None, conf_path=None):
        super(Generator, self).__init__()
        self.settings = options or Option(conf_path)
        self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
        self.init_size = self.settings.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )

        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        self.last_bn = nn.BatchNorm2d(self.settings.channels, affine=False)


    def forward(self, z, labels, noise_out=False, emb=False, last_bn=True):
        label_emb_vec = self.label_emb(labels)
        gen_input = torch.mul(label_emb_vec, z)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        if last_bn:
            img = self.last_bn(img)
        else:
            return img

        if noise_out:
            return img, gen_input
        if emb:
            return img, label_emb_vec
        return img


class Generator_imagenet(nn.Module):
    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)

        super(Generator_imagenet, self).__init__()

        self.init_size = self.settings.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

        self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
        self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
        self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
        self.conv_blocks2_4 = nn.Tanh()
        self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

    def forward(self, z, labels, vis=False):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0_0(out, labels)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1_0(img)
        img = self.conv_blocks1_1(img, labels)
        img = self.conv_blocks1_2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2_0(img)
        img = self.conv_blocks2_1(img, labels)
        img = self.conv_blocks2_2(img)
        img = self.conv_blocks2_3(img)
        img = self.conv_blocks2_4(img)
        if vis == True:
            return img
        img = self.conv_blocks2_5(img)
        return img
