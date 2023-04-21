# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from collections import OrderedDict

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import (
    SENet,
    SEResNetBottleneck,
    SEBottleneck,
    SEResNeXtBottleneck,
)
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.resnet_cbam import ResidualNet
from .backbones.resnext_ibn import resnext50_ibn_a, resnext101_ibn_a
from .backbones.hrnet import HighResolutionNet, get_cls_net



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(
        self,
        num_classes,
        last_stride,
        model_path,
        neck,
        neck_feat,
        model_name,
        pretrain_choice,
        cfg=None,
    ):
        super(Baseline, self).__init__()
        if model_name == "resnet18":
            self.in_planes = 512
            self.base = ResNet(
                last_stride=last_stride, block=BasicBlock, layers=[2, 2, 2, 2]
            )
        elif model_name == "resnet34":
            self.in_planes = 512
            self.base = ResNet(
                last_stride=last_stride, block=BasicBlock, layers=[3, 4, 6, 3]
            )
        elif model_name == "resnet50":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3]
            )
        elif model_name == "resnet101":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 4, 23, 3]
            )
        elif model_name == "resnet152":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 8, 36, 3]
            )

        elif model_name == "se_resnet50":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 6, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnet101":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 23, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnet152":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 8, 36, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnext50":
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 6, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnext101":
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 23, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "senet154":
            self.base = SENet(
                block=SEBottleneck,
                layers=[3, 8, 36, 3],
                groups=64,
                reduction=16,
                dropout_p=0.2,
                last_stride=last_stride,
            )
        elif model_name == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride)
        elif model_name == "Resnet_CBAM":
            self.base = ResidualNet(50, att_type="CBAM")
            print("using Resnet_CBAM as a backbone")
        elif model_name == "resnext101_ibn_a":
            self.in_planes = 2048
            self.base = resnext101_ibn_a()
            print("using resnext101_ibn_a as a backbone")
        elif  "cls_hrnet" in model_name:
            self.in_planes = 2048
            self.base = HighResolutionNet(cfg)
            print("using cls_hrnet as a backbone")

        if pretrain_choice == "imagenet":
            if model_name == "Resnet_CBAM":
                checkpoint = torch.load(model_path)
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.base.load_state_dict(new_state_dict)
            else:
                self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......")

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == "no":
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == "bnneck":
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(
            global_feat.shape[0], -1
        )  # flatten to (bs, 2048)

        if self.neck == "no":
            feat = global_feat
        elif self.neck == "bnneck":
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        cls_score = self.classifier(feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == "after":
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if "classifier" in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}
