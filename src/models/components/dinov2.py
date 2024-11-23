import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models
from torch.nn import functional as F


dino_backbones = {
    "dinov2_s": {"name": "dinov2_vits14", "embedding_size": 384, "patch_size": 14},
    "dinov2_b": {"name": "dinov2_vitb14", "embedding_size": 768, "patch_size": 14},
    "dinov2_l": {"name": "dinov2_vitl14", "embedding_size": 1024, "patch_size": 14},
    "dinov2_g": {"name": "dinov2_vitg14", "embedding_size": 1536, "patch_size": 14},
}


class linear_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class conv_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5):
        super(conv_head, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 64, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        x = torch.sigmoid(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self, num_classes, backbone="dinov2_s", head="linear", backbones=dino_backbones
    ):
        super(Classifier, self).__init__()
        self.heads = {"linear": linear_head}
        self.backbones = dino_backbones
        self.backbone = load(
            "facebookresearch/dinov2", self.backbones[backbone]["name"]
        )
        self.backbone.eval()
        self.head = self.heads[head](
            self.backbones[backbone]["embedding_size"], num_classes
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x


class DINOv2(nn.Module):
    def __init__(
        self, num_classes, backbone="dinov2_s", head="conv", backbones=dino_backbones
    ):
        super().__init__()
        self.heads = {"conv": conv_head}
        self.backbones = dino_backbones
        self.backbone = load(
            "facebookresearch/dinov2", self.backbones[backbone]["name"]
        )
        self.backbone.eval()
        self.num_classes = num_classes  # add a class for background if needed
        self.embedding_size = self.backbones[backbone]["embedding_size"]
        self.patch_size = self.backbones[backbone]["patch_size"]
        self.head = self.heads[head](self.embedding_size, self.num_classes)

    def forward(self, x):
        batch_size, _, H, W = x.shape

        x = F.interpolate(
            x,
            size=(14*64,14*64),
            mode="bilinear",
            align_corners=False
        )

        # batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)
        with torch.no_grad():
            x = self.backbone.forward_features(x)
            x = x["x_norm_patchtokens"]
            x = x.permute(0, 2, 1)
            x = x.reshape(
                batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1])
            )
        x = self.head(x)

        x = F.interpolate(
            x,
            size=(H,W),
            mode="bilinear",
            align_corners=False
        )

        return x


if __name__ == "__main__":
    device = "cuda:0"
    model = DINOv2(num_classes=4).to(device)
    fake_data = torch.randn((2, 3, 512, 512)).to(device)
    out = model(fake_data)
    print(out.shape)
