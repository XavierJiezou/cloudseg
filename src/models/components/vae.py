from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.plugin.ldm.modules.diffusionmodules.model import Encoder, Decoder
from src.plugin.ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        double_z: bool = True,
        z_channels: int = 3,
        resolution: int = 512,
        in_channels: int = 3,
        out_ch: int = 3,
        ch: int = 128,
        ch_mult: List = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_resolutions: List = [],
        dropout: float = 0.0,
        embed_dim: int = 3,
        ckpt_path: str = None,
        ignore_keys: List = [],
    ):
        super(AutoencoderKL, self).__init__()
        ddconfig = {
            "double_z": double_z,
            "z_channels": z_channels,
            "resolution": resolution,
            "in_channels": in_channels,
            "out_ch": out_ch,
            "ch": ch,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": attn_resolutions,
            "dropout": dropout
        }
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(
            2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)  # B, C, h, w
        moments = self.quant_conv(h)  # B, 6, h, w
        posterior = DiagonalGaussianDistribution(moments)
        return posterior  # 分布

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # 高斯分布
        if sample_posterior:
            z = posterior.sample()  # 采样
        else:
            z = posterior.mode()
        dec = self.decode(z)
        last_layer_weight = self.decoder.conv_out.weight
        return dec, posterior, last_layer_weight


if __name__ == '__main__':
    # Test the input and output shapes of the model
    model = AutoencoderKL()
    x = torch.randn(1, 3, 512, 512)
    dec, posterior, last_layer_weight = model(x)

    assert dec.shape == (1, 3, 512, 512)
    assert posterior.sample().shape == posterior.mode().shape == (1, 3, 64, 64)
    assert last_layer_weight.shape == (3, 128, 3, 3)

    # Plot the latent space and the reconstruction from the pretrained model
    model = AutoencoderKL(ckpt_path="/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/zouxuechao/Collaborative-Diffusion/outputs/512_vae/2024-06-27T06-02-04_512_vae/checkpoints/epoch=000036.ckpt")
    model.eval()
    image_path = "data/celeba/image/image_512_downsampled_from_hq_1024/0.jpg"

    from PIL import Image
    import numpy as np
    from src.data.components.celeba import DalleTransformerPreprocessor

    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.uint8)
    import copy
    original = copy.deepcopy(image)
    transform = DalleTransformerPreprocessor(size=512, phase='test')
    image = transform(image=image)['image']
    image = image.astype(np.float32)/127.5 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    dec, posterior, last_layer_weight = model(image)

    # original image
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    # sampled image from the latent space
    plt.subplot(1, 3, 2)
    x = model.decode(posterior.sample())
    x = (x+1)/2
    x = x.squeeze(0).permute(1, 2, 0).cpu()
    x = x.detach().numpy()
    x = x.clip(0, 1)
    x = (x*255).astype(np.uint8)
    plt.imshow(x)
    plt.title("Sampled")
    plt.axis("off")

    # reconstructed image
    plt.subplot(1, 3, 3)
    x = dec
    x = (x+1)/2
    x = x.squeeze(0).permute(1, 2, 0).cpu()
    x = x.detach().numpy()
    x = x.clip(0, 1)
    x = (x*255).astype(np.uint8)
    plt.imshow(x)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("vae_reconstruction.png")
