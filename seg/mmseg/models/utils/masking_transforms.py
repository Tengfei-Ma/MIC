# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
from matplotlib import pyplot as plt

from mmseg.ops import resize


def build_mask_generator(cfg):
    if cfg is None:
        return None
    t = cfg.pop('type')
    if t == 'block':
        return BlockMaskGenerator(**cfg)
    else:
        raise NotImplementedError(t)


class BlockMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        # mask0 = input_mask[0].permute(1, 2, 0)
        # mask1 = input_mask[1].permute(1, 2, 0)
        # # 显示当前图像
        # plt.imshow(mask0.detach().cpu().numpy())  # 转换张量形状为 (H, W, C)
        # plt.imshow(mask1.detach().cpu().numpy())  # 转换张量形状为 (H, W, C)
        # plt.axis('off')
        # plt.show()
        return imgs * input_mask
