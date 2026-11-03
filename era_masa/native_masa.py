"""MASA-R50 inference on modern PyTorch without compiling MMCV CUDA extensions.

Load the original MMDetection ResNet/FPN and MASA DeformFusion source. The only
learned operator substitution is torchvision deform_conv2d for MMCV DCNv2.
ROIAlign uses torchvision's aligned implementation, supported by MMCV itself.
All backbone, adapter and embedding-head checkpoint keys must load strictly.
This is an explicit inference backend, not a different appearance network.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import types
import typing
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d, roi_align
from mmcv.cnn import ConvModule
from .data import sha256


class TorchvisionDCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deform_groups=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride, self.padding, self.dilation = stride, padding, dilation

    def forward(self, image, offset, mask):
        height = (image.shape[2] + 2 * self.padding - self.dilation * (self.weight.shape[2] - 1) - 1) // self.stride + 1
        width = (image.shape[3] + 2 * self.padding - self.dilation * (self.weight.shape[3] - 1) - 1) // self.stride + 1
        if offset.shape[-2:] != (height, width):
            # Original MASA's high-resolution offset is passed to a smaller
            # feature map before upsampling (fix_upsample=False). MMCV 2.1's
            # CUDA kernel indexes the flat offset/mask pointers with computed
            # output H/W, without using their actual spatial dimensions. Match
            # that precise legacy behavior; resizing offsets changes the model.
            # Source: mmcv/v2.1.0/ops/csrc/pytorch/modulated_deform_conv.cpp
            # and common/cuda/modulated_deform_conv_cuda_kernel.cuh.
            if image.shape[0] != 1:
                raise ValueError("Legacy MASA DCNv2 requires batch size one")
            count = height * width
            offset = offset.reshape(-1)[:offset.shape[1] * count].reshape(1, offset.shape[1], height, width)
            mask = mask.reshape(-1)[:mask.shape[1] * count].reshape(1, mask.shape[1], height, width)
        return deform_conv2d(image, offset, self.weight, self.bias,
            stride=(self.stride, self.stride), padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation), mask=mask)


def _source_module(name, path, replacements=(), namespace=None):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update({k: typing.Any for k in ("ConfigType", "OptConfigType", "OptMultiConfig", "MultiConfig")})
    if namespace:
        module.__dict__.update(namespace)
    source = Path(path).read_text(encoding="utf-8")
    for before, after in replacements:
        if before not in source:
            raise RuntimeError(f"Source contract changed in {path}: expected {before}")
        source = source.replace(before, after)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@lru_cache(maxsize=1)
def original_classes(masa_root):
    root = Path(importlib.util.find_spec("mmdet").origin).parent
    layer = _source_module("_era_reslayer", root / "models/layers/res_layer.py",
        [("from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig", "")])
    resnet = _source_module("_era_resnet", root / "models/backbones/resnet.py",
        [("from ..layers import ResLayer", "")], {"ResLayer": layer.ResLayer})
    fpn = _source_module("_era_fpn", root / "models/necks/fpn.py",
        [("from mmdet.utils import ConfigType, MultiConfig, OptConfigType", "")])
    deform_path = Path(masa_root) / "masa/models/necks/deform_fusion.py"
    deform = _source_module("_era_deformfusion", deform_path,
        [("from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d", "")],
        {"ModulatedDeformConv2d": TorchvisionDCNv2})
    provenance = {str(p): sha256(p) for p in (root / "models/layers/res_layer.py",
        root / "models/backbones/resnet.py", root / "models/necks/fpn.py", deform_path)}
    return resnet.ResNet, fpn.FPN, deform.DeformFusion, provenance


class EmbeddingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([ConvModule(256, 256, 3, padding=1,
                                    norm_cfg=dict(type="GN", num_groups=32)) for _ in range(4)])
        self.fcs = nn.ModuleList([nn.Linear(256 * 7 * 7, 1024)])
        self.fc_embed = nn.Linear(1024, 256)

    def forward(self, features):
        for conv in self.convs:
            features = conv(features)
        features = features.flatten(1)
        return self.fc_embed(F.relu(self.fcs[0](features)))


class NativeMasaR50(nn.Module):
    def __init__(self, checkpoint, masa_root, device="cuda:0"):
        super().__init__()
        ResNet, FPN, DeformFusion, self.source_provenance = original_classes(str(Path(masa_root).resolve()))
        self.backbone = ResNet(depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
            frozen_stages=-1, norm_cfg=dict(type="SyncBN", requires_grad=True), norm_eval=True, style="caffe")
        self.masa_adapter = nn.Sequential(
            FPN(in_channels=[256, 512, 1024, 2048], out_channels=256,
                norm_cfg=dict(type="SyncBN", requires_grad=True), num_outs=5),
            DeformFusion(in_channels=256, out_channels=256, num_blocks=3))
        self.track_head = nn.Module()
        self.track_head.embed_head = EmbeddingHead()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        needed = {k: v for k, v in state.items() if k.startswith(("backbone.", "masa_adapter.", "track_head.embed_head."))}
        self.load_state_dict(needed, strict=True)
        self.loaded_keys = len(needed)
        self.checkpoint_hash = sha256(checkpoint)
        self.to(device).eval()
        self.device = torch.device(device)
        # Preserve float32 arithmetic; TF32 changes are not part of the method.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def preprocess(self, bgr, boxes):
        h, w = bgr.shape[:2]
        ratio = 1024 / max(h, w)
        new_w, new_h = int(w * ratio + 0.5), int(h * ratio + 0.5)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(resized[:, :, ::-1]).astype(np.float32)
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
        mean = image.new_tensor([123.675, 116.28, 103.53])[:, None, None]
        std = image.new_tensor([58.395, 57.12, 57.375])[:, None, None]
        image = (image - mean) / std
        image = F.pad(image, (0, (-new_w) % 32, 0, (-new_h) % 32)).unsqueeze(0)
        rois = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        rois = rois * rois.new_tensor([new_w / w, new_h / h, new_w / w, new_h / h])
        rois = torch.cat([rois.new_zeros((len(rois), 1)), rois], dim=1)
        return image, rois

    @torch.inference_mode()
    def extract(self, image, boxes):
        if len(boxes) == 0:
            return np.empty((0, 256), dtype=np.float32)
        inputs, rois = self.preprocess(image, boxes)
        features = self.masa_adapter(self.backbone(inputs))
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        levels = torch.floor(torch.log2(scale / 56 + 1e-6)).clamp(0, 3).long()
        pooled = features[0].new_zeros((len(boxes), 256, 7, 7))
        for level, stride in enumerate((4, 8, 16, 32)):
            indices = torch.where(levels == level)[0]
            if len(indices):
                pooled[indices] = roi_align(features[level], rois[indices], (7, 7),
                    spatial_scale=1 / stride, sampling_ratio=0, aligned=True)
        return self.track_head.embed_head(pooled).cpu().numpy()
