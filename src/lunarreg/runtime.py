"""Local-only model loading and explicit device selection."""
import importlib
import logging
import random
from pathlib import Path
import numpy as np


def seed_all(seed, *, deterministic=True):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    import cv2
    cv2.setRNGSeed(seed)
    torch.set_float32_matmul_precision('highest')
    # Newer PyTorch builds lazily import Dynamo/FSDP here. Keep V9's legacy
    # reproducibility contract, while blind runs avoid that unrelated startup
    # path; their sampling and hypothesis RNGs are still explicitly seeded.
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(name='auto'):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        logging.warning('MPS unavailable; using CPU')
        name = 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        logging.warning('CUDA unavailable; using CPU')
        name = 'cpu'
    return torch.device(name)


def checkpoint():
    import torch
    p=Path(torch.hub.get_dir())/'checkpoints/romav2.0.1.pt'
    if not p.is_file():
        raise FileNotFoundError('RoMa checkpoint is not cached. Install/cache RoMa weights explicitly before running.')
    return p


def configure_roma_device(d):
    import sys
    importlib.import_module('romav2')
    # Upstream modules capture device by value at import time. Update every
    # loaded RoMa module, including matcher/refiner, before constructing tensors.
    for name, module in tuple(sys.modules.items()):
        if name.startswith('romav2.') and hasattr(module, 'device'):
            module.device=d


def load_model(mode='fast', device='auto'):
    import torch
    checkpoint()
    cache=Path(torch.hub.get_dir())/'facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa'
    if not cache.is_dir():
        raise FileNotFoundError('Pinned DINOv3 repository is not cached; no automatic download is attempted.')
    d=select_device(device)
    configure_roma_device(d)
    from romav2 import RoMaV2
    from romav2.features import Descriptor
    cfg=RoMaV2.Cfg(descriptor=Descriptor.Cfg(enable_amp=d.type=='cuda'))
    model=RoMaV2(cfg)
    model.apply_setting(mode)
    model.bidirectional=True
    model.balanced_sampling=True
    logging.info('RoMa mode=%s device=%s; upstream CUDA-autocast warning on CPU/MPS is harmless',mode,d)
    return model


def fine_extractor(device='auto'):
    """Load only trained fine-feature weights, avoiding DINO and matcher allocation."""
    import torch
    from romav2.features import FineFeatures
    model=FineFeatures(FineFeatures.Cfg())
    state=torch.load(checkpoint(),map_location='cpu',weights_only=True,mmap=True)
    state={k.removeprefix('refiner_features.'):v for k,v in state.items() if k.startswith('refiner_features.')}
    model.load_state_dict(state,strict=True)
    return model.to(select_device(device)).eval()
