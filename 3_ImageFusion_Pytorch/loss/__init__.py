import torch

from .ssim import SSIM
from .AdversarialLoss import AdversarialLoss
from .loss import StyleLoss, PerceptualLoss


def ssim(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    model = SSIM().to(input.device)
    x = model(input, target)
    return x
