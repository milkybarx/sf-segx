"""Tests for the Super-Resolution subsystem."""
import numpy as np
import torch
import cv2
from experiments.super_resolution.models import (
    ESPCN, FSRCNN, EDSRSmall, SolarSRNet, CharbonnierLoss, SSIMLoss
)
from visualization.detail import crop_filament


def test_charbonnier_loss():
    loss_fn = CharbonnierLoss(eps=1e-3)
    pred = torch.ones(1, 1, 10, 10)
    target = torch.ones(1, 1, 10, 10)
    loss = loss_fn(pred, target)
    assert abs(loss.item() - 1e-3) < 1e-5


def test_ssim_loss():
    loss_fn = SSIMLoss(channels=1)
    pred = torch.rand(1, 1, 32, 32)
    target = pred.clone()
    loss = loss_fn(pred, target)
    assert loss.item() < 1e-3


def test_espcn_forward():
    model = ESPCN(scale_factor=2, in_channels=1)
    x = torch.rand(1, 1, 16, 16)
    out = model(x)
    assert out.shape == (1, 1, 32, 32)
    assert (out >= 0).all() and (out <= 1).all()


def test_fsrcnn_forward():
    model = FSRCNN(scale_factor=4, in_channels=1)
    x = torch.rand(1, 1, 16, 16)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)
    assert (out >= 0).all() and (out <= 1).all()


def test_edsr_small_forward():
    model = EDSRSmall(scale_factor=2, in_channels=1)
    x = torch.rand(1, 1, 16, 16)
    out = model(x)
    assert out.shape == (1, 1, 32, 32)
    assert (out >= 0).all() and (out <= 1).all()


def test_solar_sr_forward():
    model = SolarSRNet(scale_factor=4, in_channels=1)
    x = torch.rand(1, 1, 16, 16)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)
    assert (out >= 0).all() and (out <= 1).all()


def test_crop_filament_padding():
    image = np.zeros((100, 100), dtype=np.uint8)
    filament = {
        "bbox": {"x_min": 40, "y_min": 40, "x_max": 60, "y_max": 60}
    }
    crop, bounds = crop_filament(image, filament, padding=10)
    assert crop.shape == (40, 40)
    assert bounds == (30, 30, 70, 70)
