import os
import unittest

os.environ.setdefault("FWI_INIT", "poor")

import torch

import dl_timeshift as D


def _wavelet(n=700, t0=300.0, f=0.01, dtype=torch.float64):
    t = torch.arange(n, dtype=dtype)
    a = (torch.pi * f * (t - t0)) ** 2
    return ((1 - 2 * a) * torch.exp(-a))[None]


class TimeShiftTest(unittest.TestCase):
    def test_warp_delays_constant_shift(self):
        d = _wavelet()
        d2 = D.warp(d, torch.full_like(d, 25.0))
        self.assertEqual(int(d2.argmax()), int(d.argmax()) + 25)

    def test_random_shifts_monotone_warp(self):
        g = torch.Generator().manual_seed(0)
        tau = D.random_shifts(64, 700, g)
        self.assertLessEqual(float((tau[:, 1:] - tau[:, :-1]).abs().max()), 0.7 + 1e-5)
        self.assertGreater(float(tau.abs().max()), 20.0)

    def test_cc_shift_recovers_known_shift(self):
        d = _wavelet().to(D.dev)
        do = D.warp(d, torch.full_like(d, -17.3))
        self.assertAlmostEqual(float(D.cc_shift(d, do, 100)), -17.3, delta=0.3)

    def test_cc_gradient_matches_finite_difference(self):
        torch.manual_seed(0)
        d = (_wavelet() + 0.3 * _wavelet(t0=360.0, f=0.008)).to(D.dev)
        do = D.warp(d, torch.full_like(d, 12.0))
        x = d.clone().requires_grad_(True)
        J = 0.5 * D._CCShift.apply(x, do, 100).pow(2).sum()
        (g,) = torch.autograd.grad(J, x)
        v = torch.randn_like(d) * torch.exp(-((torch.arange(700, device=D.dev) - 320.0) / 80) ** 2)
        h = 1e-4
        Jp = 0.5 * D.cc_shift(d + h * v, do, 100).pow(2).sum()
        Jm = 0.5 * D.cc_shift(d - h * v, do, 100).pow(2).sum()
        fd = (Jp - Jm) / (2 * h)
        self.assertAlmostEqual(float((g * v).sum()), float(fd), delta=0.05 * abs(float(fd)) + 1e-6)

    def test_unet_shape_and_differentiable_input(self):
        net = D.TimeShiftUNet().to(D.dev).eval()
        a = torch.randn(3, 700, device=D.dev, requires_grad=True)
        tau = net(a, torch.randn(3, 700, device=D.dev))
        self.assertEqual(tuple(tau.shape), (3, 700))
        (g,) = torch.autograd.grad(tau.pow(2).sum(), a)
        self.assertTrue(torch.isfinite(g).all())


if __name__ == "__main__":
    unittest.main()
