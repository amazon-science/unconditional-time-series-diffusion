# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Tuple
from functools import partial

import numpy as np
import torch


def grad_fn(fn, x):
    x.requires_grad_(True)
    return torch.autograd.grad(fn(x), x)[0]


@torch.no_grad()
def langevin_dynamics(
    z0: torch.Tensor,
    energy_func: Callable = None,
    score_func: Callable = None,
    step_size: float = 0.1,
    noise_scale: float = 0.1,
    n_steps: int = 1,
):
    """Overdamped Langevin dynamics.

    Parameters
    ----------
    z0
        Initial guess.
    energy_func, optional
        Energy function, only one of energy function or score function
        must be specified, by default None
    score_func, optional
        Score function, only one of energy function or score function
        must be specified, by default None
    step_size, optional
        Step size, by default 0.1
    noise_scale, optional
        Scale for Brownian noise, by default 0.1
    n_steps, optional
        Number of Langevin steps, by default 1

    Returns
    -------
        Updated point.
    """
    assert energy_func is not None or score_func is not None
    z = z0
    sqrt_2eta = torch.sqrt(2 * torch.tensor(step_size))
    for _ in range(n_steps):
        if energy_func is not None:
            with torch.enable_grad():
                z.requires_grad_(True)
                Ez = energy_func(z)
                v = -torch.autograd.grad(Ez, z)[0]
        else:
            v = score_func(z)
        z = (
            z.detach()
            + step_size * v
            + sqrt_2eta * noise_scale * torch.randn_like(z)
        )
    return z


@torch.enable_grad()
def leapfrog(
    xt: torch.Tensor,
    pt: torch.Tensor,
    dynamics_p: Callable[[torch.Tensor], torch.Tensor],
    mass: float,
    h: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Leapfrong integrator.

    Parameters
    ----------
    xt
        Position.
    pt
        Momentum.
    dynamics_p
        Dynamics function for momentum
    mass
        Mass of particle
    h
        Step size
    n_steps
        Number of leapfrog integration steps

    Returns
    -------
        Updated position and momentum.
    """
    for _ in range(n_steps):
        pt = pt - (h / 2) * dynamics_p(xt)
        xt = xt + h * pt / mass
        pt = pt - (h / 2) * dynamics_p(xt)
        xt, pt = xt.detach(), pt.detach()

    return xt, pt


@torch.no_grad()
def hmc(
    x0: torch.Tensor,
    energy_func: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    mass: float,
    n_leapfrog_steps: int = 10,
    n_steps: int = 100,
) -> torch.Tensor:
    """Hamiltonian Monte Carlo.

    Parameters
    ----------
    x0
        Initial guess of shape [B, T, C].
    energy_func
        Energy function E: [B, T, C] -> []
    step_size
        Step size.
    mass
        Mass of particle.
    n_leapfrog_steps, optional
        Number of leapfrog integration steps, by default 10
    n_steps, optional
        Number of HMC steps, by default 100

    Returns
    -------
        Updated tensor of shape [B, T, C].
    """
    potential_energy_func = energy_func
    batch_size, length, ch = x0.shape

    drift_func = partial(grad_fn, potential_energy_func)
    xt = x0
    for _ in range(n_steps):
        pt = np.sqrt(mass) * torch.randn_like(xt)
        xt_prop, pt_prop = leapfrog(
            xt, pt, drift_func, mass, step_size, n_leapfrog_steps
        )
        xt = xt_prop

    return xt


def linear_midpoint_em_step(
    zt: torch.Tensor, coeff: float, h: float, sigma: float
):
    """Midpoint Euler-Maruyama step."""
    eta = torch.randn_like(zt)
    ztp1 = zt - h * coeff * zt / 2 + np.sqrt(h) * sigma * eta
    ztp1 = ztp1 / (1 + h * coeff / 2)
    return ztp1.detach()


@torch.no_grad()
def udld(
    x0: torch.Tensor,
    potential_energy_func: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    friction: float,
    mass: float,
    n_leapfrog_steps: int = 1,
    n_steps: int = 100,
) -> torch.Tensor:
    """Underdamped Langevin dynamics.

    Parameters
    ----------
    x0
        Initial guess of shape [B, T, C]
    potential_energy_func
        Energy function E: [B, T, C] -> []
    step_size
        Step size
    friction
        Friction coefficient
    mass
        Mass of the particle
    n_leapfrog_steps, optional
        Number of leapfrog integration steps, by default 1
    n_steps, optional
        Number of UDLD steps, by default 100

    Returns
    -------
         Updated tensor of shape [B, T, C].
    """
    batch_size, length, ch = x0.shape
    xt = x0
    drift_func = partial(grad_fn, potential_energy_func)

    pt = np.sqrt(mass) * torch.randn_like(xt)

    coeff = friction / mass
    sigma = np.sqrt(2 * friction)
    for _ in range(n_steps):
        pt = linear_midpoint_em_step(pt, coeff, step_size / 2, sigma)
        xt_prop, pt_prop = leapfrog(
            xt, pt, drift_func, mass, step_size, n_leapfrog_steps
        )
        xt, pt = xt_prop, pt_prop
        pt = linear_midpoint_em_step(pt, coeff, step_size / 2, sigma)

    return xt
