# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.time_feature import get_seasonality

from uncond_ts_diff.predictor import PyTorchPredictorWGrads
from uncond_ts_diff.sampler._base import (
    langevin_dynamics,
    hmc,
    udld,
)

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "future_time_feat",
    "stats",
]


class Refiner(torch.nn.Module):
    def __init__(
        self,
        model,
        prediction_length,
        fixed_t=20,
        iterations=1,
        init=None,
        num_samples=1,
        guidance="quantile",
        scale=1,
    ):
        super().__init__()
        self.model = model
        self.prediction_length = prediction_length
        self.fixed_t = fixed_t
        self.iterations = iterations
        self.init = init
        self.num_samples = num_samples
        self.guidance = guidance
        self.scale = scale

    def quantile_loss(self, y_prediction, y_target):
        assert y_target.shape == y_prediction.shape
        device = y_prediction.device
        batch_size_x_num_samples, length, ch = y_target.shape
        batch_size = batch_size_x_num_samples // self.num_samples
        # num_samples uniformly distributed quantiles between 0 and 1
        # repeat for each element in the batch
        q = (torch.arange(self.num_samples).repeat(batch_size) + 1).to(
            device
        ) / (self.num_samples + 1)
        # (batch_size x num_samples,)
        q = q[:, None, None]
        # (batch_size x num_samples, 1, 1)
        e = y_target - y_prediction
        loss = torch.max(q * e, (q - 1) * e)
        return loss

    def prior(self, y_prediction, obs, obs_mask):
        if self.guidance == "MSE":
            return (
                self.scale
                * F.mse_loss(y_prediction, obs, reduction="none")[
                    obs_mask == 1
                ].sum()
            )
        elif self.guidance == "quantile":
            return self.scale * self.quantile_loss(y_prediction, obs).sum()
        else:
            raise ValueError(f"Unknown guidance {self.guidance}!")

    def refine(self, observation, observation_mask):
        raise NotImplementedError("Must be implemented by a subclass!")

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        feat_static_cat: torch.Tensor = None,
        feat_static_real: torch.Tensor = None,
        past_time_feat: torch.Tensor = None,
        future_time_feat: torch.Tensor = None,
        stats: torch.Tensor = None,
    ):
        device = next(self.model.backbone.parameters()).device
        data = dict(
            feat_static_cat=feat_static_cat.to(device)
            if feat_static_cat is not None
            else None,
            feat_static_real=feat_static_real.to(device)
            if feat_static_real is not None
            else None,
            past_time_feat=past_time_feat.to(device)
            if past_time_feat is not None
            else None,
            past_target=past_target.to(device),
            future_target=torch.zeros(
                past_target.shape[0], self.prediction_length, device=device
            ),
            past_observed_values=past_observed_values.to(device)
            if past_observed_values is not None
            else None,
            future_time_feat=future_time_feat.to(device)
            if future_time_feat is not None
            else None,
        )

        observation, scale, features = self.model._extract_features(data)

        observation = observation.to(device)
        batch_size, length, ch = observation.shape
        observation_mask = torch.ones_like(observation, device=device)
        observation_mask[:, length - self.prediction_length :, 0] = 0

        observation = observation.repeat_interleave(self.num_samples, dim=0)
        observation_mask = observation_mask.repeat_interleave(
            self.num_samples, dim=0
        )
        if features is not None:
            features = features.repeat_interleave(self.num_samples, dim=0)

        if self.init is not None:
            init_forecasts = np.stack(
                [next(self.init).samples for _ in range(batch_size)]
            )

            if init_forecasts.shape[1] == 1:
                # Single sample, e.g., for SeasonalNaive
                init_forecasts = np.tile(
                    init_forecasts, (1, self.num_samples, 1)
                )

            # create numpy array out of list and sort them to
            # match to their corresponding quantile
            init = np.sort(init_forecasts, axis=1)
            init = torch.from_numpy(init).to(device)

            # scale input
            init = init / scale

            # reshape from B x num_samples x prediction_length to
            # B * self.num_samples x prediction_length
            init = init.reshape(
                batch_size * self.num_samples, self.prediction_length
            )

            # use it as initial guess
            observation[:, length - self.prediction_length :, 0] = init

        else:
            season_length = get_seasonality(self.model.freq)

            # Initialize using Seasonal Naive predictions
            if (length - self.prediction_length) >= season_length:
                indices = [
                    length
                    - self.prediction_length
                    - season_length
                    + k % season_length
                    for k in range(self.prediction_length)
                ]
                observation[
                    :, length - self.prediction_length :, 0
                ] = observation[:, indices, 0]

            # Initialize using the meant of the context length
            else:
                observation[
                    :, length - self.prediction_length :, 0
                ] = torch.mean(
                    observation[:, : length - self.prediction_length, 0],
                    dim=1,
                    keepdim=True,
                )

        pred = self.refine(observation, observation_mask)

        pred = pred[:, :, 0].reshape(batch_size, self.num_samples, -1)
        pred = pred * scale

        return pred[:, :, length - self.prediction_length :]

    def get_predictor(self, input_transform, batch_size=40, device=None):
        return PyTorchPredictorWGrads(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )


class MostLikelyRefiner(Refiner):
    def __init__(
        self,
        model,
        prediction_length,
        lr=1e-1,
        patience=100,
        fixed_t=20,
        iterations=1,
        init=None,
        num_samples=1,
        guidance="quantile",
        scale=1,
    ):
        super().__init__(
            model,
            prediction_length,
            fixed_t,
            iterations,
            init,
            num_samples,
            guidance,
            scale,
        )
        self.lr = lr
        self.patience = patience

    def _most_likely(self, observation, observation_mask):
        device = next(self.model.backbone.parameters()).device
        observation = observation.to(device)
        seq = nn.Parameter(torch.clone(observation), requires_grad=True)
        optim = torch.optim.SGD([seq], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, "min", patience=self.patience, factor=0.5
        )
        with torch.enable_grad():
            for i in range(self.iterations):
                optim.zero_grad()
                t = torch.randint(
                    0, self.model.timesteps, (seq.shape[0],), device=device
                ).long()
                if self.fixed_t != -1:
                    t = t * 0 + self.fixed_t
                loss = self.model.p_losses(
                    seq, t, loss_type="l2", reduction="sum"
                )[0] + self.prior(seq, observation, observation_mask)
                loss.backward()

                optim.step()
                scheduler.step(loss.item())

        return seq.detach()

    def refine(self, observation, observation_mask):
        return self._most_likely(observation, observation_mask)


class MCMCRefiner(Refiner):
    _available_methods = {"lmc", "hmc", "udld", "cdld"}

    def __init__(
        self,
        model,
        prediction_length,
        step_size=1e-1,
        method="lmc",
        method_kwargs={},
        fixed_t=20,
        iterations=1,
        init=None,
        num_samples=1,
        guidance="quantile",
        scale=1,
    ):
        super().__init__(
            model,
            prediction_length,
            fixed_t,
            iterations,
            init,
            num_samples,
            guidance,
            scale,
        )
        assert method in self._available_methods
        self.step_size: float = step_size
        self.method: str = method
        self.method_kwargs: dict = method_kwargs

    def _mcmc(self, observation, observation_mask):
        device = next(self.model.backbone.parameters()).device
        observation = observation.to(device)
        seq = torch.clone(observation)

        for i in range(self.iterations):
            t = torch.randint(
                0, self.model.timesteps, (seq.shape[0],), device=device
            ).long()
            if self.fixed_t != -1:
                t = t * 0 + self.fixed_t

            energy_func = lambda x: self.model.p_losses(  # noqa: E731
                x, t, loss_type="l2", reduction="sum"
            )[0] + self.prior(x, observation, observation_mask)

            if self.method == "lmc":
                method_kwargs = {
                    "noise_scale": 0.1,
                    "n_steps": 1,
                }
                method_kwargs.update(self.method_kwargs)
                seq = langevin_dynamics(
                    seq,
                    energy_func,
                    score_func=None,
                    step_size=self.step_size,
                    **self.method_kwargs,
                )
            elif self.method == "hmc":
                method_kwargs = {
                    "mass": 1.0,
                    "n_steps": 1,
                    "n_leapfrog_steps": 5,
                }
                method_kwargs.update(self.method_kwargs)
                seq = hmc(
                    seq, energy_func, step_size=self.step_size, **method_kwargs
                )
            elif self.method == "udld":
                method_kwargs = {
                    "mass": 1.0,
                    "friction": 1.0,
                    "n_steps": 1,
                    "n_leapfrog_steps": 5,
                }
                method_kwargs.update(self.method_kwargs)
                seq = udld(
                    seq, energy_func, step_size=self.step_size, **method_kwargs
                )
            elif self.method == "cdld":
                method_kwargs = {
                    "mass": 1.0,
                    "n_steps": 1,
                    "n_leapfrog_steps": 5,
                }
                method_kwargs.update(self.method_kwargs)
                # friction^2 = 4 x mass
                method_kwargs["friction"] = np.sqrt(4 * method_kwargs["mass"])
                seq = udld(
                    seq, energy_func, step_size=self.step_size, **method_kwargs
                )

        return seq.detach()

    def refine(self, observation, observation_mask):
        return self._mcmc(observation, observation_mask)
