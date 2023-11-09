# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import torch.nn.functional as F
from gluonts.torch.util import lagged_sequence_values

from uncond_ts_diff.predictor import PyTorchPredictorWGrads
from uncond_ts_diff.utils import extract
from uncond_ts_diff.model import TSDiff

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "future_time_feat",
    "stats",
]


class Guidance(torch.nn.Module):
    _missing_scenarios = ["none", "RM", "BM-B", "BM-E"]

    def __init__(
        self,
        model: TSDiff,
        prediction_length: int,
        scale: float = 1.0,
        num_samples: int = 1,
        guidance: str = "quantile",
        missing_scenario: str = "none",
        missing_values: int = 0,
    ):
        super().__init__()
        assert missing_scenario in self._missing_scenarios

        self.model = model
        self.prediction_length = prediction_length
        self.scale = scale
        self.num_samples = num_samples
        self.guidance = guidance
        self.missing_scenario = missing_scenario
        self.missing_values = missing_values

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

        q = q[:, None, None]  # (batch_size x num_samples, 1, 1)
        e = y_target - y_prediction
        loss = torch.max(q * e, (q - 1) * e)
        return loss

    def energy_func(self, y, t, observation, observation_mask, features):
        if self.guidance == "MSE":
            return F.mse_loss(
                self.model.fast_denoise(y, t, features),
                observation,
                reduction="none",
            )[observation_mask == 1].sum()
        elif self.guidance == "quantile":
            return self.quantile_loss(
                self.model.fast_denoise(y, t, features),
                observation,
            )[observation_mask == 1].sum()
        else:
            raise ValueError(f"Unknown guidance {self.guidance}!")

    def score_func(self, y, t, observation, observation_mask, features):
        with torch.enable_grad():
            y.requires_grad_(True)
            Ey = self.energy_func(
                y, t, observation, observation_mask, features
            )
            return -torch.autograd.grad(Ey, y)[0]

    def scale_func(self, y, t, base_scale):
        raise NotImplementedError("Must be implemented by a subclass!")

    def guide(self, observation, observation_mask, features, scale):
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
        device = next(self.model.parameters()).device

        future_target = torch.zeros(
            past_target.shape[0], self.prediction_length, device=device
        )
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
            future_target=future_target,
            past_observed_values=past_observed_values.to(device)
            if past_observed_values is not None
            else None,
            future_time_feat=future_time_feat.to(device)
            if future_time_feat is not None
            else None,
            stats=stats.to(device) if stats is not None else None,
        )

        observation, scale_params, features = self.model._extract_features(
            data
        )

        observation = observation.to(device)

        batch_size, length, ch = observation.shape
        prior_mask = past_observed_values[:, : -self.model.context_length]
        context_mask = past_observed_values[:, -self.model.context_length :]
        future_mask = torch.zeros_like(future_target)
        observation_mask = torch.cat([context_mask, future_mask], dim=1)
        if self.model.use_lags:
            lagged_mask = lagged_sequence_values(
                self.model.lags_seq,
                prior_mask,
                observation_mask,
                dim=1,
            )
            observation_mask = torch.cat(
                [observation_mask[:, :, None], lagged_mask], dim=-1
            )
        else:
            observation_mask = observation_mask[:, :, None]

        observation = observation.repeat_interleave(self.num_samples, dim=0)
        observation_mask = observation_mask.repeat_interleave(
            self.num_samples, dim=0
        )
        if features is not None:
            features = features.repeat_interleave(self.num_samples, dim=0)

        # base_scale = self.scale / (
        #     context_mask.sum() / torch.ones_like(context_mask).sum()
        # )
        base_scale = self.scale

        pred = self.guide(observation, observation_mask, features, base_scale)
        pred = pred[:, :, 0].reshape(batch_size, self.num_samples, -1)
        pred = pred * scale_params

        return pred[..., length - self.prediction_length :]

    def get_predictor(self, input_transform, batch_size=40, device=None):
        return PyTorchPredictorWGrads(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )


class DDPMGuidance(Guidance):
    def __init__(
        self,
        model: TSDiff,
        prediction_length: int,
        scale: float = 1,
        num_samples: int = 1,
        guidance: str = "quantile",
        missing_scenario: str = "none",
        missing_values: int = 0,
    ):
        super().__init__(
            model,
            prediction_length,
            scale,
            num_samples,
            guidance,
            missing_scenario,
            missing_values,
        )

    def scale_func(self, y, t, base_scale):
        return extract(self.model.posterior_variance, t, y.shape) * base_scale

    @torch.no_grad()
    def _reverse_diffusion(
        self, observation, observation_mask, features, base_scale
    ):
        device = observation.device
        batch_size = observation.shape[0]

        seq = torch.randn_like(observation)
        for i in reversed(range(0, self.model.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            seq = self.model.p_sample(seq, t, i, features)
            scale = self.scale_func(seq, t, base_scale=base_scale)
            seq = seq + scale * self.score_func(
                seq,
                t,
                observation=observation,
                observation_mask=observation_mask,
                features=features,
            )

        return seq

    def guide(self, observation, observation_mask, features, base_scale):
        return self._reverse_diffusion(
            observation, observation_mask, features, base_scale
        )


class DDIMGuidance(Guidance):
    _skip_types = ["uniform", "quadratic"]

    def __init__(
        self,
        model: TSDiff,
        prediction_length: int,
        eta: float = 0.0,
        skip_factor: int = 1,
        skip_type: str = "uniform",
        scale: float = 1,
        num_samples: int = 1,
        guidance: str = "quantile",
        missing_scenario: str = "none",
        missing_values: int = 0,
    ):
        super().__init__(
            model,
            prediction_length,
            scale,
            num_samples,
            guidance,
            missing_scenario,
            missing_values,
        )
        assert skip_type in self._skip_types
        self.eta = eta
        self.skip_factor = skip_factor
        self.skip_type = skip_type

    def scale_func(self, y, t, base_scale):
        return (
            extract(self.model.sqrt_one_minus_alphas_cumprod, t, y.shape)
            * base_scale
        )

    def _get_timesteps(self):
        if self.skip_type == "uniform":
            timesteps = range(0, self.model.timesteps, self.skip_factor)
        elif self.skip_type == "quadratic":
            n_test_timesteps = int(self.model.timesteps / self.skip_factor)
            c = 1 - self.skip_factor / self.model.timesteps
            timesteps = np.square(
                np.linspace(
                    0, np.sqrt(self.model.timesteps * c), n_test_timesteps
                )
            )
            timesteps = timesteps.astype(np.int64).tolist()
        timesteps = sorted(set(timesteps))
        return timesteps

    @torch.no_grad()
    def _reverse_ddim(
        self, observation, observation_mask, features, base_scale
    ):
        device = observation.device
        batch_size = observation.shape[0]
        timesteps = self._get_timesteps()
        timesteps_prev = [-1] + timesteps[:-1]

        seq = torch.randn_like(observation)

        for i, j in zip(reversed(timesteps), reversed(timesteps_prev)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_prev = torch.full(
                (batch_size,), j, device=device, dtype=torch.long
            )
            noise = self.model.backbone(seq, t, features)
            scale = self.scale_func(seq, t, base_scale=base_scale)
            noise = noise - scale * self.score_func(
                seq,
                t,
                observation=observation,
                observation_mask=observation_mask,
                features=features,
            )
            seq = self.model.p_sample_genddim(
                seq,
                t,
                t_index=i,
                t_prev=t_prev,
                eta=self.eta,
                features=features,
                noise=noise,
            )

        return seq

    def guide(self, observation, observation_mask, features, base_scale):
        return self._reverse_ddim(
            observation, observation_mask, features, base_scale
        )
