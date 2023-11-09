# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch

from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from gluonts.transform import TestSplitSampler, InstanceSplitter
from pytorch_lightning import Callback

from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.metrics import linear_pred_score
from uncond_ts_diff.utils import ConcatDataset


class GradNormCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_before_optimizer_step(
        self,
        trainer,
        pl_module,
        optimizer,
        opt_idx: int,
    ) -> None:
        return pl_module.log(
            "grad_norm", self.grad_norm(pl_module.parameters()), prog_bar=True
        )

    def grad_norm(self, parameters):
        parameters = [p for p in parameters if p.grad is not None]
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), 2).to(device) for p in parameters]
            ),
            2,
        )
        return total_norm


class PredictiveScoreCallback(Callback):
    def __init__(
        self,
        context_length,
        prediction_length,
        model,
        transformation,
        train_dataloader,
        train_batch_size,
        test_dataset,
        eval_every=10,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model = model
        self.transformation = transformation
        self.train_dataloader = train_dataloader
        self.train_batch_size = train_batch_size
        self.test_dataset = test_dataset
        self.eval_every = eval_every
        # Number of samples used to train the downstream predictor
        self.n_pred_samples = 10000

    def _generate_real_samples(
        self,
        data_loader,
        num_samples: int,
        n_timesteps: int,
        batch_size: int,
        cache_path: Path,
    ):
        if cache_path.exists():
            real_samples = np.load(cache_path)
            if len(real_samples) == num_samples:
                return real_samples

        real_samples = []
        data_iter = iter(data_loader)
        n_iters = math.ceil(num_samples / batch_size)
        for i in range(n_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)
            ts = np.concatenate(
                [batch["past_target"], batch["future_target"]], axis=-1
            )[:, -n_timesteps:]
            real_samples.append(ts)

        real_samples = np.concatenate(real_samples, axis=0)[:num_samples]
        np.save(cache_path, real_samples)

        return real_samples

    def _generate_synth_samples(
        self, model, num_samples: int, batch_size: int = 1000
    ):
        synth_samples = []

        n_iters = math.ceil(num_samples / batch_size)
        for _ in range(n_iters):
            samples = model.sample_n(num_samples=batch_size)
            synth_samples.append(samples)

        synth_samples = np.concatenate(synth_samples, axis=0)[:num_samples]
        return synth_samples

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.eval_every == 0:
            device = next(pl_module.backbone.parameters()).device
            pl_module.eval()
            assert pl_module.training is False

            real_samples = self._generate_real_samples(
                self.train_dataloader,
                self.n_pred_samples,
                self.context_length + self.prediction_length,
                self.train_batch_size,
                cache_path=Path(trainer.logger.log_dir) / "real_samples.npy",
            )
            synth_samples = self._generate_synth_samples(
                self.model,
                self.n_pred_samples,
            )

            # Train using synthetic samples, test on test set
            synth_metrics, _, _ = linear_pred_score(
                synth_samples,
                self.context_length,
                self.prediction_length,
                self.test_dataset,
                scaling_type="mean",
            )

            # Train using real samples, test on test set
            scaled_real_samples, _ = self.model.scaler(
                torch.from_numpy(real_samples).to(device),
                torch.from_numpy(np.ones_like(real_samples)).to(device),
            )
            real_metrics, _, _ = linear_pred_score(
                scaled_real_samples.cpu().numpy(),
                self.context_length,
                self.prediction_length,
                self.test_dataset,
                scaling_type="mean",
            )

            pl_module.log_dict(
                {
                    "synth_linear_ND": synth_metrics["ND"],
                    "synth_linear_NRMSE": synth_metrics["NRMSE"],
                    "real_linear_ND": real_metrics["ND"],
                    "real_linear_NRMSE": real_metrics["NRMSE"],
                }
            )

            pl_module.train()


class EvaluateCallback(Callback):
    def __init__(
        self,
        context_length,
        prediction_length,
        sampler,
        sampler_kwargs,
        num_samples,
        model,
        transformation,
        test_dataset,
        val_dataset,
        eval_every=50,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.sampler = sampler
        self.num_samples = num_samples
        self.sampler_kwargs = sampler_kwargs
        self.model = model
        self.transformation = transformation
        self.test_dataset = test_dataset
        self.val_data = val_dataset
        self.original_state_dict = {}
        self.eval_every = eval_every
        self.log_metrics = {
            "CRPS",
            "ND",
            "NRMSE",
        }

        if sampler == "ddpm":
            self.Guidance = DDPMGuidance
        elif sampler == "ddim":
            self.Guidance = DDIMGuidance
        else:
            raise ValueError(f"Unknown sampler type: {sampler}")

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.eval_every == 0:
            device = next(pl_module.backbone.parameters()).device
            self.original_state_dict = deepcopy(
                pl_module.backbone.state_dict()
            )
            pl_module.eval()
            assert pl_module.training is False
            for label, state_dict in zip(
                [""] + [str(rate) for rate in pl_module.ema_rate],
                [pl_module.backbone.state_dict()] + pl_module.ema_state_dicts,
            ):
                pl_module.backbone.load_state_dict(state_dict, strict=True)
                pl_module.to(device)
                prediction_splitter = InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    instance_sampler=TestSplitSampler(),
                    past_length=self.context_length + max(self.model.lags_seq),
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                )
                og = self.Guidance(
                    self.model,
                    self.prediction_length,
                    num_samples=self.num_samples,
                    **self.sampler_kwargs,
                )
                predictor_pytorch = og.get_predictor(
                    prediction_splitter,
                    batch_size=1024 // self.num_samples,
                    device=device,
                )
                evaluator = Evaluator()

                transformed_valdata = self.transformation.apply(
                    ConcatDataset(self.val_data), is_train=False
                )

                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=transformed_valdata,
                    predictor=predictor_pytorch,
                    num_samples=self.num_samples,
                )

                forecasts_pytorch = list(forecast_it)
                tss_pytorch = list(ts_it)

                metrics_pytorch, per_ts = evaluator(
                    tss_pytorch, forecasts_pytorch
                )
                metrics_pytorch["CRPS"] = metrics_pytorch["mean_wQuantileLoss"]
                if metrics_pytorch["CRPS"] < pl_module.best_crps:
                    pl_module.best_crps = metrics_pytorch["CRPS"]
                    ckpt_path = (
                        Path(trainer.logger.log_dir) / "best_checkpoint.ckpt"
                    )
                    torch.save(
                        pl_module.state_dict(),
                        ckpt_path,
                    )
                pl_module.log_dict(
                    {
                        f"val_{metric}{label}": metrics_pytorch[metric]
                        for metric in self.log_metrics
                    }
                )
            pl_module.backbone.load_state_dict(
                self.original_state_dict, strict=True
            )
            pl_module.train()
