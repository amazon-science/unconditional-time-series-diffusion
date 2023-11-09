# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import copy
import logging
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from tqdm.auto import tqdm
from gluonts.mx import DeepAREstimator, TransformerEstimator
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    get_next_file_num,
    add_config_to_argparser,
    filter_metrics,
)
from uncond_ts_diff.model import TSDiff, LinearEstimator
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.sampler import (
    MostLikelyRefiner,
    MCMCRefiner,
    DDPMGuidance,
    DDIMGuidance,
)
import uncond_ts_diff.configs as diffusion_configs

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}
refiner_map = {"most_likely": MostLikelyRefiner, "mcmc": MCMCRefiner}


def load_model(config):
    model = TSDiff(
        **getattr(
            diffusion_configs,
            config.get("diffusion_config", "diffusion_small_config"),
        ),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization="mean",
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        init_skip=config["init_skip"],
    )
    model.load_state_dict(
        torch.load(config["ckpt"], map_location="cpu"),
        strict=True,
    )
    model = model.to(config["device"])
    return model


def get_best_diffusion_step(model: TSDiff, data_loader, device):
    losses = np.zeros(model.timesteps)
    batch = {
        k: v.to(device)
        for k, v in next(iter(data_loader)).items()
        if isinstance(v, torch.Tensor)
    }
    x, features, scale = model._extract_features(batch)
    for t in range(model.timesteps):
        loss, _, _ = model.p_losses(
            x.to(device), torch.tensor([t], device=device)
        )
        losses[t] = loss

    best_t = ((losses - losses.mean()) ** 2).argmin()
    return best_t


def train_and_forecast_base_model(dataset, base_model_name, config):
    base_model_kwargs = config.get("base_model_params", {})
    if base_model_name == "deepar":
        predictor = DeepAREstimator(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            **base_model_kwargs,
        ).train(list(dataset.train), cache_data=True)
    elif base_model_name == "transformer":
        predictor = TransformerEstimator(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
            **base_model_kwargs,
        ).train(list(dataset.train), cache_data=True)
    elif base_model_name == "seasonal_naive":
        predictor = SeasonalNaivePredictor(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            **base_model_kwargs,
        )
    elif base_model_name == "linear":
        num_train_samples = 10000
        predictor = LinearEstimator(
            freq=dataset.metadata.freq,
            prediction_length=dataset.metadata.prediction_length,
            context_length=config["context_length"],
            num_train_samples=num_train_samples,
            **base_model_kwargs,
        ).train(list(dataset.train), cache_data=True)
    else:
        raise ValueError(f"Unsupported base model {base_model_name}!")

    fcst_iter, ts_iter = make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
        num_samples=config["num_samples"],
    )
    fcsts = list(tqdm(fcst_iter, total=len(dataset.test)))
    tss = list(ts_iter)

    return fcsts, tss


def forecast_guidance(
    dataset,
    base_model_name,
    config,
    diffusion_model,
    transformed_testdata,
    test_splitter,
):
    assert len(dataset.test) == len(transformed_testdata)
    base_model_kwargs = config.get("base_model_params", {})

    Guidance = guidance_map[base_model_name]
    predictor = Guidance(
        model=diffusion_model,
        prediction_length=dataset.metadata.prediction_length,
        num_samples=config["num_samples"],
        **base_model_kwargs,
    ).get_predictor(
        input_transform=test_splitter,
        batch_size=1280 // config["num_samples"],
        device=config["device"],
    )

    fcst_iter, ts_iter = make_evaluation_predictions(
        dataset=transformed_testdata,
        predictor=predictor,
        num_samples=config["num_samples"],
    )
    fcsts = list(tqdm(fcst_iter, total=len(dataset.test)))
    tss = list(ts_iter)

    return fcsts, tss


def main(config: dict, log_dir: str):
    # Read global parameters
    dataset_name = config["dataset"]
    device = config["device"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    base_model_name = config["base_model"]
    num_samples = config["num_samples"]

    # Load dataset and model
    logger.info("Loading model")
    dataset = get_gts_dataset(dataset_name)
    config["freq"] = dataset.metadata.freq

    assert prediction_length == dataset.metadata.prediction_length

    model = load_model(config)

    # Setup data transformation and loading
    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=prediction_length,
    )
    transformed_data = transformation.apply(list(dataset.train), is_train=True)

    transformed_testdata = transformation.apply(
        list(dataset.test), is_train=False
    )

    training_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq),
        future_length=prediction_length,
        mode="train",
    )
    test_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq),
        future_length=prediction_length,
        mode="test",
    )

    train_dataloader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=1024,
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=2048,
    )

    best_t = get_best_diffusion_step(model, train_dataloader, device)

    # Train base model & get initial forecasts
    logger.info("Training base model")
    if base_model_name in {"ddpm", "ddim"}:
        base_fcsts, tss = forecast_guidance(
            dataset,
            base_model_name,
            config,
            diffusion_model=model,
            transformed_testdata=transformed_testdata,
            test_splitter=test_splitter,
        )
    else:
        base_fcsts, tss = train_and_forecast_base_model(
            dataset, base_model_name, config
        )

    # Evaluate base forecasts
    evaluator = Evaluator()
    baseline_metrics, _ = evaluator(tss, base_fcsts)
    baseline_metrics = filter_metrics(baseline_metrics)

    # Run refinement
    log_dir = Path(log_dir) / "refinement_logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    base_filename = "results"
    run_num = get_next_file_num(
        base_filename, log_dir, file_type="yaml", separator="-"
    )
    save_path = log_dir / f"{base_filename}-{run_num}.yaml"

    results = [
        {
            "model": "baseline",
            "model_params": {
                "name": base_model_name,
                **config.get("base_model_params", {}),
            },
            **baseline_metrics,
        }
    ]

    n_refiner_configs = len(config["refiner_configs"])
    for i, ref_config in enumerate(config["refiner_configs"]):
        logger.info(
            f"Running refiner ({i+1}/{n_refiner_configs}): {json.dumps(ref_config)}"
        )

        refiner_config = copy.deepcopy(ref_config)
        refiner_name = refiner_config.pop("refiner_name")
        Refiner = refiner_map[refiner_name]
        refiner = Refiner(
            model,
            prediction_length,
            init=iter(base_fcsts),
            num_samples=num_samples,
            fixed_t=best_t,
            iterations=config["iterations"],
            **refiner_config,
        )
        refiner_predictor = refiner.get_predictor(
            test_splitter, batch_size=1024 // num_samples, device=device
        )
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=transformed_testdata,
            predictor=refiner_predictor,
            num_samples=num_samples,
        )
        evaluator = Evaluator()
        refined_metrics, _ = evaluator(
            list(ts_it),
            list(tqdm(forecast_it, total=len(transformed_testdata))),
        )
        refined_metrics = filter_metrics(refined_metrics)

        results.append(
            {
                "model": refiner_name,
                "model_params": json.dumps(ref_config),
                **refined_metrics,
            }
        )

    with open(save_path, "w") as fp:
        yaml.safe_dump(
            {"config": config, "metrics": results},
            fp,
            default_flow_style=False,
            sort_keys=False,
        )


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results", help="Path to results dir"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    main(config=config, log_dir=args.out_dir)
