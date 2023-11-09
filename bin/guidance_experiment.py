# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
from pathlib import Path

import yaml
import torch
from tqdm.auto import tqdm
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    get_next_file_num,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.sampler import (
    DDPMGuidance,
    DDIMGuidance,
)
import uncond_ts_diff.configs as diffusion_configs

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}


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


def evaluate_guidance(
    config, model, test_dataset, transformation, num_samples=100
):
    logger.info(f"Evaluating with {num_samples} samples.")
    results = []
    if config["setup"] == "forecasting":
        missing_data_kwargs_list = [
            {
                "missing_scenario": "none",
                "missing_values": 0,
            }
        ]
        config["missing_data_configs"] = missing_data_kwargs_list
    elif config["setup"] == "missing_values":
        missing_data_kwargs_list = config["missing_data_configs"]
    else:
        raise ValueError(f"Unknown setup {config['setup']}")

    Guidance = guidance_map[config["sampler"]]
    sampler_params = config["sampler_params"]
    for missing_data_kwargs in missing_data_kwargs_list:
        logger.info(
            f"Evaluating scenario '{missing_data_kwargs['missing_scenario']}' "
            f"with {missing_data_kwargs['missing_values']:.1f} missing_values."
        )

        sampler = Guidance(
            model=model,
            prediction_length=config["prediction_length"],
            num_samples=num_samples,
            **missing_data_kwargs,
            **sampler_params,
        )

        transformed_testdata = transformation.apply(
            test_dataset, is_train=False
        )
        test_splitter = create_splitter(
            past_length=config["context_length"] + max(model.lags_seq),
            future_length=config["prediction_length"],
            mode="test",
        )

        masking_transform = MaskInput(
            FieldName.TARGET,
            FieldName.OBSERVED_VALUES,
            config["context_length"],
            missing_data_kwargs["missing_scenario"],
            missing_data_kwargs["missing_values"],
        )
        test_transform = test_splitter + masking_transform

        predictor = sampler.get_predictor(
            test_transform,
            batch_size=1280 // num_samples,
            device=config["device"],
        )
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=transformed_testdata,
            predictor=predictor,
            num_samples=num_samples,
        )
        forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
        tss = list(ts_it)
        evaluator = Evaluator()
        metrics, _ = evaluator(tss, forecasts)
        metrics = filter_metrics(metrics)
        results.append(dict(**missing_data_kwargs, **metrics))

    return results


def main(config: dict, log_dir: str):
    # Read global parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    prediction_length = config["prediction_length"]
    num_samples = config["num_samples"]

    # Load dataset and model
    logger.info("Loading model")
    model = load_model(config)
    dataset = get_gts_dataset(dataset_name)
    assert dataset.metadata.freq == freq
    assert dataset.metadata.prediction_length == prediction_length

    # Setup data transformation and loading
    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=prediction_length,
    )

    # Run guidance
    results = evaluate_guidance(
        config, model, dataset.test, transformation, num_samples=num_samples
    )

    # Save results
    log_dir = Path(log_dir) / "guidance_logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    base_filename = "results"
    run_num = get_next_file_num(
        base_filename, log_dir, file_type="yaml", separator="-"
    )
    save_path = log_dir / f"{base_filename}-{run_num}.yaml"

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
