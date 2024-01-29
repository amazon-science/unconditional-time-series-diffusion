# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
from pathlib import Path

import yaml
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.model import TSDiffCond
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
    ConcatDataset,
)


def create_model(config):
    model = TSDiffCond(
        **getattr(diffusion_configs, config["diffusion_config"]),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization=config["normalization"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        lr=config["lr"],
        init_skip=config["init_skip"],
        noise_observed=config["noise_observed"],
    )
    model.to(config["device"])
    return model


def evaluate_conditional(
    config,
    model: TSDiffCond,
    test_dataset,
    transformation,
    num_samples=100,
):
    logger.info(f"Evaluating with {num_samples} samples.")
    logger.info(
        f"Evaluating scenario '{config['missing_scenario']}' "
        f"with {config['missing_values']:.1f} missing_values."
    )

    results = []

    transformed_testdata = transformation.apply(test_dataset, is_train=False)
    test_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="test",
    )

    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        config["context_length"],
        config["missing_scenario"],
        config["missing_values"],
    )
    test_transform = test_splitter + masking_transform

    predictor = model.get_predictor(
        test_transform,
        batch_size=1280,
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
    results.append(dict(**metrics))

    return results


def main(config, log_dir):
    # Load parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    total_length = context_length + prediction_length

    # Create model
    model = create_model(config)

    # Setup dataset and data loading
    dataset = get_gts_dataset(dataset_name)
    assert dataset.metadata.freq == freq
    assert dataset.metadata.prediction_length == prediction_length

    if config["setup"] == "forecasting":
        training_data = dataset.train
    elif config["setup"] == "missing_values":
        missing_values_splitter = OffsetSplitter(offset=-total_length)
        training_data, _ = missing_values_splitter.split(dataset.train)

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))

    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=config["prediction_length"],
    )

    training_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="train",
    )

    if config["setup"] == "forecasting":
        config["missing_scenario"] = "none"
        config["missing_values"] = 0

    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        config["context_length"],
        config.get("train_missing_scenario", config["missing_scenario"]),
        config["missing_values"],
    )
    train_transform = training_splitter + masking_transform

    callbacks = []
    val_loader = None
    if config["use_validation_set"]:
        transformed_data = transformation.apply(training_data, is_train=True)
        train_val_splitter = OffsetSplitter(
            offset=-config["prediction_length"] * num_rolling_evals
        )
        _, val_gen = train_val_splitter.split(training_data)

        val_dataset = ConcatDataset(
            val_gen.generate_instances(
                config["prediction_length"], num_rolling_evals
            )
        )
        val_splitter = create_splitter(
            past_length=config["context_length"] + max(model.lags_seq),
            future_length=config["prediction_length"],
            mode="val",
        )
        transformed_valdata = transformation.apply(val_dataset, is_train=True)
        val_loader = ValidationDataLoader(
            transformed_valdata,
            batch_size=1280,
            stack_fn=batchify,
            transform=val_splitter + masking_transform,
        )

        callbacks = []
        log_monitor = "valid_loss"
    else:
        transformed_data = transformation.apply(training_data, is_train=True)
        log_monitor = "train_loss"

    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"

    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=train_transform,
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)
    callbacks.append(RichProgressBar())

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=[int(config["device"].split(":")[-1])],
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
        gradient_clip_val=config.get("gradient_clip_val", None),
        check_val_every_n_epoch=config["eval_every"],
    )
    logger.info(f"Logging to {trainer.logger.log_dir}")
    trainer.fit(
        model, train_dataloaders=data_loader, val_dataloaders=val_loader
    )
    logger.info("Training completed.")

    best_ckpt_path = Path(trainer.logger.log_dir) / "best_checkpoint.ckpt"

    if not best_ckpt_path.exists():
        torch.save(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
            best_ckpt_path,
        )
    logger.info(f"Loading {best_ckpt_path}.")
    best_state_dict = torch.load(best_ckpt_path)
    model.load_state_dict(best_state_dict, strict=True)

    metrics = (
        evaluate_conditional(config, model, dataset.test, transformation)
        if config.get("do_final_eval", True)
        else "Final eval not performed"
    )
    with open(Path(trainer.logger.log_dir) / "results.yaml", "w") as fp:
        yaml.dump(
            {
                "config": config,
                "version": trainer.logger.version,
                "metrics": metrics,
            },
            fp,
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
        "--out_dir", type=str, default="./", help="Path to results dir"
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
