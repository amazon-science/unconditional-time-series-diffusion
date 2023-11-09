# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List
import math

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from gluonts.model import Estimator, Predictor
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSplitter,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
)
from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader
from gluonts.itertools import Cached
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    SampleForecastGenerator,
    predict_to_numpy,
)

from ._scaler import MeanScaler, NOPScaler

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


def stack(data):
    if isinstance(data[0], np.ndarray):
        data = np.array(data)
    elif isinstance(data[0], (list, tuple)):
        return list(stack(t) for t in zip(*data))
    return data


def batchify(data: List[dict]):
    return {
        key: stack(data=[item[key] for item in data]) for key in data[0].keys()
    }


class LinearModel:
    def __init__(self, weight, bias, scaler, num_parallel_samples=100) -> None:
        super().__init__()
        self.scaler = scaler
        self.weight = weight
        self.bias = bias
        self.num_parallel_samples = num_parallel_samples

    def _linear(self, x, A, b):
        return x @ A.T + b

    def __call__(self, x, mask):
        assert x.ndim == 2
        x, scale = self.scaler(x, np.ones_like(x))
        out = self._linear(x, self.weight, self.bias) * scale
        return np.tile(out[:, None], (1, self.num_parallel_samples, 1))


@predict_to_numpy.register(LinearModel)
def _(prediction_net, args) -> np.ndarray:
    return prediction_net(*args)


class LinearPredictor(Predictor):
    def __init__(
        self,
        input_names: List[str],
        prediction_net: LinearModel,
        batch_size: int,
        prediction_length: int,
        input_transform: Transformation,
        forecast_generator: ForecastGenerator = SampleForecastGenerator(),
        lead_time: int = 0,
    ) -> None:
        super().__init__(prediction_length, lead_time=lead_time)
        self.input_names = input_names
        self.prediction_net = prediction_net
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.forecast_generator = forecast_generator

    def predict(self, dataset: Dataset, num_samples: Optional[int] = None):
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=batchify,
        )

        yield from self.forecast_generator(
            inference_data_loader=inference_data_loader,
            prediction_net=self.prediction_net,
            input_names=self.input_names,
            output_transform=None,
            num_samples=num_samples,
        )


class LinearEstimator(Estimator):
    """A Linear regressor that takes inputs of size equal to `context_length`
    and outputs forecasts of size equal to `prediction_length`. This model uses
    LinearRegression from scikit-learn under the hood.

    Example usage:
    ```python
    estimator = LinearEstimator(
        dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        context_length=24 * 7 * 2,
    )

    predictor = estimator.train(dataset.train)
    ```

    Parameters
    ----------
    freq
        Frequency of the dataset (not actually used)
    prediction_length
        Prediction length
    context_length, optional
        Context length for the linear model,
        by default equal to 4 * prediction_length
    num_train_samples, optional
        Number of samples used to fit the LinearRegression model,
        by default 10000
    model, optional
        Which sklearn linear model to use, one of {"linear", "ridge"},
        by default "ridge".
    scaling, optional
        Whether to use scaling, by default True
    batch_size, optional
        Batch size (only relevant during prediction), by default 64
    """

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        num_train_samples: int = 10000,
        model: str = "ridge",
        scaling: bool = True,
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert model in {"linear", "ridge"}
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length or 4 * prediction_length
        self.num_train_samples = num_train_samples
        self.model = model

        if scaling:
            self.scaler = MeanScaler(axis=-1, keepdims=True)
        else:
            self.scaler = NOPScaler(axis=-1, keepdims=True)
        self.batch_size = batch_size

    def create_transformation(self) -> Transformation:
        return SelectFields(
            [
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
            ],
            allow_missing=True,
        ) + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1,
                min_past=self.context_length,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.OBSERVED_VALUES,
            ],
        )

    def _create_training_samples(self, training_data) -> np.ndarray:
        transformation = self._create_instance_splitter(
            "training"
        ) + SelectFields(TRAINING_INPUT_NAMES)
        num_batches_per_epoch = math.ceil(self.num_train_samples / 100)
        data_loader = TrainDataLoader(
            training_data,
            batch_size=100,
            stack_fn=batchify,
            transform=transformation,
            num_batches_per_epoch=num_batches_per_epoch,
        )

        train_X, train_y = [], []
        for batch in data_loader:
            train_X.append(batch["past_target"])
            train_y.append(batch["future_target"])
            assert np.all(batch["past_observed_values"] == 1.0) and np.all(
                batch["future_observed_values"] == 1.0
            ), "Missing values not supported!"
        train_X = np.concatenate(train_X, 0)
        train_y = np.concatenate(train_y, 0)
        train_X = train_X[: self.num_train_samples]
        train_y = train_y[: self.num_train_samples]

        assert len(train_X) == self.num_train_samples

        return train_X, train_y

    def create_predictor(self, transformation, model):
        prediction_splitter = self._create_instance_splitter("test")
        return LinearPredictor(
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=model,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            input_transform=transformation + prediction_splitter,
        )

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        cache_data: bool = False,
    ) -> Predictor:
        transformation = self.create_transformation()
        transformed_data = transformation.apply(training_data, is_train=True)

        if cache_data:
            transformed_data = Cached(transformed_data)

        train_X, train_y = self._create_training_samples(transformed_data)
        scaled_train_X, scale = self.scaler(train_X, np.ones_like(train_X))
        scaled_train_y = train_y / scale

        if self.model == "linear":
            SKLearnLinear = LinearRegression
        elif self.model == "ridge":
            SKLearnLinear = Ridge
        regressor = SKLearnLinear().fit(scaled_train_X, scaled_train_y)
        model = LinearModel(regressor.coef_, regressor.intercept_, self.scaler)
        return self.create_predictor(
            transformation=transformation, model=model
        )
