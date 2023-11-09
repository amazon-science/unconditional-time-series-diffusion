# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Iterator, Optional

from gluonts.dataset import Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model import Forecast
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor


class PyTorchPredictorWGrads(PyTorchPredictor):
    def predict(
        self, dataset: Dataset, num_samples: Optional[int] = None
    ) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=lambda data: batchify(data, self.device),
        )

        self.prediction_net.eval()

        yield from self.forecast_generator(
            inference_data_loader=inference_data_loader,
            prediction_net=self.prediction_net,
            input_names=self.input_names,
            output_transform=self.output_transform,
            num_samples=num_samples,
        )
