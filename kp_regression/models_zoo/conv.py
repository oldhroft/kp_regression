import typing as T

import torch
import torch.nn as nn

from kp_regression.models_zoo.rnn import LSTM3Inputs
from kp_regression.models_zoo.torch_common import TorchModelParams


def conv1d_block(n_inputs: int, n_outputs: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            padding="same",
        ),
        nn.ReLU(),
        nn.AvgPool1d(2, 2),
    )


def fc_layer(n_inputs: int, n_outputs: int, use_relu: bool = False) -> nn.Module:
    if use_relu:
        return nn.Sequential(nn.Linear(n_inputs, n_outputs), nn.ReLU())
    else:
        return nn.Linear(n_inputs, n_outputs)


def get_fc_net(
    input_shape: T.Tuple[int, ...], layers: T.List[int], last: bool = False
) -> nn.Sequential:
    assert len(input_shape) == 1, "FCNet only accepts 1D data"
    n_inputs = input_shape[0]

    layers_list = []

    for i, n_outputs in enumerate(layers):
        use_relu = i < len(layers) - 1 or not last
        layers_list.append(
            fc_layer(n_inputs=n_inputs, n_outputs=n_outputs, use_relu=use_relu)
        )
        n_inputs = n_outputs

    return nn.Sequential(*layers_list)


def get_conv1d_backbone(
    input_shape: T.Tuple[int, ...], layers: T.List[int], kernel_size: int = 3
) -> T.Tuple[nn.Sequential, int]:
    assert len(input_shape) == 2, "Conv1Net only accepts 2D data"
    n_inputs, n_features = input_shape

    layers_list: T.List[nn.Module] = []

    for n_outputs in layers:

        layers_list.append(
            conv1d_block(
                n_inputs=n_inputs, n_outputs=n_outputs, kernel_size=kernel_size
            )
        )
        n_inputs = n_outputs
        n_features = n_features // 2

    layers_list.append(nn.Flatten())

    return nn.Sequential(*layers_list), n_features * n_outputs


class Conv1DNet3inputs(nn.Module):
    def __init__(
        self,
        input_shape1: T.Tuple[int, ...],
        input_shape2: T.Tuple[int, ...],
        input_shape3: T.Tuple[int, ...],
        conv_layers1: T.List[int],
        conv_layers2: T.List[int],
        layers: T.List[int],
        layers_head: T.List[int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.cv_net1, n1 = get_conv1d_backbone(
            input_shape=input_shape1, layers=conv_layers1, kernel_size=kernel_size
        )
        self.cv_net2, n2 = get_conv1d_backbone(
            input_shape=input_shape2, layers=conv_layers2, kernel_size=kernel_size
        )

        self.net3 = get_fc_net(input_shape=input_shape3, layers=layers)

        input_shape_fc = n1 + n2 + layers[-1]

        self.head = get_fc_net(
            input_shape=(input_shape_fc,), layers=layers_head, last=True
        )

    def forward(self, x1, x2, x3):
        y1 = self.cv_net1(x1)
        y2 = self.cv_net2(x2)
        y3 = self.net3(x3)

        flattaned = torch.concat([y1, y2, y3], dim=1)

        return self.head(flattaned)


class Conv1DNet3InputsMulti(LSTM3Inputs):

    def build(self) -> None:
        self.torch_model_params = TorchModelParams(**self.model_params)

        assert len(self.shape) == 3, "Shape should contain 3 inputs"

        assert isinstance(self.shape[0], tuple), "First item in shape must be a tuple"
        assert isinstance(self.shape[1], tuple), "Second item in shape must be a tuple"
        assert isinstance(self.shape[2], tuple), "Third item in shape must be a tuple"
        
        self.models = [
            Conv1DNet3inputs(
                input_shape1=self.shape[0],
                input_shape2=self.shape[1],
                input_shape3=self.shape[2],
                **self.torch_model_params.model_params,
            )
            for i in range(self.output_shape[0])
        ]

        self.n_inputs = 3
        # summary(self.models[0], list(self.shape), device=self.model_params.accelerator)