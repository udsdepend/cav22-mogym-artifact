# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import json
import pathlib

import torch


def load_nn(path: pathlib.Path) -> torch.nn.Module:
    info = json.loads(path.read_text(encoding="utf-8"))

    layers = []

    for layer_info in info["layers"]:
        if layer_info["kind"] == "Linear":
            layer = torch.nn.Linear(layer_info["inputSize"], layer_info["outputSize"])
            weights = torch.tensor(layer_info["weights"])
            biases = torch.tensor(layer_info["biases"])
            with torch.no_grad():
                layer.weight.copy_(weights)
                layer.bias.copy_(biases)
        else:
            assert layer_info["kind"] == "ReLU"
            layer = torch.nn.ReLU()
        layers.append(layer)

    return torch.nn.Sequential(*layers)
