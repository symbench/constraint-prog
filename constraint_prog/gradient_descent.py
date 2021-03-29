#!/usr/bin/env python3
# Copyright (C) 2021, Zsolt Vizi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Callable

import torch


class TorchStandardScaler:
    """
    Standard scaler class using torch data methods
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor) -> None:
        """
        Calculate mean and std for data to standardize
        :param x: torch.Tensor data to standardize
        :return: None
        """
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute standardization
        :param x: torch.Tensor data to standardize
        :return: torch.Tensor standardized data
        """
        x = torch.clone(x)
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute standardization
        :param x: torch.Tensor data to standardize
        :return: torch.Tensor standardized data
        """
        x = torch.clone(x)
        x *= (self.std + 1e-7)
        x += self.mean
        return x


def gradient_descent(f: Callable, in_data: torch.Tensor, it: int,
                     lrate: float, device: torch.device) -> torch.Tensor:
    """
    Calculates 'it' many iterations of the vanilla gradient descent method.
    The input_data must of shape [*, input_size]. The func
    function must take a tensor of this shape and produce a tensor of shape
    [*, output_size].
    The returned tensor is of shape [*, input_size].
    """
    scaler = TorchStandardScaler()
    scaler.fit(x=in_data)
    scaled_in_data = scaler.transform(x=in_data)

    inp_data = torch.clone(
        scaled_in_data.reshape(-1, scaled_in_data.shape[-1]).to(device)
    )
    inp_data.requires_grad = True
    for _ in range(it):
        inp_data0 = scaler.rescale(x=inp_data)

        val = (f(inp_data0).pow(2.0).sum(dim=-1)).pow(0.5)
        inp_data.grad = torch.autograd.grad(
            outputs=val,
            inputs=inp_data0,
            grad_outputs=torch.ones(inp_data.shape[0]).to(device)
        )[0]

        with torch.no_grad():
            inp_data -= lrate * inp_data.grad
        inp_data.grad.zero_()

    return scaler.rescale(inp_data)
