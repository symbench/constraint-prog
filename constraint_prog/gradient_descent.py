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
    Calculates 'it' many iterations of the Adam method.
    The input_data must of shape [*, input_size]. The func
    function must take a tensor of this shape and produce a tensor of shape
    [*, output_size].
    The returned tensor is of shape [*, input_size].
    """
    # Adam optimizer: parameters
    beta_1, beta_2 = (0.9, 0.999)
    eps = 1e-8
    # Adam optimizer: stored moments for each iterations
    m_t = None
    v_t = None

    # Flag for benchmarking
    is_benchmark_calculated = False
    # Apply standard scaling for the input data
    scaler = TorchStandardScaler()
    scaler.fit(x=in_data)
    scaled_in_data = scaler.transform(x=in_data)

    # Proposed solution: create data
    inp_data = torch.clone(
        scaled_in_data.reshape(-1, scaled_in_data.shape[-1]).to(device)
    )
    inp_data.requires_grad = True

    # Benchmark: create data and list of optimizers
    inp_data_bench = []
    optim_list = []
    if is_benchmark_calculated:
        for x in inp_data:
            y = x.clone().detach()
            y.requires_grad = True
            inp_data_bench.append(y)
            optim_list.append(
                torch.optim.Adam(params=[y], lr=lrate))

    for t in range(1, it + 1):
        # Proposed solution
        # 1. Rescale input data for evaluating f
        inp_data_rescaled = scaler.rescale(x=inp_data)
        # 2. Compute squared error from zero
        val = (f(inp_data_rescaled).pow(2.0).sum(dim=-1)).pow(0.5)
        # 3. Compute gradients
        inp_data.grad = torch.autograd.grad(
            outputs=val,
            inputs=inp_data,
            grad_outputs=torch.ones(inp_data.shape[0]).to(device)
        )[0]
        # 4. Apply one step of Adam method
        # a) Moment estimations:
        # - m_t: first moment
        # - v_t: second moment
        if m_t is None:
            m_t = inp_data.grad
        if v_t is None:
            v_t = inp_data.grad.pow(2.0)
        m_t = beta_1 * m_t + (1 - beta_1) * inp_data.grad
        v_t = beta_2 * v_t + (1 - beta_2) * inp_data.grad.pow(2.0)
        # Bias-corrected moment estimates
        m_t_hat = m_t / (1 - pow(beta_1, float(t)))
        v_t_hat = v_t / (1 - pow(beta_2, float(t)))
        # b) Scaling for gradient update step
        step_scale = lrate / (torch.sqrt(v_t_hat) + eps)
        # Tensor for update step
        step_tensor = step_scale * m_t_hat
        # c) Apply one step of Adam method
        with torch.no_grad():
            inp_data -= step_tensor
        # 5. Reset gradient
        inp_data.grad.zero_()

        # Benchmark
        if is_benchmark_calculated:
            for data, optim in zip(inp_data_bench, optim_list):
                # 1. Rescale data (vector) for evaluating f
                data_rescaled = scaler.rescale(x=data.view((1, -1)))
                # 2. Compute squared error from zero
                val_bench = (f(data_rescaled).pow(2.0).sum(dim=-1)).pow(0.5)
                # 3. Reset gradient
                optim.zero_grad()
                # 4. Compute gradients
                val_bench.backward()
                # 5. Apply one step of Adam method
                optim.step()

        # print("OK")

    return scaler.rescale(x=inp_data)
