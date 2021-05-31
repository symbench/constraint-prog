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
        self.is_scalable = False

    def fit(self, x: torch.Tensor) -> None:
        """
        Calculate mean and std for data to standardize
        :param x: torch.Tensor data to standardize
        :return: None
        """
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        self.is_scalable = x.shape[0] > 2

    def transform(self, x: torch.Tensor) -> None:
        """
        Execute standardization
        :param x: torch.Tensor data to standardize
        :return: None
        """
        if self.is_scalable:
            x -= self.mean
            x /= (self.std + 1e-7)

    def rescale(self, x: torch.Tensor) -> None:
        """
        Execute standardization without in-place operations
        (since in-place operations cannot be applied for leaf Variables)
        :param x: torch.Tensor data to standardize
        :return: None
        """
        if self.is_scalable:
            x = x * (self.std + 1e-7)
            x = x + self.mean


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

    # Apply standard scaling for the input data
    scaler = TorchStandardScaler()
    scaler.fit(x=in_data)
    scaler.transform(x=in_data)

    # Proposed solution: create data
    inp_data = torch.clone(
        in_data.reshape(-1, in_data.shape[-1])
    )
    inp_data.requires_grad = True
    # Add rescaling to the computational graph here and apply rescale without grad later
    scaler.rescale(x=inp_data)
    with torch.no_grad():
        scaler.transform(x=inp_data)
    # Generate matrix of 1s with shape of inp_data
    ones = torch.ones(inp_data.shape[0], device=device)

    for t in range(1, it + 1):
        # Proposed solution
        # 1. Rescale input data for evaluating f
        with torch.no_grad():
            scaler.rescale(x=inp_data)
        # 2. Compute squared error from zero
        val = (f(inp_data).pow(2.0).sum(dim=-1)).pow(0.5)
        # 3. Compute gradients
        inp_data.grad = torch.autograd.grad(
            outputs=val,
            inputs=inp_data,
            grad_outputs=ones
        )[0]
        # 4. Apply one step of Adam method
        # a) Moment estimations:
        # - m_t: first moment
        # - v_t: second moment
        if m_t is None:
            m_t = torch.zeros_like(inp_data.grad)
        if v_t is None:
            v_t = torch.zeros_like(inp_data.grad)
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

    with torch.no_grad():
        scaler.rescale(x=inp_data)
    return inp_data
