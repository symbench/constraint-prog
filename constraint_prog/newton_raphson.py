#!/usr/bin/env python3
# Copyright (C) 2021, Miklos Maroti
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

from typing import Callable, Optional

import torch


def jacobian(func: Callable, input_data: torch.tensor) -> torch.tensor:
    """
    Calculates the output and the Jacobian of the function at the given
    input data. The input data is of shape [*, input_size], while the
    output data is of shape [*, output_size] and the Jacobian is of size
    [*, output_size, input_size]. The function must take a tensor of
    shape [*, input_size] and produce a tensor of shape [*, output_size].
    """
    assert input_data.ndim >= 1
    shape = input_data.shape
    input_data = input_data.reshape((-1, shape[-1]))
    input_data.requires_grad = True
    output_data = func(input_data)
    jacobian_data = torch.empty(
        (input_data.shape[0], output_data.shape[1], input_data.shape[1]),
        device=input_data.device)
    for i in range(output_data.shape[1]):
        jacobian_data[:, i, :] = torch.autograd.grad(
            output_data[:, i],
            input_data,
            torch.ones(input_data.shape[0], device=input_data.device),
            retain_graph=i + 1 < output_data.shape[1])[0]
    output_data = output_data.reshape(shape[:-1] + output_data.shape[-1:])
    jacobian_data = jacobian_data.reshape(
        shape[:-1] + jacobian_data.shape[-2:])
    return output_data.detach(), jacobian_data.detach()


def pseudo_inverse1(matrix: torch.tensor, epsilon: float = 1e-3) -> torch.tensor:
    """
    Takes a tensor of shape [*, rows, cols] and returns a tensor of shape
    [*, cols, rows]. Only the singular values above epsilon are inverted,
    the rest are zeroed out.
    """
    assert epsilon >= 0.0
    u, s, v = matrix.svd()
    if False:
        pos = s <= epsilon
        s[pos] = epsilon
        s = 1.0 / s
        s[pos] = 0.0
    else:
        s = 1.0 / torch.clamp(s, min=epsilon)
    a = torch.matmul(v, torch.diag_embed(s))
    return torch.matmul(a, u.transpose(-2, -1))


def pseudo_inverse2(matrix: torch.tensor, epsilon: float = 1e-3) -> torch.tensor:
    """
    Takes a tensor of shape [*, rows, cols] and returns a tensor of shape
    [*, cols, rows].
    """
    assert epsilon >= 0.0
    device = matrix.device

    # SVD is really slow on CUDA, do it on the CPU
    if device == "cuda":
        matrix = matrix.cpu()

    u, s, v = torch.linalg.svd(matrix, full_matrices=False, compute_uv=True)

    if device == "cuda":
        u = u.to(device)
        s = s.to(device)
        v = v.to(device)

    s = 1.0 / torch.clamp(s, min=epsilon)
    s = torch.diag_embed(s)
    a = torch.matmul(v.transpose(-2, -1), s)
    b = torch.matmul(a, u.transpose(-2, -1))

    return b


class MethodMinMax(object):
    """
    Enforces the bounding box by calculating the amount we are
    outside of the box and setting them to zero.
    """

    def __init__(self, func: Callable, bounding_box: torch.tensor):
        self.func = func
        self.bounding_box = bounding_box
        self.zero = torch.zeros((), device=bounding_box.device)

    def __call__(self, input_data: torch.tensor) -> torch.tensor:
        output_data = self.func(input_data)
        min_err = (self.bounding_box[0] - input_data).maximum(self.zero)
        max_err = (input_data - self.bounding_box[1]).maximum(self.zero)
        return torch.cat((output_data, min_err + max_err), dim=-1)


def newton_raphson(func: Callable, input_data: torch.tensor,
                   num_iter: int = 10, epsilon: float = 1e-3,
                   bounding_box: Optional[torch.tensor] = None,
                   method: str = "clip") -> torch.tensor:
    """
    Calculates num_iter many iterations of the multidimensional Newton-
    Raphson method. The input_data must of shape [*, input_size]. The func
    function must take a tensor of this shape and produce a tensor of shape
    [*, output_size]. The epsilon is controlling the pseudo inverse operation.
    The bounding box is of shape [2, input_size] and specifies the minimum
    maximum values. The returned tensor is of shape [*, input_size].
    """
    if bounding_box is None:
        method = "none"
    if method == "minmax":
        func = MethodMinMax(func, bounding_box)
        method = "none"

    for _ in range(num_iter):
        output_data, jacobian_data = jacobian(func, input_data)
        jacobian_inv = pseudo_inverse2(jacobian_data, epsilon=epsilon)
        update = torch.matmul(
            jacobian_inv, output_data.unsqueeze(dim=-1)).squeeze(-1)
        input_data -= update
        if method == "clip":
            input_data = torch.minimum(input_data, bounding_box[1])
            input_data = torch.maximum(input_data, bounding_box[0])
    return input_data
