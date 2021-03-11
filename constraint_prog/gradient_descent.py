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


def gradient_descent(f: Callable, in_data: torch.Tensor, it: int) -> torch.Tensor:
    """
    Calculates 'it' many iterations of the vanilla gradient descent method.
    The input_data must of shape [*, input_size]. The func
    function must take a tensor of this shape and produce a tensor of shape
    [*, output_size].
    The returned tensor is of shape [*, input_size].
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.01
    inp_data = torch.clone(
        in_data.reshape(-1, in_data.shape[-1]).to(device)
    )
    inp_data.requires_grad = True
    for _ in range(it):
        val = f(inp_data).pow(2).sum(dim=-1)
        inp_data.grad = torch.autograd.grad(
            outputs=val,
            inputs=inp_data,
            grad_outputs=torch.ones(inp_data.shape[0]).to(device)
        )[0]
        with torch.no_grad():
            inp_data -= learning_rate * inp_data.grad
        inp_data.grad.zero_()

    return inp_data.cpu().detach().reshape(in_data.shape)
