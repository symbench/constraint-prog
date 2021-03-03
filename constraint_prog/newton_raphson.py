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

from typing import Callable, List

import torch
import sympy
from sympy.logic.boolalg import BooleanTrue, BooleanFalse


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
        (input_data.shape[0], output_data.shape[1], input_data.shape[1]))
    for i in range(output_data.shape[1]):
        jacobian_data[:, i, :] = torch.autograd.grad(
            output_data[:, i],
            input_data,
            torch.ones(input_data.shape[0]),
            retain_graph=i + 1 < output_data.shape[1])[0]
    output_data = output_data.reshape(shape[:-1] + output_data.shape[-1:])
    jacobian_data = jacobian_data.reshape(
        shape[:-1] + jacobian_data.shape[-2:])
    return output_data.detach(), jacobian_data.detach()


def pseudo_inverse(matrix: torch.tensor, epsilon: float = 1e-3) -> torch.tensor:
    """
    Takes a tensor of shape [*, rows, cols] and returns a tensor of shape
    [*, cols, rows]. Only the singular values above epsilon are inverted,
    the rest are zeroed out.
    """
    assert epsilon >= 0.0
    u, s, v = matrix.svd()
    pos = s <= epsilon
    s[pos] = 1.0
    s = 1.0 / s
    s[pos] = 0.0
    a = torch.matmul(v, torch.diag_embed(s))
    return torch.matmul(a, u.transpose(-2, -1))


def newton_raphson(func: Callable, input_data: torch.tensor,
                   num_iter: int = 10, epsilon: float = 1e-3) -> torch.tensor:
    """
    Calculates num_iter many iterations of the multidimensional Newton-
    Raphson method. The input_data must of shape [*, input_size]. The func
    function must take a tensor of this shape and produce a tensor of shape
    [*, output_size]. The epsilon is controlling the pseudo inverse operation.
    The returned tensor is of shape [*, input_size].
    """
    for _ in range(num_iter):
        output_data, jacobian_data = jacobian(func, input_data)
        jacobian_inv = pseudo_inverse(jacobian_data, epsilon=epsilon)
        update = torch.matmul(
            jacobian_inv, output_data.unsqueeze(dim=-1)).squeeze(-1)
        input_data = input_data.detach() - update
    return input_data


class SympyFunc(object):
    """
    A callable object created from a list of sympy symbolic expressions. 
    The object can be called with a tensor of shape [*, input_size] and
    will produce a tensor of shape[*, output_size] where input_size is
    the number of symbols and output_size is the number of expressions.
    """

    def __init__(self, expressions: List[sympy.Expr]):
        self.expressions = expressions
        self.input_names = []
        for expr in expressions:
            self._add_input_names(expr)
        assert self.input_names
        self._input_data = []

    def _add_input_names(self, expr: sympy.Expr):
        if expr.func == sympy.Symbol:
            if expr.name not in self.input_names:
                self.input_names.append(expr.name)
        for arg in expr.args:
            self._add_input_names(arg)

    def __call__(self, input_data: torch.tensor) -> torch.tensor:
        return self.evaluate(self.expressions, input_data)

    def evaluate(self, expressions: List[sympy.Expr],
                 input_data: torch.tensor) -> torch.tensor:
        assert input_data.shape[-1] == len(self.input_names)
        self._input_data = input_data.unbind(dim=-1)
        output_data = []
        for expr in expressions:
            output_data.append(self._eval(expr))
        self._input_data = []
        return torch.stack(output_data, dim=-1)

    def _eval(self, expr: sympy.Expr) -> torch.tensor:
        if (expr.func == sympy.Integer or expr.func == sympy.Float
                or expr.func == sympy.numbers.NegativeOne
                or expr.func == sympy.numbers.Zero
                or expr.func == sympy.numbers.One):
            return torch.full(self._input_data[0].shape, float(expr))
        elif expr.func == sympy.Symbol:
            return self._input_data[self.input_names.index(expr.name)]
        elif expr.func == BooleanTrue or expr.func == BooleanFalse:
            return torch.full(self._input_data[0].shape, bool(expr))
        elif expr.func == sympy.Add:
            value = self._eval(expr.args[0])
            for arg in expr.args[1:]:
                other = self._eval(arg)
                value = torch.add(value, other)
            return value
        elif expr.func == sympy.Mul:
            value = self._eval(expr.args[0])
            for arg in expr.args[1:]:
                other = self._eval(arg)
                value = torch.mul(value, other)
            return value
        elif expr.func == sympy.Max:
            value = self._eval(expr.args[0])
            for arg in expr.args[1:]:
                other = self._eval(arg)
                value = torch.max(value, other)
            return value
        elif expr.func == sympy.Min:
            value = self._eval(expr.args[0])
            for arg in expr.args[1:]:
                other = self._eval(arg)
                value = torch.min(value, other)
            return value
        elif expr.func == sympy.Pow:
            assert len(expr.args) == 2
            value1 = self._eval(expr.args[0])
            value2 = float(expr.args[1])
            return torch.pow(value1, value2)
        elif expr.func == sympy.StrictLessThan:
            assert len(expr.args) == 2
            value1 = self._eval(expr.args[0])
            value2 = self._eval(expr.args[1])
            return value1 < value2
        elif expr.func == sympy.Piecewise:
            # the fallback case must be fully defined
            assert expr.args[-1][1].func == BooleanTrue
            data = self._eval(expr.args[-1][0])
            for (a, b) in expr.args[-2::-1]:
                a = self._eval(a)
                b = self._eval(b)
                data[b] = a[b]
            return data
        else:
            raise ValueError(
                "Unknown symbolic expression " + str(type(expr)))

    def __repr__(self) -> str:
        return self.expressions.__repr__()

    @property
    def input_size(self):
        return len(self.input_names)

    @property
    def output_size(self):
        return len(self.expressions)
