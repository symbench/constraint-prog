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

from typing import List

import torch
import sympy
from sympy.logic.boolalg import BooleanTrue, BooleanFalse


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
            self.add_input_symbols(expr)
        assert self.input_names
        self.input_names = sorted(self.input_names)
        self._input_data = []

    def add_input_symbols(self, expr: sympy.Expr):
        """
        Adds all symbols occuring in the expression to the list of inputs
        of this function.
        """
        if expr.func == sympy.Symbol:
            if expr.name not in self.input_names:
                self.input_names.append(expr.name)
        for arg in expr.args:
            self.add_input_symbols(arg)

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
                or expr.func == sympy.core.numbers.NegativeOne
                or expr.func == sympy.core.numbers.Zero
                or expr.func == sympy.core.numbers.One):
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
            value0 = self._eval(expr.args[0])
            value1 = float(expr.args[1])
            return torch.pow(value0, value1)
        elif expr.func == sympy.Eq:
            assert len(expr.args) == 2
            value0 = self._eval(expr.args[0])
            value1 = self._eval(expr.args[1])
            return torch.sub(value0, value1)
        elif expr.func == sympy.StrictLessThan:
            assert len(expr.args) == 2
            value1 = self._eval(expr.args[0])
            value2 = self._eval(expr.args[1])
            return value1 < value2
        elif expr.func == sympy.StrictGreaterThan:
            assert len(expr.args) == 2
            value1 = self._eval(expr.args[0])
            value2 = self._eval(expr.args[1])
            return value1 > value2
        elif expr.func == sympy.Piecewise:
            # the fallback case must be fully defined
            assert expr.args[-1][1].func == BooleanTrue
            data = self._eval(expr.args[-1][0]).clone()
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
