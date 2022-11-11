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

from typing import Callable, Dict, List

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

    def __init__(self, expressions: List[sympy.Expr], device=None):
        self.expressions = expressions
        self.device = device

        self.input_names = []
        for expr in expressions:
            self.add_input_symbols(expr)
        self.input_names = sorted(self.input_names)
        self._input_data = []
        self._input_shape = None

    def add_input_symbols(self, expr: sympy.Expr):
        """
        Adds all symbols occuring in the expression to the list of inputs
        of this function.
        """
        if isinstance(expr, float) or isinstance(expr, int) \
                or isinstance(expr, bool):
            return
        if expr.func == sympy.Symbol:
            if expr.name not in self.input_names:
                self.input_names.append(expr.name)
        for arg in expr.args:
            self.add_input_symbols(arg)

    def __call__(self, input_data: torch.Tensor,
                 equs_as_float: bool = True) -> torch.Tensor:
        return self.evaluate(self.expressions, input_data, equs_as_float)

    def evaluate(self, expressions: List[sympy.Expr],
                 input_data: torch.Tensor,
                 equs_as_float: bool) -> torch.Tensor:
        """
        Evaluates the set of expressions using the given input data. If
        equs_as_float is true, then sympy equations and inequalities are
        returned as float values as the difference between the two sides,
        as opposed to boolean tensors. The shape of the input must be
        [*, len(input_names)] and the output is of shape [*, len(expressions)].
        """
        assert input_data.shape[-1] == len(self.input_names)
        self._input_shape = input_data.shape[:-1]
        self._input_data = input_data.unbind(dim=-1)

        output_data = []
        if equs_as_float:
            for expr in expressions:
                output_data.append(self._eval_equ_as_sub(expr))
        else:
            for expr in expressions:
                output_data.append(self._eval(expr))
        self._input_data = []

        output_data = torch.stack(output_data, dim=-1)
        # zero out bad output data
        if output_data.dtype == torch.float32 or output_data.dtype == torch.float64:
            output_data = output_data.nan_to_num(
                nan=0.0, posinf=1e40, neginf=-1e40)

        return output_data

    def evaluate2(self,
                  expressions: Dict[str, sympy.Expr],
                  input_data: Dict[str, torch.Tensor],
                  equs_as_float: bool) -> Dict[str, torch.Tensor]:
        """
        New version of the evaluate function that uses a dictionary.
        """
        input_data = [input_data[name] for name in self.input_names]
        input_data = torch.stack(input_data, dim=-1)

        output_data = self.evaluate(
            expressions.values(), input_data, equs_as_float)
        output_data = output_data.unbind(dim=-1)

        return {var: output_data[idx] for idx, var in enumerate(expressions.keys())}

    def _eval_equ_as_sub(self, expr: sympy.Expr) -> torch.Tensor:
        if isinstance(expr, sympy.core.relational.Relational):
            if expr.func == sympy.Eq:
                assert len(expr.args) == 2
                value0 = self._eval(expr.args[0])
                value1 = self._eval(expr.args[1])
                return torch.sub(value0, value1)
            elif expr.func == sympy.StrictLessThan or expr.func == sympy.LessThan:
                assert len(expr.args) == 2
                value0 = self._eval(expr.args[0])
                value1 = self._eval(expr.args[1])
                return torch.sub(value0, value1).clamp_min(0.0)
            elif expr.func == sympy.StrictGreaterThan or expr.func == sympy.GreaterThan:
                assert len(expr.args) == 2
                value0 = self._eval(expr.args[0])
                value1 = self._eval(expr.args[1])
                return torch.sub(value0, value1).clamp_max(0.0)
        elif isinstance(expr, bool) or isinstance(expr, sympy.logic.boolalg.Boolean):
            return torch.full(self._input_shape, 0.0 if expr else 1.0,
                              device=self.device)
        else:
            print("WARNING: evaluation expresson as equation", expr, type(expr))
            return self._eval(expr)

    def _eval(self, expr: sympy.Expr) -> torch.Tensor:
        if (isinstance(expr, float) or isinstance(expr, int)
                or expr.func == sympy.Float
                or expr.func == sympy.Integer
                or expr.func == sympy.core.numbers.Rational
                or expr.func == sympy.core.numbers.NegativeOne
                or expr.func == sympy.core.numbers.Zero
                or expr.func == sympy.core.numbers.One
                or expr.func == sympy.core.numbers.Pi
                or expr.func == sympy.core.numbers.Half):
            return torch.full(self._input_shape, float(expr),
                              device=self.device)
        elif expr.func == sympy.Symbol:
            if expr.name not in self.input_names:
                raise ValueError("unknown variable " + expr.name)
            return self._input_data[self.input_names.index(expr.name)]
        elif expr.func == BooleanTrue or expr.func == BooleanFalse:
            return torch.full(self._input_shape, bool(expr),
                              device=self.device)
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
        elif expr.func == sympy.log:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.log(value0)
        elif expr.func == sympy.sin:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.sin(value0)
        elif expr.func == sympy.cos:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.cos(value0)
        elif expr.func == sympy.tan:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.tan(value0)
        elif expr.func == sympy.atan:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.atan(value0)
        elif expr.func == sympy.sqrt:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.sqrt(value0)
        elif expr.func == sympy.exp:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.exp(value0)
        elif expr.func == sympy.Abs:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.abs(value0)
        elif expr.func == sympy.Eq:
            assert len(expr.args) == 2
            value0 = self._eval(expr.args[0])
            value1 = self._eval(expr.args[1])
            return value0 == value1
        elif expr.func == sympy.StrictLessThan:
            assert len(expr.args) == 2
            value0 = self._eval(expr.args[0])
            value1 = self._eval(expr.args[1])
            return value0 < value1
        elif expr.func == sympy.LessThan:
            assert len(expr.args) == 2
            value0 = self._eval(expr.args[0])
            value1 = self._eval(expr.args[1])
            return value0 <= value1
        elif expr.func == sympy.StrictGreaterThan:
            assert len(expr.args) == 2
            value0 = self._eval(expr.args[0])
            value1 = self._eval(expr.args[1])
            return value0 > value1
        elif expr.func == sympy.GreaterThan:
            assert len(expr.args) == 2
            value0 = self._eval(expr.args[0])
            value1 = self._eval(expr.args[1])
            return value0 >= value1
        elif expr.func == sympy.And:
            value = self._eval(expr.args[0])
            for arg in expr.args[1:]:
                other = self._eval(arg)
                value = torch.logical_and(value, other)
            return value
        elif expr.func == sympy.Or:
            value = self._eval(expr.args[0])
            for arg in expr.args[1:]:
                other = self._eval(arg)
                value = torch.logical_or(value, other)
            return value
        elif expr.func == sympy.ceiling:
            assert len(expr.args) == 1
            value0 = self._eval(expr.args[0])
            return torch.ceil(value0)
        elif expr.func == sympy.Piecewise:
            # the fallback case must be fully defined
            assert expr.args[-1][1].func == BooleanTrue
            data = self._eval(expr.args[-1][0]).clone()
            for (a, b) in expr.args[-2::-1]:
                a = self._eval(a)
                b = self._eval(b)
                data[b] = a[b]
            return data
        elif isinstance(expr, NeuralFunc):
            assert len(expr.args) == expr.arity
            inputs = [self._eval(arg) for arg in expr.args]
            inputs = torch.stack(inputs, dim=-1)
            output = expr.network(inputs)
            return torch.squeeze(output, dim=-1)
        elif isinstance(expr, sympy.Function):
            values = list()
            for a in expr.args:
                values.append(self._eval(a))
            return expr.func(*values)
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


class Scaler(object):
    """
    A callable wrapper object for SympyFunc, allowing to scale
    equations.
    """

    def __init__(self, func: Callable, scaling: torch.Tensor):
        """
        The scaling must be a tensor of shape [output_size].
        """
        self.func = func
        assert scaling.ndim == 1
        self.scaling = scaling

    def __call__(self, input_data: torch.Tensor,
                 equs_as_float: bool = True) -> torch.Tensor:
        output_data = self.func(input_data, equs_as_float)
        return output_data * self.scaling


class NeuralFunc(sympy.Function):
    @classmethod
    def eval(cls, *args):
        assert len(args) == cls.arity
        if all([arg.is_Number for arg in args]):
            inputs = torch.tensor([float(arg) for arg in args])
            return cls.network(inputs).item()

    def _eval_is_real(self):
        return True


def neural_func(name: str, arity: int, network: torch.nn.Module) -> NeuralFunc:
    """
    Creates a new sympy function with the given name and arity
    implemented using with the provided network.
    """

    network.eval()  # switch the network to evaluation mode

    return type(name, (NeuralFunc, ), {
        "arity": arity,
        "network": network,
    })
