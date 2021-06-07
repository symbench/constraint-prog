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

from typing import Dict

import math
import sympy
import torch

from constraint_prog.sympy_func import SympyFunc
from constraint_prog.point_cloud import PointCloud
from constraint_prog.newton_raphson import newton_raphson


def get_dynamics_equs() -> Dict[str, sympy.Expr]:
    # constants
    earth_gravitation = 9.80665  # in m/s^2

    # time independent parameters
    mass = sympy.Symbol("mass")  # in kg
    inertia = sympy.Symbol("inertia")  # in kg*m (already divided by radial distance)

    # time dependent state variables
    x_pos = sympy.Symbol("x_pos")  # in m
    x_pos = sympy.Symbol("y_pos")  # in m
    angle = sympy.Symbol("angle")  # in radian
    x_vel = sympy.Symbol("x_vel")  # in m/s
    y_vel = sympy.Symbol("y_vel")  # in m/s
    a_vel = sympy.Symbol("a_vel")  # in radian/s

    # type dependent control variables
    trust1 = sympy.Symbol("trust1")  # in N (kg*m/s^2)
    trust2 = sympy.Symbol("trust2")  # in N

    # differential equations
    return {
        "der_x_pos": x_vel,
        "der_y_pos": y_vel,
        "der_angle": a_vel,
        "der_x_vel": - (trust1 + trust2) / mass * sympy.sin(angle),
        "der_y_vel": - earth_gravitation + (trust1 + trust2) / mass * sympy.cos(angle),
        "der_a_vel": (trust1 - trust2) / inertia,
    }


def get_fourier_expr(name: str, order: int) -> sympy.Expr:
    assert order >= 0
    name = name + "_c"
    time = sympy.Symbol("time")
    expr = sympy.Float(0.0)
    for idx in range(order):
        param = sympy.Symbol(name + str(idx))
        if idx % 2 == 0:
            expr += param * sympy.cos(idx // 2 * time)
        else:
            expr += param * sympy.sin((idx + 1) // 2 * time)
    return expr


def get_control_exprs(order: int) -> Dict[str, sympy.Expr]:
    return {
        "trust1": get_fourier_expr("trust1", order),
        "trust2": get_fourier_expr("trust2", order),
    }


class ErrorFunc(object):
    def __init__(self, order: int, steps: int = 100):
        self.steps = steps

        self.dynamics_equs = get_dynamics_equs()
        self.dynamics_func = SympyFunc(self.dynamics_equs.values())
        self.parameters = ["mass", "inertia"]

        self.control_exprs = get_control_exprs(order)
        self.control_func = SympyFunc(self.control_exprs.values())
        self.parameters.extend(self.control_func.input_names)
        self.parameters.remove("time")

        self.state_vars = ["x_pos", "y_pos", "angle", "x_vel", "y_vel", "a_vel"]
        self.trace_vars = ["x_pos", "y_pos", "angle", "x_vel", "y_vel", "a_vel", "trust1", "trust2", "time"]

    def get_trace(self, input_data: torch.Tensor) -> torch.Tensor:
        assert input_data.shape[-1] == len(self.parameters)

        shape = input_data.shape[:-1]
        device = input_data.device
        input_data = input_data.unbind(dim=-1)

        step_data = dict()
        for index, param in enumerate(self.parameters):
            step_data[param] = input_data[index]

        zeros = torch.zeros(shape, dtype=torch.float32, device=device)
        for name in self.state_vars:
            step_data[name] = zeros

        trace = []

        delta_time = math.pi / self.steps
        for step in range(self.steps):
            step_data["time"] = torch.full(shape, step * delta_time, dtype=torch.float32, device=device)
            step_data.update(self.control_func.evaluate2(self.control_exprs, step_data, True))
            step_data_der = self.dynamics_func.evaluate2(self.dynamics_equs, step_data, True)

            trace.append(torch.stack([step_data[name] for name in self.trace_vars], dim=-1))

            for name in self.state_vars:
                step_data[name] = step_data[name] + step_data_der["der_" + name] * delta_time

        return torch.stack(trace, dim=-1)

    def print_equs(self):
        for key, val in self.dynamics_equs.items():
            print("{}: {}".format(key, val))
        for key, val in self.control_exprs.items():
            print("{}: {}".format(key, val))

    def print_trace(self, input_data: torch.Tensor, trace: torch.Tensor):
        assert input_data.ndim == 1 and input_data.shape[0] == len(self.parameters)
        for idx, var in enumerate(self.parameters):
            print("{}: {}".format(var, input_data[idx]))

        assert trace.ndim == 2 and trace.shape[0] == len(self.trace_vars)
        for idx, var in enumerate(self.trace_vars):
            print("{}: {}".format(var, list(trace[idx].numpy())))

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        assert input_data.shape[-1] == len(self.parameters)

        shape = input_data.shape[:-1]
        device = input_data.device
        input_data = input_data.unbind(dim=-1)

        step_data = dict()
        for index, param in enumerate(self.parameters):
            step_data[param] = input_data[index]

        zeros = torch.zeros(shape, dtype=torch.float32, device=device)
        for name in self.state_vars:
            step_data[name] = zeros

        delta_time = math.pi / self.steps
        for step in range(self.steps):
            step_data["time"] = torch.full(shape, step * delta_time, dtype=torch.float32, device=device)
            step_data.update(self.control_func.evaluate2(self.control_exprs, step_data, True))
            step_data_der = self.dynamics_func.evaluate2(self.dynamics_equs, step_data, True)

            for name in self.state_vars:
                step_data[name] = step_data[name] + step_data_der["der_" + name] * delta_time

        output_data = [
            step_data["x_pos"],
            step_data["y_pos"],
            step_data["angle"] - math.pi * 2.0,
            step_data["x_vel"],
            step_data["y_vel"],
            step_data["a_vel"],
        ]
        return torch.stack(output_data, dim=-1)


if __name__ == '__main__':
    func = ErrorFunc(order=11, steps=100)

    points = PointCloud.generate(func.parameters,
                                 [-1.0] * len(func.parameters),
                                 [1.0] * len(func.parameters),
                                 num_points=1000)

    bounding_box = torch.zeros((2, len(func.parameters)), dtype=torch.float32)
    bounding_box[0, :] = -1
    bounding_box[0, 0] = 0
    bounding_box[0, 1] = 0
    bounding_box[1, :] = 1

    points.float_data = newton_raphson(
        func,
        points.float_data,
        bounding_box=bounding_box,
        method="mmclip")

    errors = PointCloud(
        ["x_pos_err", "y_pos_err", "angle_err", "x_vel_err", "y_vel_err", "a_vel_err"],
        func(points.float_data))
    points = points.prune_by_tolerances(errors, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    # print(points.float_data)

    errors = PointCloud(
        ["x_pos_err", "y_pos_err", "angle_err", "x_vel_err", "y_vel_err", "a_vel_err"],
        func(points.float_data))
    print(errors.float_data.pow(2.0).sum(dim=-1))

    func.print_equs()
    trace = func.get_trace(points.float_data)
    func.print_trace(points.float_data[0], trace[0])
