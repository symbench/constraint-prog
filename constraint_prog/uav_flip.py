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

import sympy


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
    name = name + "_"
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


if __name__ == '__main__':
    equs = get_dynamics_equs()
    print(list(equs.keys()))
    hihi = get_fourier_expr("hihi", 5)
    print(hihi)
