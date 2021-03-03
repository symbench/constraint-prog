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

import torch
import sympy

from constraint_prog.newton_raphson import newton_raphson
from constraint_prog.sympy_func import SympyFunc


def test1():
    # pylint: disable=unused-variable
    ocean_density = 1023.6  # in kg/m^3 for seawater
    yield_strength = 34000.0 * 6894.7572932  # in Pa = N/m^2 = kg/ms^2 for steel
    earth_gravitation = 9.80665  # in m/s^2

    glide_slope = sympy.Symbol("glide_slope")  # radians
    dive_depth = sympy.Symbol("dive_depth")  # m
    horizontal_distance = sympy.Symbol("horizontal_distance")  # m
    nominal_horizontal_speed = sympy.Symbol("nominal_horizontal_speed")  # m/s
    inner_hull_diameter = sympy.Symbol("inner_hull_diamater")  # m
    outer_hull_diameter = sympy.Symbol("outer_hull_diameter")  # m

    nominal_glide_speed = nominal_horizontal_speed / \
        sympy.cos(glide_slope)  # m/s
    nominal_vertical_speed = nominal_glide_speed * \
        sympy.sin(glide_slope)  # m/s
    nominal_dive_duration = 2.0 * dive_depth / nominal_vertical_speed  # s
    actual_dive_duration = 1.1 * nominal_dive_duration  # s
    horizontal_distance_per_dive = nominal_horizontal_speed * \
        nominal_dive_duration  # m
    dives_per_mission = horizontal_distance / horizontal_distance_per_dive
    mission_duration = dives_per_mission * actual_dive_duration  # s

    # hull thickness constraint
    inner_hull_radius = 0.5 * inner_hull_diameter  # m
    outer_hull_radius = 0.5 * outer_hull_diameter  # m
    hull_thickness = outer_hull_radius - inner_hull_radius  # m
    thin_hoop_condition = hull_thickness < 0.1 * inner_hull_radius  # bool
    thin_hoop_coefficient = inner_hull_radius / hull_thickness
    thick_hoop_coefficient = (outer_hull_radius ** 2 + inner_hull_radius ** 2) / \
        (outer_hull_radius ** 2 - inner_hull_radius ** 2)
    hoop_coefficient = sympy.Piecewise(
        (thin_hoop_coefficient, thin_hoop_condition),
        (thick_hoop_coefficient, True))
    maximum_hull_pressure = ocean_density * earth_gravitation * dive_depth  # in Pa
    maximum_hoop_stress = maximum_hull_pressure * hoop_coefficient  # CHECK THIS
    hull_thickness_equation = maximum_hoop_stress * 1.25 - yield_strength

    param1 = sympy.Symbol("param1")
    print(sympy.srepr(thin_hoop_coefficient))

    # print(sympy.srepr(thin_hoop_coefficient))
    func = SympyFunc([
        # hull_thickness_equation,
        # dive_depth - param1 ** 2,
        # 10 * hull_thickness - inner_hull_radius,
        thin_hoop_coefficient - 10,
        sympy.Min(hull_thickness - 0.1, 0),
    ])
    print(func.input_names)

    input_data = torch.randn([10, func.input_size]) * 50.0
    input_data = newton_raphson(func, input_data, num_iter=20)
    print(input_data)
    print(func(input_data))
    print(func.evaluate([thin_hoop_coefficient], input_data))
    # print(func.evaluate([thin_hoop_condition], input_data))
    # print(func.evaluate(
    #     [hoop_coefficient, maximum_hull_pressure], input_data))


def test2():
    ocean_density = 1023.6  # in kg/m^3 for seawater
    yield_strength = 34000.0 * 6894.7572932  # in Pa = N/m^2 = kg/ms^2 for steel
    earth_gravitation = 9.80665  # in m/s^2

    dive_depth = sympy.Symbol("dive_depth")  # m
    inner_hull_diameter = sympy.Symbol("inner_hull_diamater")  # m
    outer_hull_diameter = sympy.Symbol("outer_hull_diameter")  # m

    inner_hull_radius = 0.5 * inner_hull_diameter  # m
    outer_hull_radius = 0.5 * outer_hull_diameter  # m
    hull_thickness = outer_hull_radius - inner_hull_radius  # m

    thin_hoop_coefficient = sympy.Symbol("thin_hoop_coefficient")
    thin_hoop_equation = sympy.Eq(
        thin_hoop_coefficient * hull_thickness, inner_hull_radius)

    thick_hoop_coefficient = sympy.Symbol("thick_hoop_coefficient")
    thick_hoop_equation = sympy.Eq(
        thick_hoop_coefficient *
        (outer_hull_radius ** 2 - inner_hull_radius ** 2),
        outer_hull_radius ** 2 + inner_hull_radius ** 2)

    thin_hoop_condition = hull_thickness < 0.1 * inner_hull_radius  # bool
    hoop_coefficient = sympy.Piecewise(
        (thin_hoop_coefficient, thin_hoop_condition),
        (thick_hoop_coefficient, True))

    maximum_hull_pressure = ocean_density * earth_gravitation * dive_depth  # in Pa
    maximum_hoop_stress = maximum_hull_pressure * hoop_coefficient  # CHECK THIS
    hull_thickness_equation = sympy.Eq(
        maximum_hoop_stress * 1.25, yield_strength)

    func = SympyFunc([
        thin_hoop_equation,
        thick_hoop_equation,
        hull_thickness_equation,
        sympy.Min(dive_depth, 0),
        sympy.Min(hull_thickness - 0.01, 0),
        sympy.Min(inner_hull_diameter - 0.1, 0),
    ])
    print(func.input_names)

    input_data = torch.randn([100, func.input_size]) * 10.0
    input_data = newton_raphson(func, input_data, num_iter=20, epsilon=0.1)
    print(input_data)
    output_data = func(input_data)
    print(output_data)
    result_data = func.evaluate(
        [inner_hull_diameter, hull_thickness, dive_depth], input_data)
    result_data = torch.cat(
        (result_data, output_data.norm(dim=-1, keepdim=True)), dim=-1)
    print(result_data)
