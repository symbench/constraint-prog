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

import sympy

ocean_density = 1023.6  # in kg/m^3 for seawater
hull_yield_strength = 34000.0 * 6894.7572932  # in Pa = N/m^2 = kg/ms^2
earth_gravitation = 9.80665  # in m/s^2
buoyancy_engine_efficiency = 0.5

glide_slope = sympy.Symbol("glide_slope")  # radians
maximal_dive_depth = sympy.Symbol("maximal_dive_depth")  # m
horizontal_distance = sympy.Symbol("horizontal_distance")  # m
nominal_horizontal_speed = sympy.Symbol("nominal_horizontal_speed")  # m/s
maximal_horizontal_speed = sympy.Symbol("nominal_horizontal_speed")  # m/s
inner_hull_diameter = sympy.Symbol("inner_hull_diamater")  # m
outer_hull_diameter = sympy.Symbol("outer_hull_diameter")  # m
mission_duration = sympy.Symbol("mission_duration")  # s
drag_coefficient = sympy.Symbol("drag_coefficient")

# mission duration equation (step 1, page 11)

nominal_glide_speed = nominal_horizontal_speed / sympy.cos(glide_slope)  # m/s
nominal_vertical_speed = nominal_glide_speed * sympy.sin(glide_slope)  # m/s
maximal_glide_speed = maximal_horizontal_speed / sympy.cos(glide_slope)  # m/s
maximal_vertical_speed = maximal_glide_speed * sympy.sin(glide_slope)  # m/s
nominal_dive_duration = 2.0 * maximal_dive_depth / nominal_vertical_speed  # s
actual_dive_duration = 1.1 * nominal_dive_duration  # s
horizontal_distance_per_dive = nominal_horizontal_speed * \
    nominal_dive_duration  # m
dives_per_mission = horizontal_distance / horizontal_distance_per_dive
mission_duration_equation = sympy.Eq(
    mission_duration, dives_per_mission * actual_dive_duration)  # s

# hull thickness equation (step 3, page 15)

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

maximal_hull_pressure = ocean_density * \
    earth_gravitation * maximal_dive_depth  # in Pa
maximal_hoop_stress = maximal_hull_pressure * hoop_coefficient
hull_thickness_equation = sympy.Eq(
    maximal_hoop_stress * 1.25, hull_yield_strength)

# calculate drag forces (step 4, page 19)

projected_frontal_area = sympy.pi * (outer_hull_radius ** 2.0)  # m^2
nominal_drag_force = 0.5 * ocean_density * (nominal_glide_speed ** 2.0) * \
    drag_coefficient * projected_frontal_area  # N
nominal_net_buoyancy = nominal_drag_force / sympy.sin(glide_slope)  # N
maximal_drag_force = 0.5 * ocean_density * (maximal_glide_speed ** 2.0) * \
    drag_coefficient * projected_frontal_area  # N
maximal_net_buoyancy = maximal_drag_force / sympy.sin(glide_slope)  # N

# buoyancy calculation (step 5, page 23)

maximal_buoyancy_equivalent_mass = 2.0 * maximal_net_buoyancy / \
    earth_gravitation  # kg
maximal_buoyancy_equivalent_volume = maximal_buoyancy_equivalent_mass / \
    ocean_density  # m^3
buoyancy_reservoir_volume = maximal_buoyancy_equivalent_volume + 460e-6  # m^3

# total propulsion energy (step 6, page 24)

nominal_buoyancy_equivalent_mass = 2.0 * nominal_net_buoyancy / \
    earth_gravitation  # kg
nominal_buoyancy_equivalent_volume = nominal_buoyancy_equivalent_mass / \
    ocean_density  # m^3
pump_energy_per_dive = 0.01 * nominal_buoyancy_equivalent_volume * \
    maximal_dive_depth / buoyancy_engine_efficiency  # in J
total_propulsion_energy = pump_energy_per_dive * dives_per_mission
