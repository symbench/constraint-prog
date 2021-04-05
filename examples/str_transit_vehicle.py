#!/usr/bin/env python3
# Copyright (C) 2021, Will Hedgecock, Miklos Maroti
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

import mpmath
import sympy

from sympy import pi, cos, sin, tan, sqrt, log, exp


# CONSTANTS AND ASSUMED VALUES ----------------------------------------------------------------------------------------

AVERAGE_COEFFICIENT_OF_FORM_DRAG = 0.15
AVERAGE_COEFFICIENT_OF_LIFT = 0.55

MATERIAL_YIELD_SAFETY_FACTOR = 0.8
BUOYANCY_FLUID_SAFETY_FACTOR = 1.25
BATTERY_CAPACITY_SAFETY_FACTOR = 0.8
PUMP_DURATION_INCREASE_FACTOR = 1.1
FINENESS_RATIO_MIN = 5.0
FINENESS_RATIO_MAX = 9.0


# DESIGN PARAMETERS ---------------------------------------------------------------------------------------------------

salinity = sympy.Symbol('salinity')  # PSU
water_temperature = sympy.Symbol('water_temperature')  # C
dive_depth = sympy.Symbol('dive_depth')  # m
horizontal_range = sympy.Symbol('horizontal_range')  # m
time_endurance = sympy.Symbol('time_endurance')  # s
mission_latitude = sympy.Symbol('mission_latitude')  # decimal degrees
nominal_horizontal_speed = sympy.Symbol('nominal_horizontal_speed')  # m/s
maximum_horizontal_speed = sympy.Symbol('maximum_horizontal_speed')  # m/s
hull_thickness = sympy.Symbol('hull_thickness')  # m
hull_length_external = sympy.Symbol('hull_length_external')  # m
hull_radius_external = sympy.Symbol('hull_radius_external')  # m
hull_material_yield = sympy.Symbol('hull_material_yield')  # Pa = kg/(m*s^2)
hull_material_density = sympy.Symbol('hull_material_density')  # kg/m^3
payload_length = sympy.Symbol('payload_length')  # m
payload_mass = sympy.Symbol('payload_mass')  # kg
payload_power = sympy.Symbol('payload_power')  # W
hotel_volume = sympy.Symbol('hotel_volume')  # m^3
hotel_mass = sympy.Symbol('hotel_mass')  # kg
hotel_power = sympy.Symbol('hotel_power')  # W
propulsion_fluid_density = sympy.Symbol('propulsion_fluid_density')  # kg/m^3
propulsion_fluid_volume = sympy.Symbol('propulsion_fluid_volume')  # m^3
propulsion_engine_efficiency = sympy.Symbol('propulsion_engine_efficiency')  # %
battery_cell_energy_capacity = sympy.Symbol('battery_cell_energy_capacity')  # J
battery_cell_radius = sympy.Symbol('battery_cell_radius')  # m
battery_cell_length = sympy.Symbol('battery_cell_length')  # m
battery_cell_mass = sympy.Symbol('battery_cell_mass')  # kg
battery_cell_quantity = sympy.Symbol('battery_cell_quantity')
wing_mass = sympy.Symbol('wing_mass')  # kg
wing_area = sympy.Symbol('wing_area')  # m^2
wing_volume = sympy.Symbol('wing_volume')  # m^3
wing_span = sympy.Symbol('wing_span')  # %
wing_span_efficiency = sympy.Symbol('wing_span_efficiency')  # %


# HULL-BASED GEOMETRIC EXPRESSIONS ------------------------------------------------------------------------------------

hull_radius_internal = hull_radius_external - hull_thickness  # m
hull_length_internal = hull_length_external - 2*hull_thickness  # m
hull_volume_external = pi * hull_radius_external**2 * hull_length_external  # m^3
hull_volume_internal = pi * hull_radius_internal**2 * hull_length_internal  # m^3
hull_wetted_area = (2 * pi * hull_radius_external * hull_length_external) + (2 * pi * hull_radius_external**2)  # m^2
hull_frontal_area = pi * hull_radius_external**2  # m^2
hull_mass = (hull_volume_external - hull_volume_internal) * hull_material_density  # kg
payload_volume = pi * hull_radius_internal**2 * payload_length  # m^3


# OCEAN AND LOCATION-BASED EXPRESSIONS --------------------------------------------------------------------------------

gravitational_acceleration = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * sin(mpmath.radians(mission_latitude))**2) * \
    sin(mpmath.radians(mission_latitude))**2)  # m/s^2
pressure_at_depth = 10000 * (2.398599584e05 - sqrt(5.753279964e10 - (4.833657881e05 * dive_depth)))  # Pa = kg/(m*s^2)
water_density = 1000 + ((((((((((((5.2787e-8 * water_temperature) - 6.12293e-6) * water_temperature) + \
    8.50935e-5) + ((((9.1697e-10 * water_temperature) + 2.0816e-8) * water_temperature) - 9.9348e-7) * \
    salinity) * pressure_at_depth * 1e-5) + ((((1.91075e-4 * sqrt(salinity)) + \
    ((((-1.6078e-6 * water_temperature) - 1.0981e-5) * water_temperature) + 2.2838e-3)) * salinity) + \
    ((((((-5.77905e-7 * water_temperature) + 1.16092e-4) * water_temperature) + 1.43713e-3) * \
    water_temperature) + 3.239908))) * pressure_at_depth * 1e-5) + \
    ((((((((-5.3009e-4 * water_temperature) + 1.6483e-2) * water_temperature) + 7.944e-2) * sqrt(salinity)) + \
    ((((((-6.1670e-5 * water_temperature) + 1.09987e-2) * water_temperature) - 0.603459) * \
    water_temperature) + 54.6746)) * salinity) + ((((((((-5.155288e-5 * water_temperature) + 1.360477e-2) * \
    water_temperature) - 2.327105) * water_temperature) + 148.4206) * water_temperature) + 19652.21))) * \
    ((((4.8314e-4 * salinity) + (((((-1.6546e-6 * water_temperature) + 1.0227e-4) * water_temperature) - \
    5.72466e-3) * sqrt(salinity)) + ((((((((5.3875e-9 * water_temperature) - 8.2467e-7) * \
    water_temperature) + 7.6438e-5) * water_temperature) - 4.0899e-3) * water_temperature) + 0.824493)) * \
    salinity) + (((((((((6.536332e-9 * water_temperature) - 1.120083e-6) * water_temperature) + \
    1.001685e-4) * water_temperature) - 9.095290e-3) * water_temperature) + 6.793952e-2) * water_temperature - \
    0.157406))) + (pressure_at_depth * 1e-2)) / ((((((((((5.2787e-8 * water_temperature) - 6.12293e-6) * \
    water_temperature) + 8.50935e-5) + ((((9.1697e-10 * water_temperature) + 2.0816e-8) * water_temperature) - \
    9.9348e-7) * salinity) * pressure_at_depth * 1e-5) + ((((1.91075e-4 * sqrt(salinity)) + ((((-1.6078e-6 * \
    water_temperature) - 1.0981e-5) * water_temperature) + 2.2838e-3)) * salinity) + ((((((-5.77905e-7 * \
    water_temperature) + 1.16092e-4) * water_temperature) + 1.43713e-3) * water_temperature) + 3.239908))) * \
    pressure_at_depth * 1e-5) + ((((((((-5.3009e-4 * water_temperature) + 1.6483e-2) * water_temperature) + \
    7.944e-2) * sqrt(salinity)) + ((((((-6.1670e-5 * water_temperature) + 1.09987e-2) * water_temperature) - \
    0.603459) * water_temperature) + 54.6746)) * salinity) + ((((((((-5.155288e-5 * water_temperature) + \
    1.360477e-2) * water_temperature) - 2.327105) * water_temperature) + 148.4206) * water_temperature) + \
    19652.21))) - (pressure_at_depth * 1e-5)))  # kg/m^3
freshwater_absolute_viscosity = ((0.00000000000277388442 * water_temperature**6) - \
    (0.00000000124359703683 * water_temperature**5) + (0.00000022981389243372 * water_temperature**4) - \
    (0.00002310372106867350 * water_temperature**3) + (0.00143393546700877000 * water_temperature**2) - \
    (0.06064140920049450000 * water_temperature) + 1.79157254681817000000) / 1000  # Pa*s = kg/(m*s)
seawater_absolute_viscosity = freshwater_absolute_viscosity * \
    (1.0 + ((1.541 + (1.998e-2 * water_temperature) - (9.52e-5 * water_temperature**2)) * \
    salinity * 0.001) + ((7.974 - (7.561e-2 * water_temperature) + (4.724e-4 * water_temperature**2)) * \
    (salinity * 0.001)**2))  # Pa*s = kg/(m*s)
seawater_kinematic_viscosity = seawater_absolute_viscosity / water_density  # m^2/s


# STRESS-BASED EXPRESSIONS --------------------------------------------------------------------------------------------

hoop_stress_thin_wall_condition = sympy.Lt(hull_thickness, hull_radius_internal / 10)
hoop_stress_equation_thin = hull_radius_internal / hull_thickness
hoop_stress_equation_thick = (hull_radius_external**2 + hull_radius_internal**2) / \
    (hull_radius_external**2 - hull_radius_internal**2)
hoop_stress_coefficient = sympy.Piecewise(
    (hoop_stress_equation_thin, hoop_stress_thin_wall_condition),
    (hoop_stress_equation_thick, True))
hoop_stress = hoop_stress_coefficient * pressure_at_depth  # Pa


# MISSION PARAMETER EXPRESSIONS ---------------------------------------------------------------------------------------
# TODO: Parameterize these equations

# Glide slope = tan^-1(C_D / C_L)
#atan(glide_slope) = coefficient_of_lift / coefficient_of_drag
glide_slope = 0.610865
nominal_glide_speed = nominal_horizontal_speed / cos(glide_slope)
nominal_vertical_speed = nominal_horizontal_speed * tan(glide_slope)
maximum_glide_speed = maximum_horizontal_speed / cos(glide_slope)
maximum_vertical_speed = maximum_horizontal_speed * tan(glide_slope)
dive_duration = PUMP_DURATION_INCREASE_FACTOR * (2 * dive_depth / nominal_vertical_speed)
dive_horizontal_distance = nominal_horizontal_speed * (2 * dive_depth / nominal_vertical_speed)
dives_per_mission = horizontal_range / dive_horizontal_distance
mission_duration = dives_per_mission * dive_duration


# LIFT-BASED EXPRESSIONS ----------------------------------------------------------------------------------------------
# TODO: Create lift model from parameters
coefficient_of_lift = AVERAGE_COEFFICIENT_OF_LIFT
wing_aspect_ratio = wing_span**2 / wing_area
nominal_lift = wing_area * coefficient_of_lift * water_density * nominal_glide_speed**2 / 2  # N
maximum_lift = wing_area * coefficient_of_lift * water_density * maximum_glide_speed**2 / 2  # N


# DRAG-BASED EXPRESSIONS ----------------------------------------------------------------------------------------------

nominal_reynolds_number = nominal_glide_speed * hull_length_external / seawater_kinematic_viscosity
maximum_reynolds_number = maximum_glide_speed * hull_length_external / seawater_kinematic_viscosity
coefficient_of_induced_drag = coefficient_of_lift**2 / (pi * wing_span_efficiency * wing_aspect_ratio)
nominal_speed_coefficient_of_friction = (0.0776 / (log(nominal_reynolds_number, 10) - 1.88)**2) + \
    (60.0 / nominal_reynolds_number)
nominal_speed_coefficient_of_form = AVERAGE_COEFFICIENT_OF_FORM_DRAG
maximum_speed_coefficient_of_friction = (0.0776 / (log(maximum_reynolds_number, 10) - 1.88)**2) + \
    (60.0 / maximum_reynolds_number)
maximum_speed_coefficient_of_form = AVERAGE_COEFFICIENT_OF_FORM_DRAG
nominal_speed_friction_drag = hull_wetted_area * nominal_speed_coefficient_of_friction * water_density * \
    nominal_glide_speed**2 / 2  # N
nominal_speed_form_drag = hull_frontal_area * nominal_speed_coefficient_of_form * water_density * \
    nominal_glide_speed**2 / 2  # N
nominal_speed_induced_drag = wing_area * coefficient_of_induced_drag * water_density * \
    nominal_glide_speed**2 / 2  # N
maximum_speed_friction_drag = hull_wetted_area * maximum_speed_coefficient_of_friction * water_density * \
    maximum_glide_speed**2 / 2  # N
maximum_speed_form_drag = hull_frontal_area * maximum_speed_coefficient_of_form * water_density * \
    maximum_glide_speed**2 / 2  # N
maximum_speed_induced_drag = wing_area * coefficient_of_induced_drag * water_density * \
    maximum_glide_speed**2 / 2  # N
nominal_drag = nominal_speed_form_drag + nominal_speed_friction_drag + nominal_speed_induced_drag  # N
maximum_drag = maximum_speed_form_drag + maximum_speed_friction_drag + maximum_speed_induced_drag  # N


# PROPULSION-BASED EXPRESSIONS ----------------------------------------------------------------------------------------

nominal_net_buoyancy = nominal_drag / sin(glide_slope)  # N
maximum_net_buoyancy = maximum_drag / sin(glide_slope)  # N
nominal_buoyancy_equivalent_mass = 2.0 * nominal_net_buoyancy / gravitational_acceleration  # kg
maximum_buoyancy_equivalent_mass = 2.0 * maximum_net_buoyancy / gravitational_acceleration  # kg
nominal_buoyancy_fluid_required = nominal_buoyancy_equivalent_mass / water_density  # m^3
maximum_buoyancy_fluid_required = maximum_buoyancy_equivalent_mass / water_density  # m^3
propulsion_mass = maximum_buoyancy_fluid_required * BUOYANCY_FLUID_SAFETY_FACTOR * propulsion_fluid_density  # kg


# ENERGY-BASED EXPRESSIONS --------------------------------------------------------------------------------------------

propulsion_energy_per_dive = 10000 * nominal_buoyancy_fluid_required * dive_depth / propulsion_engine_efficiency  # J
propulsion_energy = propulsion_energy_per_dive * dives_per_mission  # J
hotel_energy = hotel_power * time_endurance  # J
payload_energy = payload_power * time_endurance  # J
mission_energy = propulsion_energy + hotel_energy + payload_energy  # J
num_battery_cells_required = mission_energy / (battery_cell_energy_capacity * BATTERY_CAPACITY_SAFETY_FACTOR)
battery_volume = num_battery_cells_required * pi * battery_cell_radius**2 * battery_cell_length  # m^3
battery_mass = num_battery_cells_required * battery_cell_mass  # kg


# MODEL CONSTRAINT EQUATIONS ------------------------------------------------------------------------------------------

# Mission Duration Constraints
mission_duration_constraint = sympy.Le(mission_duration, time_endurance)

# Neutral Buoyancy Constraint
uuv_mass = hull_mass + battery_mass + payload_mass + propulsion_mass + hotel_mass + wing_mass  # kg
uuv_neutral_buoyancy_displaced_volume = hull_volume_external + wing_volume + (propulsion_fluid_volume / 2)  # m^3
neutral_buoyancy_constraint = sympy.Eq(uuv_mass, uuv_neutral_buoyancy_displaced_volume * water_density)

# Internal Hull Volume Constraint
internal_hull_volume_constraint = sympy.Ge(hull_volume_internal, \
    battery_volume + hotel_volume + payload_volume + propulsion_fluid_volume)

# Hull Integrity Constraint
hull_integrity_constraint = sympy.Ge(hull_material_yield * MATERIAL_YIELD_SAFETY_FACTOR, hoop_stress)

# Propulsion Fluid Volume Constraint
propulsion_fluid_volume_constraint = sympy.Ge(propulsion_fluid_volume, \
    maximum_buoyancy_fluid_required * BUOYANCY_FLUID_SAFETY_FACTOR)  # m^3

# Power Supply Constraint
power_supply_constraint = sympy.Ge(battery_cell_quantity, num_battery_cells_required)

# Fineness Ratio Constraints
fineness_ratio_min_constraint = sympy.Ge(hull_length_external, FINENESS_RATIO_MIN * 2 * hull_radius_external)
fineness_ratio_max_constraint = sympy.Le(hull_length_external, FINENESS_RATIO_MAX * 2 * hull_radius_external)

# Geometry Constraints
payload_length_constraint = sympy.Le(payload_length, hull_length_internal / 2)


# USER-SPECIFIC CONSTRAINT EQUATIONS ----------------------------------------------------------------------------------

# External Hull Volume Constraint
external_hull_volume_constraint = sympy.Le(hull_volume_external, 3.471145623)
