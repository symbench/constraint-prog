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


# CONSTRAINT EQUATIONS ------------------------------------------------------------------------------------------------

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


# TESTS ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Test hull geometry equations
    print('\nHull Geometry Tests:')
    print('Given hull_radius_external = 0.13 m, hull_length_external = 1.82 m, hull_thickness = 0.013347245 m, hull_material_density = 2700 kg/m^3:\n')
    print('\thull_radius_internal (m): {}'.format(hull_radius_internal.subs([(hull_radius_external, 0.13), (hull_thickness, 0.013347245)]).evalf()))
    print('\thull_length_internal (m): {}'.format(hull_length_internal.subs([(hull_length_external, 1.82), (hull_thickness, 0.013347245)]).evalf()))
    print('\thull_volume_external (m^3): {}'.format(hull_volume_external.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82)]).evalf()))
    print('\thull_volume_internal (m^3): {}'.format(hull_volume_internal.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (hull_thickness, 0.013347245)]).evalf()))
    print('\thull_wetted_area (m^2): {}'.format(hull_wetted_area.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82)]).evalf()))
    print('\thull_frontal_area (m^2): {}'.format(hull_frontal_area.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82)]).evalf()))
    print('\thull_mass (kg): {}'.format(hull_mass.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (hull_thickness, 0.013347245), (hull_material_density, 2700)]).evalf()))

    # Test ocean and location-based equations
    print('\nOcean Parameter Tests:')
    print('Given mission_latitude = 45 deg, dive_depth = 2000 m, water_temperature = 10 C, salinity = 35 PSU:\n')
    print('\tgravitational_acceleration (m/s^2): {}'.format(gravitational_acceleration.subs([(mission_latitude, 45)]).evalf()))
    print('\tpressure_at_depth (Pa): {}'.format(pressure_at_depth.subs([(dive_depth, 2000)]).evalf()))
    print('\twater_density (kg/m^3): {}'.format(water_density.subs([(water_temperature, 10), (salinity, 35), (dive_depth, 2000)]).evalf()))
    print('\tfreshwater_absolute_viscosity (Pa*s): {}'.format(freshwater_absolute_viscosity.subs([(water_temperature, 10)]).evalf()))
    print('\tseawater_absolute_viscosity (Pa*s): {}'.format(seawater_absolute_viscosity.subs([(water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tseawater_kinematic_viscosity (m^2/s): {}'.format(seawater_kinematic_viscosity.subs([(water_temperature, 10), (salinity, 35), (dive_depth, 2000)]).evalf()))

    # Test material stress equations
    print('\nMaterial Stress Tests:')
    print('Given hull_radius_external = 0.13 m, hull_thickness = 0.013347245 m, dive_depth = 2000 m:\n')
    print('\thoop_stress_equation_thin: {}'.format(hoop_stress_equation_thin.subs([(hull_radius_external, 0.13), (hull_thickness, 0.013347245)]).evalf()))
    print('\thoop_stress_equation_thick: {}'.format(hoop_stress_equation_thick.subs([(hull_radius_external, 0.13), (hull_thickness, 0.013347245)]).evalf()))
    print('\thoop_stress_coefficient: {}'.format(hoop_stress_coefficient.subs([(hull_radius_external, 0.13), (hull_thickness, 0.013347245)]).evalf()))
    print('\thoop_stress (Pa): {}'.format(hoop_stress.subs([(hull_radius_external, 0.13), (hull_thickness, 0.013347245), (dive_depth, 2000)]).evalf()))

    # Test mission parameter equations
    #TODO:
    print('\nMission Parameter Tests:')
    print('Given :\n')
    print('\tglide_slope (deg): {}'.format(glide_slope))
    print('\tnominal_glide_speed (m/s): {}'.format(nominal_glide_speed))
    print('\tmaximum_glide_speed (m/s): {}'.format(maximum_glide_speed))

    # Test lift computation equations
    print('\nLift Computation Tests:')
    print('Given wing_area = 0.1 m^2, wing_span = 0.8 m, dive_depth = 2000 m, water_temperature = 10 C, salinity = 35 PSU:\n')
    print('\twing_aspect_ratio: {}'.format(wing_aspect_ratio.subs([(wing_area, 0.1), (wing_span, 0.8)]).evalf()))
    print('\tnominal_lift (N): {}'.format(nominal_lift.subs([(wing_area, 0.1), (wing_span, 0.8), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tmaximum_lift (N): {}'.format(maximum_lift.subs([(wing_area, 0.1), (wing_span, 0.8), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))

    # Test drag computation equations
    print('\nDrag Computation Tests:')
    print('Given hull_radius_external = 0.13 m, hull_length_external = 1.82 m, dive_depth = 2000 m, water_temperature = 10 C, salinity = 35 PSU, wing_area = 0.1 m^2, wing_span = 0.8 m, wing_span_efficiency = 0.8:\n')
    print('\tnominal_reynolds_number: {}'.format(nominal_reynolds_number.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tnominal_speed_coefficient_of_friction: {}'.format(nominal_speed_coefficient_of_friction.subs([(hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tnominal_speed_coefficient_of_form: {}'.format(nominal_speed_coefficient_of_form))
    print('\tnominal_speed_friction_drag (N): {}'.format(nominal_speed_friction_drag.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tnominal_speed_form_drag (N): {}'.format(nominal_speed_form_drag.subs([(hull_radius_external, 0.13), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tnominal_speed_induced_drag (N): {}'.format(nominal_speed_induced_drag.subs([(dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tnominal_drag (N): {}'.format(nominal_drag.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tmaximum_reynolds_number: {}'.format(maximum_reynolds_number.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tmaximum_speed_coefficient_of_friction: {}'.format(maximum_speed_coefficient_of_friction.subs([(hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tmaximum_speed_coefficient_of_form: {}'.format(maximum_speed_coefficient_of_form))
    print('\tmaximum_speed_friction_drag (N): {}'.format(maximum_speed_friction_drag.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tmaximum_speed_form_drag (N): {}'.format(maximum_speed_form_drag.subs([(hull_radius_external, 0.13), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]).evalf()))
    print('\tmaximum_speed_induced_drag (N): {}'.format(maximum_speed_induced_drag.subs([(dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tmaximum_drag (N): {}'.format(maximum_drag.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))

    # Test propulsion equations
    print('\nPropulsion Equation Tests:')
    print('Given hull_radius_external = 0.13 m, hull_length_external = 1.82 m, mission_latitude = 45 deg, dive_depth = 2000 m, water_temperature = 10 C, salinity = 35 PSU, propulsion_fluid_density = 950 kg/m^3, wing_area = 0.1 m^2, wing_span = 0.8 m, wing_span_efficiency = 0.8:\n')
    print('\tnominal_net_buoyancy (N): {}'.format(nominal_net_buoyancy.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tmaximum_net_buoyancy (N): {}'.format(maximum_net_buoyancy.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tnominal_buoyancy_equivalent_mass (kg): {}'.format(nominal_buoyancy_equivalent_mass.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tmaximum_buoyancy_equivalent_mass (kg): {}'.format(maximum_buoyancy_equivalent_mass.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tnominal_buoyancy_fluid_required (m^3): {}'.format(nominal_buoyancy_fluid_required.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tmaximum_buoyancy_fluid_required (m^3): {}'.format(maximum_buoyancy_fluid_required.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tpropulsion_fluid_volume (m^3): {}'.format(propulsion_fluid_volume_constraint.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf().rhs))
    print('\tpropulsion_mass (kg): {}'.format(propulsion_mass.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_fluid_density, 950), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))

    # Test energy equations
    print('\nEnergy Equation Tests:')
    print('Given hull_radius_external = 0.13 m, hull_length_external = 1.82 m, mission_latitude = 45 deg, dive_depth = 2000 m, water_temperature = 10 C, salinity = 35 PSU, wing_area = 0.1 m^2, wing_span = 0.8 m, wing_span_efficiency = 0.8, propulsion_engine_efficiency = 0.5, hotel_power = 5 W, payload_power = 20 W, time_endurance = 3306173 s, battery_cell_energy_capacity = 144000 J, battery_cell_length = 0.0615 m, battery_cell_radius = 0.0171 m, battery_cell_mass = 0.135 kg:\n')
    print('\tpropulsion_energy_per_dive (J): {}'.format(propulsion_energy_per_dive.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\tpropulsion_energy (J): {}'.format(propulsion_energy.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7)]).evalf()))
    print('\thotel_energy (J): {}'.format(hotel_energy.subs([(hotel_power, 5), (time_endurance, 3306173)]).evalf()))
    print('\tpayload_energy (J): {}'.format(payload_energy.subs([(payload_power, 20), (time_endurance, 3306173)]).evalf()))
    print('\tmission_energy (J): {}'.format(mission_energy.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7), (hotel_power, 5), (payload_power, 20), (time_endurance, 3306173)]).evalf()))
    print('\tbattery_cells_required: {}'.format(num_battery_cells_required.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7), (hotel_power, 5), (payload_power, 20), (time_endurance, 3306173), (battery_cell_energy_capacity, 144000)]).evalf()))
    print('\tbattery_volume (m^3): {}'.format(battery_volume.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7), (hotel_power, 5), (payload_power, 20), (time_endurance, 3306173), (battery_cell_energy_capacity, 144000), (battery_cell_length, 0.0615), (battery_cell_radius, 0.0171)]).evalf()))
    print('\tbattery_mass (kg): {}'.format(battery_mass.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7), (hotel_power, 5), (payload_power, 20), (time_endurance, 3306173), (battery_cell_energy_capacity, 144000), (battery_cell_mass, 0.135)]).evalf()))

    # Test solvers
    print('\nSolver Examples:')
    print('\thull_thickness (m): {}'.format(sympy.solve(hull_integrity_constraint.subs([(hull_radius_external, 0.13), (dive_depth, 2000), (hull_material_yield, 2.34422e8)]), hull_thickness)))
    print('\tpropulsion_fluid_volume (m^3): {}'.format(sympy.solve(propulsion_fluid_volume_constraint.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35)]), propulsion_fluid_volume)))
    print('\tbattery_cell_quantity: {}'.format(sympy.solve(power_supply_constraint.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (time_endurance, 3306173), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7), (hotel_power, 5), (payload_power, 20), (battery_cell_energy_capacity, 144000)]), battery_cell_quantity)))
    print('\tbattery_cell_energy_capacity (J): {}'.format(sympy.solve(power_supply_constraint.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (time_endurance, 3306173), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7), (hotel_power, 5), (payload_power, 20), (battery_cell_quantity, 825)]), battery_cell_energy_capacity)))
    print('\ttime_endurance (s): {}'.format(sympy.solve(power_supply_constraint.subs([(hull_radius_external, 0.13), (hull_length_external, 1.82), (mission_latitude, 45), (dive_depth, 2000), (water_temperature, 10), (salinity, 35), (propulsion_engine_efficiency, 0.5), (wing_area, 0.1), (wing_span, 0.8), (wing_span_efficiency, 0.7), (hotel_power, 5), (payload_power, 20), (battery_cell_energy_capacity, 144000), (battery_cell_quantity, 400)]), time_endurance)))
