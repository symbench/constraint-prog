#!/usr/bin/env python3
# Copyright (C) 2021, Miklos Maroti and Will Hedgecock
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

import math
import sympy, mpmath
from sympy import cos, sin, tan, sqrt, log, exp, ceiling, atan
from typing import Tuple
from constraint_prog.point_cloud import PointCloud, PointFunc

# PARAMETER SPECIFICATIONS FROM MISSION AND SPREADSHEET ---------------------------------------------------------------

allowable_pitch_error_at_neutral = 1.5  # degrees
allowable_roll_error_at_neutral = 0.5  # degrees
movable_pitch_diameter = 0.02  # m
movable_roll_height = 0.10  # m
glider_depth_rating = 3000  # m
vehicle_hull_thickness = 0.02  # m
child_vehicle_dry_mass = 65.06  # kg
child_vehicle_length = 1.22  # m
child_vehicle_diameter = 0.46  # m
child_vehicle_cg = child_vehicle_length - 0.482  # m from bow
child_vehicle_cb = child_vehicle_length - 0.619  # m from bow
num_child_vehicles = 1
antenna_dry_mass = 0.5  # kg
antenna_length = 1.0  # m

concrete_parameters = {
   'mission_latitude': 80.0,
   'mission_maximum_depth': 3000.0,
   'mission_transit_distance': 1500.0,
   'mission_surveying_duration': 604800,
   'mission_water_salinity': 34,
   'mission_minimum_water_temperature': 0.0,
   'vehicle_fairing_thickness': 0.0047752,
   'vehicle_fairing_material_density': 1522.4,
   'vehicle_nose_length': 0.3556,
   'vehicle_tail_length': 0.5110,
   'vehicle_depth_rating': 3000.0,
   'payload_power_draw': 0.0,
   'hotel_power_draw_in_transit': 1.0,
   'hotel_power_draw_surveying': 2.0,
   'coefficient_of_drag': 0.23,
   'minimum_glide_slope': 15.0,
   'nominal_glide_slope': 35.0,
   'nominal_horizontal_speed': 0.75,
   'peak_horizontal_speed': 1.5,
   'buoyancy_engine_pump_efficiency': 0.85,
   'buoyancy_engine_motor_efficiency': 0.7,
   'buoyancy_engine_fluid_displacement': 300,
   'buoyancy_engine_rpm': 1500.0,
   'surveying_buoyancy_percent_of_weight': 0.01,
   'wing_span': 1.5178,
   'wing_taper_ratio': 0.3,
   'wing_wetted_area': 0.28096,
   'wing_material_thickness': 0.001524,
   'wing_coefficient_of_lift': 0.4545
}


# CONSTANTS AND ASSUMED VALUES ----------------------------------------------------------------------------------------

GRAVITATIONAL_CONSTANT = 9.806  # m/s^2
WATER_DENSITY_AT_SEA_LEVEL = 1027.0  # kg/m^3
WATER_DENSITY_AT_DIVE_DEPTH = 1041.02  # kg/m^3
DEPTH_RATING_SAFETY_FACTOR = 1.25
INTERNAL_MODULE_PACKING_FACTOR = 0.95
ADDITIONAL_BUOYANCY_FLUID_FOR_RAISED_ATTACHMENTS = 400

OIL_DENSITY = 837.0  # kg/m^3
FOAM_DENSITY = 406.0  # kg/m^3
LEAD_DENSITY = 11343.0  # kg/m^3

ALUMINIUM_DENSITY = 2700.0  # kg/m^3
ALUMINIUM_YOUNG_MODULUS = 6.89e10  # Pa
ALUMINIUM_YIELD_STRESS = 2.76e8  # Pa
ALUMINIUM_POISSON_RATIO = 0.33

BATTERY_DERATING_SAFETY_FACTOR = 0.2
BATTERY_CELL_PACKING_FACTOR = 0.85
BATTERY_CAPACITY_PER_MASS = 621  # Wh / kg
BATTERY_CAPACITY_PER_VOLUME = 1211e3  # Wh / m^3


# DESIGN PARAMETERS ---------------------------------------------------------------------------------------------------

mission_latitude = sympy.Symbol('mission_latitude')  # decimal degrees
mission_maximum_depth = sympy.Symbol('mission_maximum_depth')  # m
mission_transit_distance = sympy.Symbol('mission_transit_distance')  # km
mission_surveying_duration = sympy.Symbol('mission_surveying_duration')  # s
mission_water_salinity = sympy.Symbol('mission_water_salinity')  # PSU
mission_minimum_water_temperature = sympy.Symbol('mission_minimum_water_temperature')  # C
vehicle_diameter_external = sympy.Symbol('vehicle_diameter_external')  # m
vehicle_length_external = sympy.Symbol('vehicle_length_external')  # m
vehicle_fairing_thickness = sympy.Symbol('vehicle_fairing_thickness')  # m
vehicle_fairing_material_density = sympy.Symbol('vehicle_fairing_material_density')  # kg/m^3
vehicle_nose_length = sympy.Symbol('vehicle_nose_length')  # m
vehicle_tail_length = sympy.Symbol('vehicle_tail_length')  # m
vehicle_depth_rating = sympy.Symbol('vehicle_depth_rating')  # m
payload_power_draw = sympy.Symbol('payload_power_draw')  # W
payload_weight_in_air = sympy.Symbol('payload_weight_in_air')  # kg
payload_displaced_volume = sympy.Symbol('payload_displaced_volume')  # m^3
payload_module_material_density = sympy.Symbol('payload_module_material_density')  # g/cm^3
payload_module_material_youngs_modulus = sympy.Symbol('payload_module_material_youngs_modulus')  # Pa
payload_module_material_yield_stress = sympy.Symbol('payload_module_material_yield_stress')  # Pa
payload_module_material_poissons_ratio = sympy.Symbol('payload_module_material_poissons_ratio')
payload_module_external_diameter = sympy.Symbol('payload_module_external_diameter')  # m
payload_module_external_nose_length = sympy.Symbol('payload_module_external_nose_length')  # m
payload_module_external_tail_length = sympy.Symbol('payload_module_external_tail_length')  # m
payload_module_external_center_length = sympy.Symbol('payload_module_external_center_length')  # m
surveying_buoyancy_percent_of_weight = sympy.Symbol('surveying_buoyancy_percent_of_weight')  # %
hotel_power_draw_in_transit = sympy.Symbol('hotel_power_draw_in_transit')  # W
hotel_power_draw_surveying = sympy.Symbol('hotel_power_draw_surveying')  # W
coefficient_of_drag = sympy.Symbol('coefficient_of_drag')  # unitless
minimum_glide_slope = sympy.Symbol('minimum_glide_slope')  # degrees
nominal_glide_slope = sympy.Symbol('nominal_glide_slope')  # degrees
nominal_horizontal_speed = sympy.Symbol('nominal_horizontal_speed')  # m/s
peak_horizontal_speed = sympy.Symbol('peak_horizontal_speed')  # m/s
buoyancy_engine_pump_efficiency = sympy.Symbol('buoyancy_engine_pump_efficiency')  # percent
buoyancy_engine_motor_efficiency = sympy.Symbol('buoyancy_engine_motor_efficiency')  # percent
buoyancy_engine_fluid_displacement = sympy.Symbol('buoyancy_engine_fluid_displacement')  # mm^3/rev
buoyancy_engine_rpm = sympy.Symbol('buoyancy_engine_rpm')  # rev/min
buoyancy_engine_module_material_density = sympy.Symbol('buoyancy_engine_module_material_density')  # g/cm^3
buoyancy_engine_module_material_youngs_modulus = sympy.Symbol('buoyancy_engine_module_material_youngs_modulus')  # Pa
buoyancy_engine_module_material_yield_stress = sympy.Symbol('buoyancy_engine_module_material_yield_stress')  # Pa
buoyancy_engine_module_material_poissons_ratio = sympy.Symbol('buoyancy_engine_module_material_poissons_ratio')
buoyancy_engine_module_external_diameter = sympy.Symbol('buoyancy_engine_module_external_diameter')  # m
buoyancy_engine_module_external_nose_length = sympy.Symbol('buoyancy_engine_module_external_nose_length')  # m
buoyancy_engine_module_external_tail_length = sympy.Symbol('buoyancy_engine_module_external_tail_length')  # m
buoyancy_engine_module_external_center_length = sympy.Symbol('buoyancy_engine_module_external_center_length')  # m
battery_energy_density = sympy.Symbol('battery_energy_density')  # Wh/L
battery_specific_energy_density = sympy.Symbol('battery_specific_energy_density')  # Wh/kg
battery_pack_module_material_density = sympy.Symbol('battery_pack_module_material_density')  # g/cm^3
battery_pack_module_material_youngs_modulus = sympy.Symbol('battery_pack_module_material_youngs_modulus')  # Pa
battery_pack_module_material_yield_stress = sympy.Symbol('battery_pack_module_material_yield_stress')  # Pa
battery_pack_module_material_poissons_ratio = sympy.Symbol('battery_pack_module_material_poissons_ratio')
battery_pack_module_external_nose_length = sympy.Symbol('battery_pack_module_external_nose_length')  # m
battery_pack_module_external_tail_length = sympy.Symbol('battery_pack_module_external_tail_length')  # m
electronics_module_material_density = sympy.Symbol('electronics_module_material_density')  # g/cm^3
electronics_module_material_youngs_modulus = sympy.Symbol('electronics_module_material_youngs_modulus')  # Pa
electronics_module_material_yield_stress = sympy.Symbol('electronics_module_material_yield_stress')  # Pa
electronics_module_material_poissons_ratio = sympy.Symbol('electronics_module_material_poissons_ratio')
electronics_module_external_diameter = sympy.Symbol('electronics_module_external_diameter')  # m
electronics_module_external_nose_length = sympy.Symbol('electronics_module_external_nose_length')  # m
electronics_module_external_tail_length = sympy.Symbol('electronics_module_external_tail_length')  # m
electronics_module_external_center_length = sympy.Symbol('electronics_module_external_center_length')  # m
rudder_wetted_area = sympy.Symbol('rudder_wetted_area')  # m^2
wing_span = sympy.Symbol('wing_span')  # m
wing_taper_ratio = sympy.Symbol('wing_taper_ratio')  # %
wing_wetted_area = sympy.Symbol('wing_wetted_area')  # m^2
wing_material_thickness = sympy.Symbol('wing_material_thickness')  # m
wing_coefficient_of_lift = sympy.Symbol('wing_coefficient_of_lift')


# OCEAN AND LOCATION-BASED EXPRESSIONS --------------------------------------------------------------------------------

gravitational_acceleration = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * sin(mpmath.radians(mission_latitude))**2) * \
   sin(mpmath.radians(mission_latitude))**2)  # m/s^2
pressure_at_maximum_depth = 10000 * (2.398599584e05 - sqrt(5.753279964e10 - (4.833657881e05 * vehicle_depth_rating)))
crush_pressure_at_maximum_depth = pressure_at_maximum_depth * DEPTH_RATING_SAFETY_FACTOR
water_density_at_maximum_depth = 1000 + ((((((((((((5.2787e-8 * mission_minimum_water_temperature) - 6.12293e-6) * \
   mission_minimum_water_temperature) + 8.50935e-5) + ((((9.1697e-10 * mission_minimum_water_temperature) + \
   2.0816e-8) * mission_minimum_water_temperature) - 9.9348e-7) * mission_water_salinity) * \
   pressure_at_maximum_depth * 1e-5) + ((((1.91075e-4 * sqrt(mission_water_salinity)) + ((((-1.6078e-6 * \
   mission_minimum_water_temperature) - 1.0981e-5) * mission_minimum_water_temperature) + 2.2838e-3)) * \
   mission_water_salinity) + ((((((-5.77905e-7 * mission_minimum_water_temperature) + 1.16092e-4) * \
   mission_minimum_water_temperature) + 1.43713e-3) * mission_minimum_water_temperature) + 3.239908))) * \
   pressure_at_maximum_depth * 1e-5) + ((((((((-5.3009e-4 * mission_minimum_water_temperature) + 1.6483e-2) * \
   mission_minimum_water_temperature) + 7.944e-2) * sqrt(mission_water_salinity)) + ((((((-6.1670e-5 * \
   mission_minimum_water_temperature) + 1.09987e-2) * mission_minimum_water_temperature) - 0.603459) * \
   mission_minimum_water_temperature) + 54.6746)) * mission_water_salinity) + ((((((((-5.155288e-5 * \
   mission_minimum_water_temperature) + 1.360477e-2) * mission_minimum_water_temperature) - 2.327105) * \
   mission_minimum_water_temperature) + 148.4206) * mission_minimum_water_temperature) + 19652.21))) * \
   ((((4.8314e-4 * mission_water_salinity) + (((((-1.6546e-6 * mission_minimum_water_temperature) + 1.0227e-4) * \
   mission_minimum_water_temperature) - 5.72466e-3) * sqrt(mission_water_salinity)) + ((((((((5.3875e-9 * \
   mission_minimum_water_temperature) - 8.2467e-7) * mission_minimum_water_temperature) + 7.6438e-5) * \
   mission_minimum_water_temperature) - 4.0899e-3) * mission_minimum_water_temperature) + 0.824493)) * \
   mission_water_salinity) + (((((((((6.536332e-9 * mission_minimum_water_temperature) - 1.120083e-6) * \
   mission_minimum_water_temperature) + 1.001685e-4) * mission_minimum_water_temperature) - 9.095290e-3) * \
   mission_minimum_water_temperature) + 6.793952e-2) * mission_minimum_water_temperature - 0.157406))) + \
   (pressure_at_maximum_depth * 1e-2)) / ((((((((((5.2787e-8 * mission_minimum_water_temperature) - 6.12293e-6) * \
   mission_minimum_water_temperature) + 8.50935e-5) + ((((9.1697e-10 * mission_minimum_water_temperature) + \
   2.0816e-8) * mission_minimum_water_temperature) - 9.9348e-7) * mission_water_salinity) * \
   pressure_at_maximum_depth * 1e-5) + ((((1.91075e-4 * sqrt(mission_water_salinity)) + ((((-1.6078e-6 * \
   mission_minimum_water_temperature) - 1.0981e-5) * mission_minimum_water_temperature) + 2.2838e-3)) * \
   mission_water_salinity) + ((((((-5.77905e-7 * mission_minimum_water_temperature) + 1.16092e-4) * \
   mission_minimum_water_temperature) + 1.43713e-3) * mission_minimum_water_temperature) + 3.239908))) * \
   pressure_at_maximum_depth * 1e-5) + ((((((((-5.3009e-4 * mission_minimum_water_temperature) + 1.6483e-2) * \
   mission_minimum_water_temperature) + 7.944e-2) * sqrt(mission_water_salinity)) + ((((((-6.1670e-5 * \
   mission_minimum_water_temperature) + 1.09987e-2) * mission_minimum_water_temperature) - 0.603459) * \
   mission_minimum_water_temperature) + 54.6746)) * mission_water_salinity) + ((((((((-5.155288e-5 * \
   mission_minimum_water_temperature) + 1.360477e-2) * mission_minimum_water_temperature) - 2.327105) * \
   mission_minimum_water_temperature) + 148.4206) * mission_minimum_water_temperature) + 19652.21))) - \
   (pressure_at_maximum_depth * 1e-5)))  # kg/m^3
freshwater_absolute_viscosity = ((0.00000000000277388442 * mission_minimum_water_temperature**6) - \
    (0.00000000124359703683 * mission_minimum_water_temperature**5) + (0.00000022981389243372 * mission_minimum_water_temperature**4) - \
    (0.00002310372106867350 * mission_minimum_water_temperature**3) + (0.00143393546700877000 * mission_minimum_water_temperature**2) - \
    (0.06064140920049450000 * mission_minimum_water_temperature) + 1.79157254681817000000) / 1000  # Pa*s = kg/(m*s)
seawater_absolute_viscosity = freshwater_absolute_viscosity * \
    (1.0 + ((1.541 + (1.998e-2 * mission_minimum_water_temperature) - (9.52e-5 * mission_minimum_water_temperature**2)) * \
    mission_water_salinity * 0.001) + ((7.974 - (7.561e-2 * mission_minimum_water_temperature) + (4.724e-4 * mission_minimum_water_temperature**2)) * \
    (mission_water_salinity * 0.001)**2))  # Pa*s = kg/(m*s)
seawater_kinematic_viscosity = seawater_absolute_viscosity / water_density_at_maximum_depth  # m^2/s


# VEHICLE GEOMETRY EXPRESSIONS ----------------------------------------------------------------------------------------

vehicle_inner_diameter = vehicle_diameter_external - (2.0 * vehicle_fairing_thickness)
vehicle_uniform_center_length = vehicle_length_external - vehicle_nose_length - vehicle_tail_length
vehicle_nose_wetted_area = 2 * math.pi * pow(((pow(vehicle_diameter_external/2.0, 1.6075) * pow(vehicle_diameter_external/2.0, 1.6075)) + \
                                              (pow(vehicle_diameter_external/2.0, 1.6075) * pow(vehicle_nose_length, 1.6075)) + \
                                              (pow(vehicle_diameter_external/2.0, 1.6075) * pow(vehicle_nose_length, 1.6075))) / 3.0, (1.0 / 1.6075))
vehicle_tail_wetted_area = 2 * math.pi * pow(((pow(vehicle_diameter_external/2.0, 1.6075) * pow(vehicle_diameter_external/2.0, 1.6075)) + \
                                              (pow(vehicle_diameter_external/2.0, 1.6075) * pow(vehicle_tail_length, 1.6075)) + \
                                              (pow(vehicle_diameter_external/2.0, 1.6075) * pow(vehicle_tail_length, 1.6075))) / 3.0, (1.0 / 1.6075))
vehicle_center_wetted_area = math.pi * vehicle_diameter_external * vehicle_uniform_center_length
total_vehicle_frontal_area = math.pi * pow((vehicle_diameter_external / 2.0), 2)
total_vehicle_wetted_area = vehicle_nose_wetted_area + vehicle_tail_wetted_area + vehicle_center_wetted_area + wing_wetted_area
vehicle_fairing_dry_mass = vehicle_fairing_material_density * (total_vehicle_wetted_area - wing_wetted_area) * vehicle_fairing_thickness
vehicle_fairing_displacement = WATER_DENSITY_AT_SEA_LEVEL * (total_vehicle_wetted_area - wing_wetted_area) * vehicle_fairing_thickness


# MISSION SPEED EXPRESSIONS -------------------------------------------------------------------------------------------

nominal_glide_speed = nominal_horizontal_speed / cos(mpmath.radians(nominal_glide_slope))
peak_glide_speed = peak_horizontal_speed / cos(mpmath.radians(nominal_glide_slope))
nominal_vertical_speed = nominal_horizontal_speed * tan(mpmath.radians(nominal_glide_slope))
peak_vertical_speed = peak_horizontal_speed * tan(mpmath.radians(nominal_glide_slope))


# DRAG FORCE EXPRESSIONS ----------------------------------------------------------------------------------------------

reynolds_number_nominal_speed = WATER_DENSITY_AT_SEA_LEVEL * nominal_horizontal_speed * vehicle_diameter_external / seawater_absolute_viscosity
reynolds_number_maximum_speed = WATER_DENSITY_AT_SEA_LEVEL * peak_horizontal_speed * vehicle_diameter_external / seawater_absolute_viscosity
coefficient_of_skin_friction = 1.328 / sqrt(reynolds_number_maximum_speed)
nominal_form_drag_force = 0.5 * WATER_DENSITY_AT_SEA_LEVEL * nominal_glide_speed * nominal_glide_speed * total_vehicle_frontal_area * coefficient_of_drag
nominal_friction_drag_force = 0.05 * WATER_DENSITY_AT_SEA_LEVEL * nominal_glide_speed * nominal_glide_speed * total_vehicle_wetted_area * coefficient_of_skin_friction
nominal_buoyancy_required = (nominal_form_drag_force + nominal_friction_drag_force) * nominal_glide_speed / nominal_vertical_speed
maximum_form_drag_force = 0.5 * WATER_DENSITY_AT_SEA_LEVEL * peak_glide_speed * peak_glide_speed * total_vehicle_frontal_area * coefficient_of_drag
maximum_friction_drag_force = 0.05 * WATER_DENSITY_AT_SEA_LEVEL * peak_glide_speed * peak_glide_speed * total_vehicle_wetted_area * coefficient_of_skin_friction
required_maximum_buoyancy = (maximum_form_drag_force + maximum_friction_drag_force) * peak_glide_speed / peak_vertical_speed


# BUOYANCY ENGINE EXPRESSIONS -----------------------------------------------------------------------------------------

buoyancy_engine_fluid_mass = 2.0 * required_maximum_buoyancy / gravitational_acceleration
buoyancy_engine_fluid_volume = buoyancy_engine_fluid_mass / WATER_DENSITY_AT_SEA_LEVEL
buoyancy_engine_reservoir_volume = (1e6 * buoyancy_engine_fluid_volume) + ADDITIONAL_BUOYANCY_FLUID_FOR_RAISED_ATTACHMENTS
buoyancy_engine_flow_rate = 1e-9 * buoyancy_engine_fluid_displacement * buoyancy_engine_rpm / 60.0
buoyancy_engine_per_dive_pump_time = buoyancy_engine_fluid_volume / buoyancy_engine_flow_rate
buoyancy_engine_per_dive_energy = buoyancy_engine_per_dive_pump_time * \
   ((buoyancy_engine_flow_rate * gravitational_acceleration * mission_maximum_depth * water_density_at_maximum_depth) /\
    (buoyancy_engine_pump_efficiency * buoyancy_engine_motor_efficiency))


# MISSION DURATION EXPRESSIONS ----------------------------------------------------------------------------------------

dive_duration = buoyancy_engine_per_dive_pump_time + (2 * mission_maximum_depth / nominal_vertical_speed)
dive_horizontal_distance = nominal_horizontal_speed * (2 * mission_maximum_depth / nominal_vertical_speed)
dives_per_mission = ceiling(1000.0 * mission_transit_distance / dive_horizontal_distance)
mission_duration = dives_per_mission * dive_duration


# ENERGY REQUIREMENT EXPRESSIONS --------------------------------------------------------------------------------------

total_hotel_energy_required = (mission_duration * hotel_power_draw_in_transit) + \
   (mission_surveying_duration * hotel_power_draw_surveying)
total_payload_energy_required = mission_duration * payload_power_draw
total_propulsion_energy_required = dives_per_mission * buoyancy_engine_per_dive_energy
total_mission_energy_required = total_hotel_energy_required + total_payload_energy_required + total_propulsion_energy_required
battery_capacity_required = (total_mission_energy_required / 3600.0) * (1.0 + BATTERY_DERATING_SAFETY_FACTOR)


# BATTERY EXPRESSIONS -------------------------------------------------------------------------------------------------

battery_pack_weight = battery_capacity_required / battery_specific_energy_density
unpacked_battery_pack_volume = (battery_capacity_required / battery_energy_density) * 0.001
packed_battery_pack_volume = unpacked_battery_pack_volume / BATTERY_CELL_PACKING_FACTOR
num_spherical_battery_packs_required = ceiling(packed_battery_pack_volume / ((4.0 / 3.0) * math.pi * pow(0.5 * vehicle_inner_diameter * INTERNAL_MODULE_PACKING_FACTOR, 3.0)))
spherical_battery_pack_required_diameter = 2.0 * pow((3.0 * packed_battery_pack_volume) / (4.0 * math.pi * num_spherical_battery_packs_required), (1.0 / 3.0))
spherical_battery_pack_unit_weight = battery_pack_weight / num_spherical_battery_packs_required
cylindrical_battery_pack_required_diameter = vehicle_inner_diameter * INTERNAL_MODULE_PACKING_FACTOR
cylindrical_battery_pack_required_length = packed_battery_pack_volume / (pow(0.5 * cylindrical_battery_pack_required_diameter, 2.0) * math.pi)


# WING EXPRESSIONS ----------------------------------------------------------------------------------------------------

syntactic_foam_density = sympy.Piecewise((320.0, vehicle_depth_rating <= 500.0), (350.0, vehicle_depth_rating <= 1000.0), (385.0, vehicle_depth_rating <= 2000.0),
                                         (416.0, vehicle_depth_rating <= 3000.0), (465.0, vehicle_depth_rating <= 4000.0), (495.0, vehicle_depth_rating <= 5000.0),
                                         (545.0, True))
maximum_lift_drag_ratio = 1.0 / tan(mpmath.radians(minimum_glide_slope))
minimum_wing_lift = (nominal_form_drag_force + nominal_friction_drag_force) * maximum_lift_drag_ratio
single_wing_length = 0.5 * (wing_span - vehicle_diameter_external)
wing_surface_area = (2.0 * minimum_wing_lift) / (WATER_DENSITY_AT_SEA_LEVEL * nominal_glide_speed * nominal_glide_speed * wing_coefficient_of_lift)
wing_aspect_ratio = (wing_span * wing_span) / wing_surface_area
wing_mean_chord = wing_surface_area / (2.0 * single_wing_length)
wing_root_chord = wing_surface_area / ((2.0 * single_wing_length) * (wing_taper_ratio + ((1.0 - wing_taper_ratio) / 2.0)))
wing_tip_chord = wing_taper_ratio * wing_root_chord
wing_thickness = 0.06 * wing_root_chord
wing_volume = wing_mean_chord * wing_span * wing_thickness #wing_surface_area * wing_thickness
wing_dry_mass = (wing_surface_area * wing_material_thickness * vehicle_fairing_material_density) + \
   ((wing_volume - (wing_surface_area * wing_material_thickness)) * syntactic_foam_density)
wing_displacement = wing_volume * WATER_DENSITY_AT_SEA_LEVEL


# INTERNAL MODULE EXPRESSIONS -----------------------------------------------------------------------------------------

battery_pack_module_external_diameter = vehicle_inner_diameter * INTERNAL_MODULE_PACKING_FACTOR
battery_pack_module_external_center_length = cylindrical_battery_pack_required_length
battery_pack_module_external_length = battery_pack_module_external_center_length + battery_pack_module_external_nose_length + battery_pack_module_external_tail_length
battery_pack_module_material_thickness_cylinder = sympy.Max(pow((0.5 * crush_pressure_at_maximum_depth / battery_pack_module_material_youngs_modulus) * \
   (1.0 - battery_pack_module_material_poissons_ratio**2), 1.0 / 3.0) * battery_pack_module_external_diameter,
   (0.5 * (1.0 - sqrt(1.0 - (2.0 * crush_pressure_at_maximum_depth / battery_pack_module_material_yield_stress))) * battery_pack_module_external_diameter))
battery_pack_module_material_thickness_sphere = sympy.Max(sqrt(crush_pressure_at_maximum_depth * (0.5 * battery_pack_module_external_diameter)**2 / \
   (0.365 * battery_pack_module_material_youngs_modulus)), (crush_pressure_at_maximum_depth * 0.5 * battery_pack_module_external_diameter) / \
   (2.0 * battery_pack_module_material_yield_stress))
battery_pack_module_material_thickness = sympy.Piecewise((battery_pack_module_material_thickness_sphere, battery_pack_module_external_center_length <= 0.0001),
   (sympy.Max(battery_pack_module_material_thickness_cylinder, battery_pack_module_material_thickness_sphere), True))
battery_pack_module_internal_diameter = battery_pack_module_external_diameter - (2.0 * battery_pack_module_material_thickness)
buoyancy_engine_module_external_length = buoyancy_engine_module_external_center_length + buoyancy_engine_module_external_nose_length + buoyancy_engine_module_external_tail_length
buoyancy_engine_module_material_thickness_cylinder = sympy.Max(pow((0.5 * crush_pressure_at_maximum_depth / buoyancy_engine_module_material_youngs_modulus) * \
   (1.0 - buoyancy_engine_module_material_poissons_ratio**2), 1.0 / 3.0) * buoyancy_engine_module_external_diameter,
   (0.5 * (1.0 - sqrt(1.0 - (2.0 * crush_pressure_at_maximum_depth / buoyancy_engine_module_material_yield_stress))) * buoyancy_engine_module_external_diameter))
buoyancy_engine_module_material_thickness_sphere = sympy.Max(sqrt(crush_pressure_at_maximum_depth * (0.5 * buoyancy_engine_module_external_diameter)**2 / \
   (0.365 * buoyancy_engine_module_material_youngs_modulus)), (crush_pressure_at_maximum_depth * 0.5 * buoyancy_engine_module_external_diameter) / \
   (2.0 * buoyancy_engine_module_material_yield_stress))
buoyancy_engine_module_material_thickness = sympy.Piecewise((buoyancy_engine_module_material_thickness_sphere, buoyancy_engine_module_external_center_length <= 0.0001),
   (sympy.Max(buoyancy_engine_module_material_thickness_cylinder, buoyancy_engine_module_material_thickness_sphere), True))
buoyancy_engine_module_internal_diameter = buoyancy_engine_module_external_diameter - (2.0 * buoyancy_engine_module_material_thickness)
electronics_module_external_length = electronics_module_external_center_length + electronics_module_external_nose_length + electronics_module_external_tail_length
electronics_module_material_thickness_cylinder = sympy.Max(pow((0.5 * crush_pressure_at_maximum_depth / electronics_module_material_youngs_modulus) * \
   (1.0 - electronics_module_material_poissons_ratio**2), 1.0 / 3.0) * electronics_module_external_diameter,
   (0.5 * (1.0 - sqrt(1.0 - (2.0 * crush_pressure_at_maximum_depth / electronics_module_material_yield_stress))) * electronics_module_external_diameter))
electronics_module_material_thickness_sphere = sympy.Max(sqrt(crush_pressure_at_maximum_depth * (0.5 * electronics_module_external_diameter)**2 / \
   (0.365 * electronics_module_material_youngs_modulus)), (crush_pressure_at_maximum_depth * 0.5 * electronics_module_external_diameter) / \
   (2.0 * electronics_module_material_yield_stress))
electronics_module_material_thickness = sympy.Piecewise((electronics_module_material_thickness_sphere, electronics_module_external_center_length <= 0.0001),
   (sympy.Max(electronics_module_material_thickness_cylinder, electronics_module_material_thickness_sphere), True))
electronics_module_internal_diameter = electronics_module_external_diameter - (2.0 * electronics_module_material_thickness)
payload_module_external_length = payload_module_external_center_length + payload_module_external_nose_length + payload_module_external_tail_length
payload_module_material_thickness_cylinder = sympy.Max(pow((0.5 * crush_pressure_at_maximum_depth / payload_module_material_youngs_modulus) * \
   (1.0 - payload_module_material_poissons_ratio**2), 1.0 / 3.0) * payload_module_external_diameter,
   (0.5 * (1.0 - sqrt(1.0 - (2.0 * crush_pressure_at_maximum_depth / payload_module_material_yield_stress))) * payload_module_external_diameter))
payload_module_material_thickness_sphere = sympy.Max(sqrt(crush_pressure_at_maximum_depth * (0.5 * payload_module_external_diameter)**2 / \
   (0.365 * payload_module_material_youngs_modulus)), (crush_pressure_at_maximum_depth * 0.5 * payload_module_external_diameter) / \
   (2.0 * payload_module_material_yield_stress))
payload_module_material_thickness = sympy.Piecewise((payload_module_material_thickness_sphere, payload_module_external_center_length <= 0.0001),
   (sympy.Max(payload_module_material_thickness_cylinder, payload_module_material_thickness_sphere), True))
payload_module_internal_diameter = payload_module_external_diameter - (2.0 * payload_module_material_thickness)


# VEHICLE COMPONENT ORDERING ------------------------------------------------------------------------------------------
# CENTER LINE (z = 0) IS ALONG THE CENTER OF THE PRESSURE VESSELS AND FOAM

pressure_vessel_outer_diameter = vehicle_inner_diameter.subs(concrete_parameters).evalf() - movable_pitch_diameter  # m
glider_crush_pressure = WATER_DENSITY_AT_DIVE_DEPTH * glider_depth_rating * DEPTH_RATING_SAFETY_FACTOR * GRAVITATIONAL_CONSTANT
aluminum_buckling_failure_sphere = sqrt(glider_crush_pressure * (0.5 * pressure_vessel_outer_diameter)**2 / (0.365 * ALUMINIUM_YOUNG_MODULUS))
aluminum_stress_failure_sphere = (glider_crush_pressure * 0.5 * pressure_vessel_outer_diameter) / (2.0 * ALUMINIUM_YIELD_STRESS)
pressure_vessel_thickness_sphere = sympy.Max(aluminum_buckling_failure_sphere, aluminum_stress_failure_sphere)
pressure_vessel_inner_diameter = pressure_vessel_outer_diameter - 2 * pressure_vessel_thickness_sphere
pressure_vessel_inner_volume = math.pi / 6 * (pressure_vessel_inner_diameter ** 3)  # m^3

foam1_length = sympy.Symbol("foam1_length")
foam1_x_left = 0
foam1_x_right = foam1_x_left + foam1_length

vessel1_x_left = foam1_x_right
vessel1_x_right = vessel1_x_left + pressure_vessel_outer_diameter

vessel2_x_left = vessel1_x_right
vessel2_x_right = vessel2_x_left + pressure_vessel_outer_diameter

reservoir1_oil_volume = pressure_vessel_inner_volume
bladder1_length_full = reservoir1_oil_volume / (math.pi / 4 * pressure_vessel_outer_diameter ** 2)
bladder1_x_left = vessel2_x_right
bladder1_x_right = bladder1_x_left + bladder1_length_full

foam2_length = sympy.Symbol("foam2_length")
foam2_x_left = bladder1_x_right
foam2_x_right = foam2_x_left + foam2_length

wing_x_left = foam2_x_right
wing_x_right = wing_x_left + wing_root_chord.subs(concrete_parameters).evalf()

vessel3_x_left = wing_x_right
vessel3_x_right = vessel3_x_left + pressure_vessel_outer_diameter

foam3_length = sympy.Symbol("foam3_length")
foam3_x_left = vessel3_x_right
foam3_x_right = foam3_x_left + foam3_length

vessel4_x_left = foam3_x_right
vessel4_x_right = vessel4_x_left + pressure_vessel_outer_diameter
vehicle_inner_length = vessel4_x_right
vehicle_length_external_equation = sympy.Eq(vehicle_length_external.subs(concrete_parameters),
                                            (2 * vehicle_fairing_thickness.subs(concrete_parameters)) + vehicle_inner_length)


# FORWARD CALCULATION RESULTS -----------------------------------------------------------------------------------------

antenna_displacement = antenna_dry_mass / ALUMINIUM_DENSITY * WATER_DENSITY_AT_SEA_LEVEL
pressure_vessel_outer_volume = math.pi / 6 * (pressure_vessel_outer_diameter ** 3)
pressure_vessel_dry_mass = (pressure_vessel_outer_volume - pressure_vessel_inner_volume) * ALUMINIUM_DENSITY
pressure_vessel_displacement = pressure_vessel_outer_volume * WATER_DENSITY_AT_SEA_LEVEL
vehicle_fairing_dry_mass = vehicle_fairing_dry_mass.subs(concrete_parameters).evalf()
vehicle_fairing_displacement = vehicle_fairing_displacement.subs(concrete_parameters).evalf()
vehicle_inner_diameter = vehicle_inner_diameter.subs(concrete_parameters).evalf()
required_maximum_buoyancy = required_maximum_buoyancy.subs(concrete_parameters).evalf()
battery_capacity_required = battery_capacity_required.subs(concrete_parameters).evalf()
surveying_buoyancy_percent_of_weight = surveying_buoyancy_percent_of_weight.subs(concrete_parameters).evalf()
wing_root_chord = wing_root_chord.subs(concrete_parameters).evalf()
wing_dry_mass = wing_dry_mass.subs(concrete_parameters).evalf()
wing_thickness = wing_thickness.subs(concrete_parameters).evalf()
wing_displacement = wing_displacement.subs(concrete_parameters).evalf()
print('\nForward calculation constants:\n')
print('   vehicle_fairing_dry_mass =', vehicle_fairing_dry_mass)
print('   vehicle_fairing_displacement =', vehicle_fairing_displacement)
print('   vehicle_inner_diameter =', vehicle_inner_diameter)
print('   required_maximum_buoyancy =', required_maximum_buoyancy)
print('   battery_capacity_required =', battery_capacity_required)
print('   wing_root_chord =', wing_root_chord)
print('   wing_dry_mass =', wing_dry_mass)
print('   wing_thickness =', wing_thickness)
print('   wing_displacement =', wing_displacement)
print("   pressure_vessel_outer_diameter:", pressure_vessel_outer_diameter)
print("   pressure_vessel_inner_diameter:", pressure_vessel_inner_diameter)
print("   pressure_vessel_crush_pressure: ", glider_crush_pressure)
print("   pressure_vessel_thickness:", pressure_vessel_thickness_sphere)
print("   pressure_vessel_inner_volume:", pressure_vessel_inner_volume)
print("   pressure_vessel_dry_mass:", pressure_vessel_dry_mass)
print("   pressure_vessel_displacement:", pressure_vessel_displacement)


# MASS, VOLUME, AND CENTER CALCULATIONS -------------------------------------------------------------------------------

foam1_volume = foam1_length * math.pi / 4 * (vehicle_inner_diameter ** 2 - movable_pitch_diameter ** 2)  # m^3
foam1_dry_mass = foam1_volume * FOAM_DENSITY  # kg
foam1_displacement = foam1_volume * WATER_DENSITY_AT_SEA_LEVEL  # kg
foam1_x_center = (foam1_x_left + foam1_x_right) / 2  # m
foam1_z_center = -movable_pitch_diameter / 2

vessel1_dry_mass = pressure_vessel_dry_mass
vessel1_displacement = pressure_vessel_displacement
vessel1_x_center = (vessel1_x_left + vessel1_x_right) / 2

battery1_capacity = sympy.Symbol("battery1_capacity")
battery1_dry_mass = battery1_capacity / BATTERY_CAPACITY_PER_MASS
battery1_x_center = vessel1_x_center
battery1_packing_volume = battery1_capacity / BATTERY_CAPACITY_PER_VOLUME / BATTERY_CELL_PACKING_FACTOR

vessel2_dry_mass = pressure_vessel_dry_mass
vessel2_displacement = pressure_vessel_displacement
vessel2_x_center = (vessel2_x_left + vessel2_x_right) / 2

reservoir1_dry_mass_full = reservoir1_oil_volume * OIL_DENSITY
reservoir1_dry_mass_half = reservoir1_dry_mass_full / 2
reservoir1_dry_mass_empty = 0
reservoir1_x_center = vessel2_x_center

bladder1_dry_mass_full = reservoir1_oil_volume * OIL_DENSITY
bladder1_dry_mass_half = bladder1_dry_mass_full / 2
bladder1_dry_mass_empty = 0
bladder1_displacement_full = reservoir1_oil_volume * WATER_DENSITY_AT_SEA_LEVEL
bladder1_displacement_half = bladder1_displacement_full / 2
bladder1_displacement_empty = 0
bladder1_x_center = (bladder1_x_left + bladder1_x_right) / 2

foam2_volume = foam2_length * math.pi / 4 * (vehicle_inner_diameter ** 2 - movable_pitch_diameter ** 2)  # m^3
foam2_dry_mass = foam2_volume * FOAM_DENSITY  # kg
foam2_displacement = foam2_volume * WATER_DENSITY_AT_SEA_LEVEL  # kg
foam2_x_center = (foam2_x_left + foam2_x_right) / 2  # m
foam2_z_center = -movable_pitch_diameter / 2

wing_x_center = (wing_x_left + wing_x_right) / 2
wing_z_center = -movable_pitch_diameter / 2 + movable_roll_height / 2 + wing_thickness / 2

movable_roll_length = wing_root_chord
movable_roll_width = sympy.Symbol("movable_roll_width")
movable_roll_volume = movable_roll_length * movable_roll_width * movable_roll_height
movable_roll_dry_mass = movable_roll_volume * LEAD_DENSITY
movable_roll_displacement = movable_roll_volume * WATER_DENSITY_AT_SEA_LEVEL
movable_roll_x_center = wing_x_center
movable_roll_y_center_stb = vehicle_inner_diameter / 2 - movable_roll_width / 2 - 0.05  # TODO: look out for magic
movable_roll_y_center_mid = 0
movable_roll_y_center_prt = -movable_roll_y_center_stb
movable_roll_z_center = -movable_pitch_diameter / 2

vessel3_dry_mass = pressure_vessel_dry_mass
vessel3_displacement = pressure_vessel_displacement
vessel3_x_center = (vessel3_x_left + vessel3_x_right) / 2

electronics_dry_mass = 5.5 + 6.0  # kg (includes pump and electronics)
electronics_x_center = vessel3_x_center

foam3_volume = foam3_length * math.pi / 4 * (vehicle_inner_diameter ** 2 - movable_pitch_diameter ** 2)  # m^3
foam3_dry_mass = foam3_volume * FOAM_DENSITY  # kg
foam3_displacement = foam3_volume * WATER_DENSITY_AT_SEA_LEVEL  # kg
foam3_x_center = (foam3_x_left + foam3_x_right) / 2  # m
foam3_z_center = -movable_pitch_diameter / 2

vessel4_dry_mass = pressure_vessel_dry_mass
vessel4_displacement = pressure_vessel_displacement
vessel4_x_center = (vessel4_x_left + vessel4_x_right) / 2

battery2_capacity = sympy.Symbol("battery2_capacity")
battery2_dry_mass = battery2_capacity / BATTERY_CAPACITY_PER_MASS
battery2_x_center = vessel4_x_center
battery2_packing_volume = battery2_capacity / BATTERY_CAPACITY_PER_VOLUME / BATTERY_CELL_PACKING_FACTOR

vehicle_fairing_x_center = vehicle_inner_length / 2
vehicle_fairing_z_center = -movable_pitch_diameter / 2

# movable pitch is completeley below the other modules
movable_pitch_length = sympy.Symbol("movable_pitch_length")
movable_pitch_volume = movable_pitch_length * math.pi / 4 * movable_pitch_diameter ** 2
movable_pitch_dry_mass = movable_pitch_volume * LEAD_DENSITY
movable_pitch_displacement = movable_pitch_volume * WATER_DENSITY_AT_SEA_LEVEL
movable_pitch_x_center_fwd = movable_pitch_length / 2
movable_pitch_x_center_aft = vehicle_inner_length - movable_pitch_length / 2
movable_pitch_x_center_mid = vehicle_inner_length / 2
movable_pitch_z_center = -(pressure_vessel_outer_diameter + movable_pitch_diameter) / 2

# antenna is above the hull
antenna_x_center = vehicle_inner_length * sympy.Symbol("antenna_x_relpos")
antenna_z_center = pressure_vessel_outer_diameter / 2 + vehicle_hull_thickness + antenna_length / 2

# child vehicle is below the hull
child_vehicle_displacement = child_vehicle_dry_mass
child_vehicle_x_center = vehicle_inner_length * sympy.Symbol("child_vehicle_x_relpos")
child_vehicle_x_left = child_vehicle_x_center - child_vehicle_length / 2
child_vehicle_z_center = -pressure_vessel_outer_diameter / 2 - movable_pitch_diameter - vehicle_hull_thickness - child_vehicle_diameter / 2


def get_center_of_gravity(bladder: str, pitch: str, roll: str, antenna: str, children: int) \
        -> Tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]:
    """
    Returns the total dry weight and the x, y and z coordinates of
    the center of gravity.
    """
    total_mass = 0
    total_x_sum = 0
    total_y_sum = 0
    total_z_sum = 0

    total_mass += foam1_dry_mass
    total_x_sum += foam1_dry_mass * foam1_x_center

    total_mass += vessel1_dry_mass
    total_x_sum += vessel1_dry_mass * vessel1_x_center

    total_mass += battery1_dry_mass
    total_x_sum += battery1_dry_mass * battery1_x_center

    total_mass += vessel2_dry_mass
    total_x_sum += vessel2_dry_mass * vessel2_x_center

    if bladder == "empty":
        total_mass += reservoir1_dry_mass_full
        total_x_sum += reservoir1_dry_mass_full * reservoir1_x_center
    elif bladder == "half":
        total_mass += reservoir1_dry_mass_half
        total_x_sum += reservoir1_dry_mass_half * reservoir1_x_center
    else:
        assert bladder == "full"
        total_mass += reservoir1_dry_mass_empty
        total_x_sum += reservoir1_dry_mass_empty * reservoir1_x_center

    if bladder == "empty":
        total_mass += bladder1_dry_mass_empty
        total_x_sum += bladder1_dry_mass_empty * bladder1_x_center
    elif bladder == "half":
        total_mass += bladder1_dry_mass_half
        total_x_sum += bladder1_dry_mass_half * bladder1_x_center
    elif bladder == "full":
        total_mass += bladder1_dry_mass_full
        total_x_sum += bladder1_dry_mass_full * bladder1_x_center

    total_mass += foam2_dry_mass
    total_x_sum += foam2_dry_mass * foam2_x_center

    total_mass += wing_dry_mass
    total_x_sum += wing_dry_mass * wing_x_center
    total_z_sum += wing_dry_mass * wing_z_center

    total_mass += movable_roll_dry_mass
    total_x_sum += movable_roll_dry_mass * movable_roll_x_center
    total_z_sum += movable_roll_dry_mass * movable_roll_z_center
    if roll == "starboard":
        total_y_sum += movable_roll_dry_mass * movable_roll_y_center_stb
    elif roll == "center":
        total_y_sum += movable_roll_dry_mass * movable_roll_y_center_mid
    else:
        assert roll == "port"
        total_y_sum += movable_roll_dry_mass * movable_roll_y_center_prt

    total_mass += vessel3_dry_mass
    total_x_sum += vessel3_dry_mass * vessel3_x_center

    total_mass += electronics_dry_mass
    total_x_sum += electronics_dry_mass * electronics_x_center

    total_mass += foam3_dry_mass
    total_x_sum += foam3_dry_mass * foam3_x_center

    total_mass += vessel4_dry_mass
    total_x_sum += vessel4_dry_mass * vessel4_x_center

    total_mass += battery2_dry_mass
    total_x_sum += battery2_dry_mass * battery2_x_center

    total_mass += vehicle_fairing_dry_mass
    total_x_sum += vehicle_fairing_dry_mass * vehicle_fairing_x_center
    total_z_sum += vehicle_fairing_dry_mass * vehicle_fairing_z_center

    total_mass += movable_pitch_dry_mass
    total_z_sum += movable_pitch_dry_mass * movable_pitch_z_center
    if pitch == "forward":
        total_x_sum += movable_pitch_dry_mass * movable_pitch_x_center_fwd
    elif pitch == "middle":
        total_x_sum += movable_pitch_dry_mass * movable_pitch_x_center_mid
    else:
        assert pitch == "aft"
        total_x_sum += movable_pitch_dry_mass * movable_pitch_x_center_aft

    if antenna == "on":
        total_mass += antenna_dry_mass
        total_x_sum += antenna_dry_mass * antenna_x_center
        total_z_sum += antenna_dry_mass * antenna_z_center
    else:
        assert antenna == "off"

    if children == 1:
        total_mass += child_vehicle_dry_mass
        total_x_sum += child_vehicle_dry_mass * (child_vehicle_x_left + child_vehicle_cg)
        total_z_sum += child_vehicle_dry_mass * child_vehicle_z_center
    else:
        assert children == 0

    return (total_mass, total_x_sum / total_mass, total_y_sum / total_mass, total_z_sum / total_mass)


def get_center_of_buoyancy(bladder: str, pitch: str, roll: str, antenna: str, children: int) \
        -> Tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]:
    """
    Returns the total weight of displaced water and the x, y and z coordinates
    of the center of gravity.
    """
    total_mass = 0
    total_x_sum = 0
    total_y_sum = 0
    total_z_sum = 0

    total_mass += foam1_displacement
    total_x_sum += foam1_displacement * foam1_x_center

    total_mass += vessel1_displacement
    total_x_sum += vessel1_displacement * vessel1_x_center

    total_mass += vessel2_displacement
    total_x_sum += vessel2_displacement * vessel2_x_center

    if bladder == "empty":
        total_mass += bladder1_displacement_empty
        total_x_sum += bladder1_displacement_empty * bladder1_x_center
    elif bladder == "half":
        total_mass += bladder1_displacement_half
        total_x_sum += bladder1_displacement_half * bladder1_x_center
    elif bladder == "full":
        total_mass += bladder1_displacement_full
        total_x_sum += bladder1_displacement_full * bladder1_x_center

    total_mass += foam2_displacement
    total_x_sum += foam2_displacement * foam2_x_center

    total_mass += wing_displacement
    total_x_sum += wing_displacement * wing_x_center
    total_z_sum += wing_displacement * wing_z_center

    total_mass += movable_roll_displacement
    total_x_sum += movable_roll_displacement * movable_roll_x_center
    total_z_sum += movable_roll_displacement * movable_roll_z_center
    if roll == "starboard":
        total_y_sum += movable_roll_displacement * movable_roll_y_center_stb
    elif roll == "center":
        total_y_sum += movable_roll_displacement * movable_roll_y_center_mid
    else:
        assert roll == "port"
        total_y_sum += movable_roll_displacement * movable_roll_y_center_prt

    total_mass += vessel3_displacement
    total_x_sum += vessel3_displacement * vessel3_x_center

    total_mass += foam3_displacement
    total_x_sum += foam3_displacement * foam3_x_center

    total_mass += vessel4_displacement
    total_x_sum += vessel4_displacement * vessel4_x_center

    total_mass += vehicle_fairing_displacement
    total_x_sum += vehicle_fairing_displacement * vehicle_fairing_x_center
    total_z_sum += vehicle_fairing_displacement * vehicle_fairing_z_center

    total_mass += movable_pitch_displacement
    total_z_sum += movable_pitch_displacement * movable_pitch_z_center
    if pitch == "forward":
        total_x_sum += movable_pitch_displacement * movable_pitch_x_center_fwd
    elif pitch == "middle":
        total_x_sum += movable_pitch_displacement * movable_pitch_x_center_mid
    else:
        assert pitch == "aft"
        total_x_sum += movable_pitch_displacement * movable_pitch_x_center_aft

    if antenna == "on":
        total_mass += antenna_displacement
        total_x_sum += antenna_displacement * antenna_x_center
        total_z_sum += antenna_displacement * antenna_z_center
    else:
        assert antenna == "off"

    if children == 1:
        total_mass += child_vehicle_displacement
        total_x_sum += child_vehicle_displacement * (child_vehicle_x_left + child_vehicle_cb)
        total_z_sum += child_vehicle_displacement * child_vehicle_z_center
    else:
        assert children == 0

    return (total_mass, total_x_sum / total_mass, total_y_sum / total_mass, total_z_sum / total_mass)


def get_buoyancy_minus_gravity(bladder: str, pitch: str, roll: str, antenna: str, children: int) \
        -> Tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]:
    cb_mass, cb_x, cb_y, cb_z = get_center_of_buoyancy(bladder, pitch, roll, antenna, children)
    cg_mass, cg_x, cg_y, cg_z = get_center_of_gravity(bladder, pitch, roll, antenna, children)
    return cb_mass - cg_mass, cb_x - cg_x, cb_y - cg_y, cb_z - cg_z


vehicle_dry_mass_stage1, _, _, _ = get_center_of_gravity(bladder="half", pitch="middle",
                                                         roll="center", antenna="on", children=1)
vehicle_dry_mass_stage2, _, _, _ = get_center_of_gravity(bladder="half", pitch="middle",
                                                         roll="center", antenna="off", children=0)

battery1_packing_equation = battery1_packing_volume <= pressure_vessel_inner_volume
battery2_packing_equation = battery2_packing_volume <= pressure_vessel_inner_volume
battery_capacity_equation = battery1_capacity + battery2_capacity >= battery_capacity_required

pitch_minimum_cbmg_buoyancy_stage1, pitch_minimum_cbmg_x_stage1, pitch_minimum_cbmg_y_stage1, pitch_minimum_cbmg_z_stage1 = \
   get_buoyancy_minus_gravity(bladder="empty", pitch="forward", roll="center", antenna="on", children=1)
pitch_minimum_equation1_stage1 = -pitch_minimum_cbmg_x_stage1 / pitch_minimum_cbmg_z_stage1 <= math.tan(-60 * math.pi / 180)
pitch_minimum_equation2_stage1 = pitch_minimum_cbmg_buoyancy_stage1 <= -required_maximum_buoyancy / GRAVITATIONAL_CONSTANT  # sinks to bottom

pitch_minimum_cbmg_buoyancy_stage2, pitch_minimum_cbmg_x_stage2, pitch_minimum_cbmg_y_stage2, pitch_minimum_cbmg_z_stage2 = \
   get_buoyancy_minus_gravity(bladder="empty", pitch="forward", roll="center", antenna="off", children=0)
pitch_minimum_equation1_stage2 = -pitch_minimum_cbmg_x_stage2 / pitch_minimum_cbmg_z_stage2 <= math.tan(-60 * math.pi / 180)
pitch_minimum_equation2_stage2 = pitch_minimum_cbmg_buoyancy_stage2 <= -required_maximum_buoyancy / GRAVITATIONAL_CONSTANT  # sinks to bottom

pitch_maximum_cbmg_buoyancy_stage1, pitch_maximum_cbmg_x_stage1, pitch_maximum_cbmg_y_stage1, pitch_maximum_cbmg_z_stage1 = \
   get_buoyancy_minus_gravity(bladder="full", pitch="aft", roll="center", antenna="on", children=1)
pitch_maximum_equation1_stage1 = -pitch_maximum_cbmg_x_stage1 / pitch_maximum_cbmg_z_stage1 >= math.tan(60 * math.pi / 180)

pitch_maximum_cbmg_buoyancy_stage2, pitch_maximum_cbmg_x_stage2, pitch_maximum_cbmg_y_stage2, pitch_maximum_cbmg_z_stage2 = \
   get_buoyancy_minus_gravity(bladder="full", pitch="aft", roll="center", antenna="off", children=0)
pitch_maximum_equation1_stage2 = -pitch_maximum_cbmg_x_stage2 / pitch_maximum_cbmg_z_stage2 >= math.tan(60 * math.pi / 180)

pitch_neutral_cbmg_buoyancy_stage1, pitch_neutral_cbmg_x_stage1, pitch_neutral_cbmg_y_stage1, pitch_neutral_cbmg_z_stage1 = \
   get_buoyancy_minus_gravity(bladder="half", pitch="middle", roll="center", antenna="on", children=1)
pitch_neutral_equation1_stage1 = -pitch_neutral_cbmg_x_stage1 / pitch_neutral_cbmg_z_stage1 <= math.tan(allowable_pitch_error_at_neutral * math.pi / 180)
pitch_neutral_equation2_stage1 = -pitch_neutral_cbmg_x_stage1 / pitch_neutral_cbmg_z_stage1 >= math.tan(-allowable_pitch_error_at_neutral * math.pi / 180)
pitch_neutral_equation3_stage1 = sympy.Abs(pitch_neutral_cbmg_buoyancy_stage1 - 0.25) <= 0.5  # TODO: more magic

pitch_neutral_cbmg_buoyancy_stage2, pitch_neutral_cbmg_x_stage2, pitch_neutral_cbmg_y_stage2, pitch_neutral_cbmg_z_stage2 = \
   get_buoyancy_minus_gravity(bladder="half", pitch="middle", roll="center", antenna="off", children=0)
pitch_neutral_equation1_stage2 = -pitch_neutral_cbmg_x_stage2 / pitch_neutral_cbmg_z_stage2 <= math.tan(allowable_pitch_error_at_neutral * math.pi / 180)
pitch_neutral_equation2_stage2 = -pitch_neutral_cbmg_x_stage2 / pitch_neutral_cbmg_z_stage2 >= math.tan(-allowable_pitch_error_at_neutral * math.pi / 180)
pitch_neutral_equation3_stage2 = sympy.Abs(pitch_neutral_cbmg_buoyancy_stage2 - 0.25) <= 0.5  # TODO: more magic

roll_minimum_cbmg_buoyancy_stage1, roll_minimum_cbmg_x_stage1, roll_minimum_cbmg_y_stage1, roll_minimum_cbmg_z_stage1 = \
   get_buoyancy_minus_gravity(bladder="half", pitch="middle", roll="port", antenna="on", children=1)
roll_minimum_equation1_stage1 = -roll_minimum_cbmg_y_stage1 / roll_minimum_cbmg_z_stage1 <= math.tan(-20 * math.pi / 180)

roll_minimum_cbmg_buoyancy_stage2, roll_minimum_cbmg_x_stage2, roll_minimum_cbmg_y_stage2, roll_minimum_cbmg_z_stage2 = \
   get_buoyancy_minus_gravity(bladder="half", pitch="middle", roll="port", antenna="off", children=0)
roll_minimum_equation1_stage2 = -roll_minimum_cbmg_y_stage2 / roll_minimum_cbmg_z_stage2 <= math.tan(-20 * math.pi / 180)

constraints = PointFunc({
   "vehicle_length_constraint": vehicle_length_external_equation,
   "battery1_packing_equation": battery1_packing_equation,
   "battery2_packing_equation": battery2_packing_equation,
   "battery_capacity_equation": battery_capacity_equation,
   #"pitch_minimum_equation1_stage1": pitch_minimum_equation1_stage1,
   #"pitch_minimum_equation2_stage1": pitch_minimum_equation2_stage1,
   #"pitch_minimum_equation1_stage2": pitch_minimum_equation1_stage2,
   #"pitch_minimum_equation2_stage2": pitch_minimum_equation2_stage2,
   #"pitch_maximum_equation1_stage1": pitch_maximum_equation1_stage1,
   #"pitch_maximum_equation1_stage2": pitch_maximum_equation1_stage2,
   "pitch_neutral_equation1_stage1": pitch_neutral_equation1_stage1,
   "pitch_neutral_equation2_stage1": pitch_neutral_equation2_stage1,
   "pitch_neutral_equation3_stage1": pitch_neutral_equation3_stage1,
   "pitch_neutral_equation1_stage2": pitch_neutral_equation1_stage2,
   "pitch_neutral_equation2_stage2": pitch_neutral_equation2_stage2,
   #"pitch_neutral_equation3_stage2": pitch_neutral_equation3_stage2,
   "roll_minimum_equation1_stage1": roll_minimum_equation1_stage1,
   "roll_minimum_equation1_stage2": roll_minimum_equation1_stage2,
   "surveying_buoyancy_force": vehicle_dry_mass_stage2 * surveying_buoyancy_percent_of_weight * GRAVITATIONAL_CONSTANT <= required_maximum_buoyancy,
})

print("\nConstraint variables:", constraints.input_names)
print("\nConstraint equations:", list(constraints.output_names))

derived_values = PointFunc({
   "pitch_minimum_buoyancy": pitch_minimum_cbmg_buoyancy_stage1,
   "pitch_minimum_angle": atan(-pitch_minimum_cbmg_x_stage1 / pitch_minimum_cbmg_z_stage1) * 180 / math.pi,
   "pitch_neutral_buoyancy": pitch_neutral_cbmg_buoyancy_stage1,
   "pitch_neutral_angle": atan(-pitch_neutral_cbmg_x_stage1 / pitch_neutral_cbmg_z_stage1) * 180 / math.pi,
   "pitch_maximum_buoyancy": pitch_maximum_cbmg_buoyancy_stage1,
   "pitch_maximum_angle": atan(-pitch_maximum_cbmg_x_stage1 / pitch_maximum_cbmg_z_stage1) * 180 / math.pi,
   "roll_minimum_angle": atan(-roll_minimum_cbmg_y_stage1 / roll_minimum_cbmg_z_stage1) * 180 / math.pi,
   "foam1_x_center": foam1_x_center,
   "foam1_z_center": foam1_z_center,
   "foam1_volume": foam1_volume,
   "foam1_dry_mass": foam1_dry_mass,
   "vessel1_x_center": vessel1_x_center,
   "battery1_x_center": battery1_x_center,
   "battery1_packing_volume": battery1_packing_volume,
   "battery1_dry_mass": battery1_dry_mass,
   "vessel2_x_center": vessel2_x_center,
   "reservoir1_x_center": reservoir1_x_center,
   "reservoir1_dry_mass_full": reservoir1_dry_mass_full,
   "bladder1_length_full": bladder1_length_full,
   "bladder1_x_center": bladder1_x_center,
   "wing_x_center": wing_x_center,
   "wing_z_center": wing_z_center,
   "wing_dry_mass": wing_dry_mass,
   "movable_roll_x_center": movable_roll_x_center,
   "movable_roll_y_center_prt": movable_roll_y_center_prt,
   "movable_roll_y_center_mid": movable_roll_y_center_mid,
   "movable_roll_y_center_stb": movable_roll_y_center_stb,
   "movable_roll_z_center": movable_roll_z_center,
   "movable_roll_dry_mass": movable_roll_dry_mass,
   "foam2_x_center": foam2_x_center,
   "foam2_z_center": foam2_z_center,
   "foam2_volume": foam2_volume,
   "foam2_dry_mass": foam2_dry_mass,
   "vessel3_x_center": vessel3_x_center,
   "electronics_x_center": electronics_x_center,
   "electronics_dry_mass": electronics_dry_mass,
   "vessel4_x_center": vessel4_x_center,
   "battery2_x_center": battery2_x_center,
   "battery2_packing_volume": battery2_packing_volume,
   "battery2_dry_mass": battery2_dry_mass,
   "foam3_x_center": foam3_x_center,
   "foam3_z_center": foam3_z_center,
   "foam3_volume": foam3_volume,
   "foam3_dry_mass": foam3_dry_mass,
   "movable_pitch_x_center_fwd": movable_pitch_x_center_fwd,
   "movable_pitch_x_center_mid": movable_pitch_x_center_mid,
   "movable_pitch_x_center_aft": movable_pitch_x_center_aft,
   "movable_pitch_z_center": movable_pitch_z_center,
   "movable_pitch_dry_mass": movable_pitch_dry_mass,
   "vehicle_fairing_x_center": vehicle_fairing_x_center,
   "vehicle_fairing_z_center": vehicle_fairing_z_center,
   "total_battery_capacity": battery1_capacity + battery2_capacity,
   "vehicle_dry_mass_stage1": vehicle_dry_mass_stage1,
   "vehicle_dry_mass_stage2": vehicle_dry_mass_stage2,
   "vehicle_inner_length": vehicle_inner_length,
   'vehicle_inner_volume': math.pi * (vehicle_inner_diameter/2)**2 * vehicle_inner_length,
   "vehicle_fineness_ratio": vehicle_inner_length / vehicle_inner_diameter,
   "antenna_x_center": antenna_x_center,
   "antenna_z_center": antenna_z_center,
   "child_vehicle_x_center": child_vehicle_x_center,
   "child_vehicle_z_center": child_vehicle_z_center,
   "surveying_buoyancy_percent_of_weight": surveying_buoyancy_percent_of_weight
})

def print_solutions(points, num=None):
   points = points.extend(derived_values(points, equs_as_float=False))
   if num is None:
      num = points.num_points
   else:
      num = min(num, points.num_points)
   for sol in range(num):
      print("\nSolution #:", sol)
      for idx, var in enumerate(points.float_vars):
         print(var + ":", points.float_data[sol, idx].item())
   print()

bounds = {
   "antenna_x_relpos": (0.25, 0.75),
   "battery1_capacity": (0.0, 80000.0),
   "battery2_capacity": (0.0, 80000.0),
   "child_vehicle_x_relpos": (0.25, 0.75),
   "foam1_length": (0.0, 2.0),
   "foam2_length": (0.0, 2.0),
   "foam3_length": (0.0, 2.0),
   "movable_pitch_length": (0.10, 1.0),
   "movable_roll_width": (0.05, 0.5),
   "vehicle_diameter_external": (0.2, 1.0),
   "vehicle_length_external": (1.0, 10.0),
}

resolutions = {
   "antenna_x_relpos": 0.1,
   "battery1_capacity": 1000.0,
   "battery2_capacity": 1000.0,
   "child_vehicle_x_relpos": 0.1,
   "foam1_length": 0.1,
   "foam2_length": 0.1,
   "foam3_length": 0.1,
   "movable_pitch_length": 0.1,
   "movable_roll_width": 0.1,
   "vehicle_diameter_external": 0.1,
   "vehicle_length_external": 0.1,
}

print("\nConstraint variable bounds:", bounds)
print("\nConstraint variable resolutions:", resolutions)
print()

assert list(bounds.keys()) == list(constraints.input_names)

# generate random points
points = PointCloud.generate(bounds, 10000)

# minimize errors with newton raphson
points = points.newton_raphson(constraints, bounds)
# points.print_info()

# calculate remaining errors
errors = constraints(points)
# errors.print_info()

# check constraints with loose tolerance
points = points.prune_by_tolerances(errors, 0.5)
print("Design points found:", points.num_points)

target_func = PointFunc({
    "total_battery_capacity": battery1_capacity + battery2_capacity,
    "vehicle_dry_mass_stage1": vehicle_dry_mass_stage1,
    "vehicle_dry_mass_stage2": vehicle_dry_mass_stage2,
    "vehicle_inner_length": vehicle_inner_length,
    "wing_x_center": wing_x_center,
    "pitch_minimum_angle": atan(-pitch_minimum_cbmg_x_stage1 / pitch_minimum_cbmg_z_stage1) * 180 / math.pi,
    "roll_minimum_angle": atan(-roll_minimum_cbmg_y_stage1 / roll_minimum_cbmg_z_stage1) * 180 / math.pi,
    "battery1_capacity": battery1_capacity,
    "foam1_length": foam1_length,
})

for step in range(5):
    if not points.num_points:
        print("No design points left!")
        break
    points.add_mutations(resolutions, 10000, multiplier=2.0)
    points = points.newton_raphson(constraints, bounds)
    points = points.prune_by_tolerances(constraints(points), 1.0 if step <= 2 else 0.1)
    points = points.prune_close_points2(resolutions)
    # points2 = points.extend(target_func(points))
    print("Design points found:", points.num_points)

if points.num_points:
    points2 = target_func(points)
    points2.plot2d("total_battery_capacity", "vehicle_dry_mass_stage1")
    # points2.plot2d("total_battery_capacity", "vehicle_inner_length")
    points2.plot2d("vehicle_inner_length", "wing_x_center")
    points2.plot2d("pitch_minimum_angle", "roll_minimum_angle")
    points2.plot2d("battery1_capacity", "foam1_length")
    print_solutions(points, 10)
    points2 = points.extend(derived_values(points, equs_as_float=False))
    points2.save("str_transit_packing.csv")
