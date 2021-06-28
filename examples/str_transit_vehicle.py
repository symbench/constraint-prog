#!/usr/bin/env python3
# Copyright (C) 2021, Will Hedgecock
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

from sympy import pi, cos, sin, tan, sqrt, log, exp, ceiling


# CONSTANTS AND ASSUMED VALUES ----------------------------------------------------------------------------------------

WATER_DENSITY_AT_SEA_LEVEL = 1027.0
DEPTH_RATING_SAFETY_FACTOR = 1.25
BATTERY_DERATING_SAFETY_FACTOR = 0.2
BATTERY_CELL_PACKING_FACTOR = 0.85
INTERNAL_MODULE_PACKING_FACTOR = 0.95
ADDITIONAL_BUOYANCY_FLUID_FOR_RAISED_ATTACHMENTS = 400


# DESIGN PARAMETERS ---------------------------------------------------------------------------------------------------

mission_latitude = sympy.Symbol('mission_latitude')  # decimal degrees
mission_maximum_depth = sympy.Symbol('mission_maximum_depth')  # m
mission_transit_distance = sympy.Symbol('mission_transit_distance')  # km
mission_water_salinity = sympy.Symbol('mission_water_salinity')  # PSU
mission_minimum_water_temperature = sympy.Symbol('mission_minimum_water_temperature')  # C
total_vehicle_diameter = sympy.Symbol('total_vehicle_diameter')  # m
total_vehicle_length = sympy.Symbol('total_vehicle_length')  # m
vehicle_nose_length = sympy.Symbol('vehicle_nose_length')  # m
vehicle_tail_length = sympy.Symbol('vehicle_tail_length')  # m
vehicle_depth_rating = sympy.Symbol('vehicle_depth_rating')  # m
vehicle_material_density = sympy.Symbol('vehicle_material_density')  # g/cm^3
payload_power_draw = sympy.Symbol('payload_power_draw')  # W
payload_weight_in_air = sympy.Symbol('payload_weight_in_air')  # kg
payload_displaced_volume = sympy.Symbol('payload_displaced_volume')  # m^3
payload_module_material_density = sympy.Symbol('payload_module_material_density')  # g/cm^3
payload_module_external_diameter = sympy.Symbol('payload_module_external_diameter')  # m
payload_module_external_nose_length = sympy.Symbol('payload_module_external_nose_length')  # m
payload_module_external_tail_length = sympy.Symbol('payload_external_tail_length')  # m
payload_module_external_center_length = sympy.Symbol('payload_module_external_center_length')  # m
hotel_power_draw = sympy.Symbol('hotel_power_draw')  # W
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
buoyancy_engine_module_external_diameter = sympy.Symbol('buoyancy_engine_module_external_diameter')  # m
buoyancy_engine_module_external_nose_length = sympy.Symbol('buoyancy_engine_module_external_nose_length')  # m
buoyancy_engine_module_external_tail_length = sympy.Symbol('buoyancy_engine_module_external_tail_length')  # m
buoyancy_engine_module_external_center_length = sympy.Symbol('buoyancy_engine_module_external_center_length')  # m
battery_energy_density = sympy.Symbol('battery_energy_density')  # Wh/L
battery_specific_energy_density = sympy.Symbol('battery_specific_energy_density')  # Wh/kg
battery_pack_module_material_density = sympy.Symbol('battery_pack_module_material_density')  # g/cm^3
battery_pack_module_external_nose_length = sympy.Symbol('battery_pack_module_external_nose_length')  # m
battery_pack_module_external_tail_length = sympy.Symbol('battery_pack_module_external_tail_length')  # m
electronics_module_material_density = sympy.Symbol('electronics_module_material_density')  # g/cm^3
electronics_module_external_diameter = sympy.Symbol('electronics_module_external_diameter')  # m
electronics_module_external_nose_length = sympy.Symbol('electronics_module_external_nose_length')  # m
electronics_module_external_tail_length = sympy.Symbol('electronics_module_external_tail_length')  # m
electronics_module_external_center_length = sympy.Symbol('electronics_module_external_center_length')  # m
rudder_wetted_area = sympy.Symbol('rudder_wetted_area')  # m^2
wing_span = sympy.Symbol('wing_span')  # m
wing_taper_ratio = sympy.Symbol('wing_taper_ratio')  # %
wing_volume = sympy.Symbol('wing_volume')  # m^3
wing_surface_area = sympy.Symbol('wing_surface_area')  # m^2
wing_material_thickness = sympy.Symbol('wing_material_thickness')  # m
wing_coefficient_of_lift = sympy.Symbol('wing_coefficient_of_lift')


# OCEAN AND LOCATION-BASED EXPRESSIONS --------------------------------------------------------------------------------

gravitational_acceleration = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * sin(mpmath.radians(mission_latitude))**2) * \
   sin(mpmath.radians(mission_latitude))**2)  # m/s^2
pressure_at_maximum_depth = 10000 * (2.398599584e05 - sqrt(5.753279964e10 - (4.833657881e05 * vehicle_depth_rating)))
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

vehicle_uniform_center_length = total_vehicle_length - vehicle_nose_length - vehicle_tail_length
vehicle_nose_wetted_area = 2 * pi * pow(((pow(total_vehicle_diameter/2.0, 1.6075) * pow(total_vehicle_diameter/2.0, 1.6075)) + \
                                         (pow(total_vehicle_diameter/2.0, 1.6075) * pow(vehicle_nose_length, 1.6075)) + \
                                         (pow(total_vehicle_diameter/2.0, 1.6075) * pow(vehicle_nose_length, 1.6075))) / 3.0, (1.0 / 1.6075))
vehicle_tail_wetted_area = 2 * pi * pow(((pow(total_vehicle_diameter/2.0, 1.6075) * pow(total_vehicle_diameter/2.0, 1.6075)) + \
                                         (pow(total_vehicle_diameter/2.0, 1.6075) * pow(vehicle_tail_length, 1.6075)) + \
                                         (pow(total_vehicle_diameter/2.0, 1.6075) * pow(vehicle_tail_length, 1.6075))) / 3.0, (1.0 / 1.6075))
vehicle_center_wetted_area = pi * total_vehicle_diameter * vehicle_uniform_center_length
total_vehicle_frontal_area = pi * pow((total_vehicle_diameter / 2.0), 2)
total_vehicle_wetted_area = vehicle_nose_wetted_area + vehicle_tail_wetted_area + vehicle_center_wetted_area + wing_surface_area


# MISSION SPEED EXPRESSIONS -------------------------------------------------------------------------------------------

nominal_glide_speed = nominal_horizontal_speed / cos(mpmath.radians(nominal_glide_slope))
peak_glide_speed = peak_horizontal_speed / cos(mpmath.radians(nominal_glide_slope))
nominal_vertical_speed = nominal_horizontal_speed * tan(mpmath.radians(nominal_glide_slope))
peak_vertical_speed = peak_horizontal_speed * tan(mpmath.radians(nominal_glide_slope))


# DRAG FORCE EXPRESSIONS ----------------------------------------------------------------------------------------------

reynolds_number_nominal_speed = WATER_DENSITY_AT_SEA_LEVEL * nominal_horizontal_speed * total_vehicle_diameter / seawater_absolute_viscosity
reynolds_number_maximum_speed = WATER_DENSITY_AT_SEA_LEVEL * peak_horizontal_speed * total_vehicle_diameter / seawater_absolute_viscosity
coefficient_of_skin_friction = 1.328 / sqrt(reynolds_number_maximum_speed)
nominal_form_drag_force = 0.5 * WATER_DENSITY_AT_SEA_LEVEL * nominal_glide_speed * nominal_glide_speed * total_vehicle_frontal_area * coefficient_of_drag
nominal_friction_drag_force = 0.05 * WATER_DENSITY_AT_SEA_LEVEL * nominal_glide_speed * nominal_glide_speed * total_vehicle_wetted_area * coefficient_of_skin_friction
nominal_buoyancy_required = (nominal_form_drag_force + nominal_friction_drag_force) * nominal_glide_speed / nominal_vertical_speed
maximum_form_drag_force = 0.5 * WATER_DENSITY_AT_SEA_LEVEL * peak_glide_speed * peak_glide_speed * total_vehicle_frontal_area * coefficient_of_drag
maximum_friction_drag_force = 0.05 * WATER_DENSITY_AT_SEA_LEVEL * peak_glide_speed * peak_glide_speed * total_vehicle_wetted_area * coefficient_of_skin_friction
maximum_buoyancy_required = (maximum_form_drag_force + maximum_friction_drag_force) * peak_glide_speed / peak_vertical_speed


# BUOYANCY ENGINE EXPRESSIONS -----------------------------------------------------------------------------------------

buoyancy_engine_fluid_mass = 2.0 * maximum_buoyancy_required / gravitational_acceleration
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

total_hotel_energy_required = mission_duration * hotel_power_draw
total_payload_energy_required = mission_duration * payload_power_draw
total_propulsion_energy_required = dives_per_mission * buoyancy_engine_per_dive_energy
total_mission_energy_required = total_hotel_energy_required + total_payload_energy_required + total_propulsion_energy_required
battery_capacity_required = (total_mission_energy_required / 3600.0) * (1.0 + BATTERY_DERATING_SAFETY_FACTOR)


# BATTERY EXPRESSIONS -------------------------------------------------------------------------------------------------

battery_pack_weight = battery_capacity_required / battery_specific_energy_density
unpacked_battery_pack_volume = (battery_capacity_required / battery_energy_density) * 0.001
packed_battery_pack_volume = unpacked_battery_pack_volume / BATTERY_CELL_PACKING_FACTOR
num_spherical_battery_packs_required = ceiling(packed_battery_pack_volume / ((4.0 / 3.0) * pi * pow(0.5 * total_vehicle_diameter * INTERNAL_MODULE_PACKING_FACTOR, 3.0)))
spherical_battery_pack_required_diameter = 2.0 * pow((3.0 * packed_battery_pack_volume) / (4.0 * pi * num_spherical_battery_packs_required), (1.0 / 3.0))
spherical_battery_pack_unit_weight = battery_pack_weight / num_spherical_battery_packs_required
cylindrical_battery_pack_required_diameter = total_vehicle_diameter * INTERNAL_MODULE_PACKING_FACTOR
cylindrical_battery_pack_required_length = packed_battery_pack_volume / (pow(0.5 * cylindrical_battery_pack_required_diameter, 2.0) * pi)


# WING EXPRESSIONS ----------------------------------------------------------------------------------------------------

syntactic_foam_density = sympy.Piecewise((320.0, vehicle_depth_rating <= 500.0), (350.0, vehicle_depth_rating <= 1000.0), (385.0, vehicle_depth_rating <= 2000.0),
                                         (416.0, vehicle_depth_rating <= 3000.0), (465.0, vehicle_depth_rating <= 4000.0), (495.0, vehicle_depth_rating <= 5000.0),
                                         (545.0, True))
maximum_lift_drag_ratio = 1.0 / tan(mpmath.radians(minimum_glide_slope))
minimum_wing_lift = (nominal_form_drag_force + nominal_friction_drag_force) * maximum_lift_drag_ratio
single_wing_length = 0.5 * (wing_span - total_vehicle_diameter)
wing_area_calculated = (2.0 * minimum_wing_lift) / (WATER_DENSITY_AT_SEA_LEVEL * nominal_glide_speed * nominal_glide_speed * wing_coefficient_of_lift)
wing_aspect_ratio = (wing_span * wing_span) / wing_area_calculated
wing_mean_chord = wing_area_calculated / (2.0 * single_wing_length)
wing_root_chord = wing_area_calculated / ((2.0 * single_wing_length) * (wing_taper_ratio + ((1.0 - wing_taper_ratio) / 2.0)))
wing_tip_chord = wing_taper_ratio * wing_root_chord


# INTERNAL MODULE EXPRESSIONS -----------------------------------------------------------------------------------------

battery_pack_module_external_diameter = total_vehicle_diameter * INTERNAL_MODULE_PACKING_FACTOR
battery_pack_module_external_center_length = cylindrical_battery_pack_required_length
battery_pack_module_external_length = battery_pack_module_external_center_length + battery_pack_module_external_nose_length + battery_pack_module_external_tail_length
buoyancy_engine_module_external_length = buoyancy_engine_module_external_center_length + buoyancy_engine_module_external_nose_length + buoyancy_engine_module_external_tail_length
electronics_module_external_length = electronics_module_external_center_length + electronics_module_external_nose_length + electronics_module_external_tail_length
payload_module_external_length = payload_module_external_center_length + payload_module_external_nose_length + payload_module_external_tail_length


# TEST APPLICATION ----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

   # Fill in test parameter values
   test_parameter_values = [
      (mission_latitude, 45.0),
      (mission_maximum_depth, 3000.0),
      (mission_transit_distance, 2000.0),
      (mission_water_salinity, 34),
      (mission_minimum_water_temperature, 2.0),
      (total_vehicle_diameter, 0.3556),
      (total_vehicle_length, 2.4892),
      (vehicle_nose_length, 0.3556),
      (vehicle_tail_length, 0.5110),
      (vehicle_depth_rating, 3000.0),
      (vehicle_material_density, 4.429),
      (payload_power_draw, 0.5),
      (payload_weight_in_air, 1.0),
      (payload_displaced_volume, 0.18),
      (payload_module_material_density, 2.7),
      (payload_module_external_diameter, 0.9 * 0.3556),
      (payload_module_external_center_length, 1.0),
      (payload_module_external_nose_length, 0.02),
      (payload_module_external_tail_length, 0.02),
      (hotel_power_draw, 2.0),
      (coefficient_of_drag, 0.23),
      (minimum_glide_slope, 15.0),
      (nominal_glide_slope, 35.0),
      (nominal_horizontal_speed, 0.5),
      (peak_horizontal_speed, 1.0),
      (buoyancy_engine_pump_efficiency, 0.85),
      (buoyancy_engine_motor_efficiency, 0.7),
      (buoyancy_engine_fluid_displacement, 300),
      (buoyancy_engine_rpm, 1500.0),
      (buoyancy_engine_module_material_density, 2.7),
      (buoyancy_engine_module_external_diameter, 0.338),
      (buoyancy_engine_module_external_center_length, 0.0),
      (buoyancy_engine_module_external_nose_length, 0.0),
      (buoyancy_engine_module_external_tail_length, 0.338 / 2.0),
      (battery_energy_density, 1211.0),
      (battery_specific_energy_density, 621.0),
      (battery_pack_module_material_density, 2.7),
      (battery_pack_module_external_nose_length, 0.0),
      (battery_pack_module_external_tail_length, 0.0),
      (electronics_module_material_density, 2.7),
      (electronics_module_external_diameter, 0.338),
      (electronics_module_external_center_length, 0.0),
      (electronics_module_external_nose_length, 0.338 / 2.0),
      (electronics_module_external_tail_length, 0.0),
      (wing_span, 1.5178),
      (wing_taper_ratio, 0.3),
      (wing_volume, 0.002454),
      (wing_surface_area, 0.28096),
      (wing_material_thickness, 0.001524),
      (wing_coefficient_of_lift, 0.4545)]
   
   # Print out design parameter values
   print('\nDesign Parameter Values:\n')
   for val in test_parameter_values:
      print('   {}: {}'.format(val[0], val[1]))
   
   # Print out derived parameter values
   print('\nDerived Parameter Values:\n')
   print('   Gravitational Acceleration: {} m/s^2'.format(gravitational_acceleration.subs(test_parameter_values).evalf()))
   print('   Pressure at Rated Dive Depth: {} Pa'.format(pressure_at_maximum_depth.subs(test_parameter_values).evalf()))
   print('   Water Density at Rated Dive Depth: {} kg/m^3'.format(water_density_at_maximum_depth.subs(test_parameter_values).evalf()))
   print('   Water Absolute Viscosity at Rated Dive Depth: {} Pa*s'.format(seawater_absolute_viscosity.subs(test_parameter_values).evalf()))
   print('   Water Kinematic Viscosity at Rated Dive Depth: {} m^2/s'.format(seawater_kinematic_viscosity.subs(test_parameter_values).evalf()))
   print('   Vehicle Uniform Center Length: {} m'.format(vehicle_uniform_center_length.subs(test_parameter_values).evalf()))
   print('   Vehicle Frontal Area: {} m^2'.format(total_vehicle_frontal_area.subs(test_parameter_values).evalf()))
   print('   Vehicle Nose Wetted Area: {} m^2'.format(vehicle_nose_wetted_area.subs(test_parameter_values).evalf()))
   print('   Vehicle Tail Wetted Area: {} m^2'.format(vehicle_tail_wetted_area.subs(test_parameter_values).evalf()))
   print('   Vehicle Center Wetted Area: {} m^2'.format(vehicle_center_wetted_area.subs(test_parameter_values).evalf()))
   print('   Vehicle Total Wetted Area: {} m^2'.format(total_vehicle_wetted_area.subs(test_parameter_values).evalf()))
   print('   Reynolds Number at Nominal Speed: {}'.format(reynolds_number_nominal_speed.subs(test_parameter_values).evalf()))
   print('   Reynolds Number at Maximum Speed: {}'.format(reynolds_number_maximum_speed.subs(test_parameter_values).evalf()))
   print('   Coefficient of Skin Friction: {}'.format(coefficient_of_skin_friction.subs(test_parameter_values).evalf()))
   print('   Nominal Form Drag Force: {} N'.format(nominal_form_drag_force.subs(test_parameter_values).evalf()))
   print('   Nominal Friction Drag Force: {} N'.format(nominal_friction_drag_force.subs(test_parameter_values).evalf()))
   print('   Maximum Form Drag Force: {} N'.format(maximum_form_drag_force.subs(test_parameter_values).evalf()))
   print('   Maximum Friction Drag Force: {} N'.format(maximum_friction_drag_force.subs(test_parameter_values).evalf()))
   print('   Nominal Buoyancy Force Required: {} N'.format(nominal_buoyancy_required.subs(test_parameter_values).evalf()))
   print('   Maximum Buoyancy Force Required: {} N'.format(maximum_buoyancy_required.subs(test_parameter_values).evalf()))
   print('   Buoyancy Engine Fluid Mass: {} kg'.format(buoyancy_engine_fluid_mass.subs(test_parameter_values).evalf()))
   print('   Buoyancy Engine Fluid Volume: {} m^3'.format(buoyancy_engine_fluid_volume.subs(test_parameter_values).evalf()))
   print('   Buoyancy Engine Reservoir Volume: {} cc'.format(buoyancy_engine_reservoir_volume.subs(test_parameter_values).evalf()))
   print('   Buoyancy Engine Flow Rate: {} m^3/s'.format(buoyancy_engine_flow_rate.subs(test_parameter_values).evalf()))
   print('   Buoyancy Engine Per-Dive Pump Time: {} s'.format(buoyancy_engine_per_dive_pump_time.subs(test_parameter_values).evalf()))
   print('   Buoyancy Engine Per-Dive Energy: {} J'.format(buoyancy_engine_per_dive_energy.subs(test_parameter_values).evalf()))
   print('   Buoyancy Engine Module External Length: {} m'.format(buoyancy_engine_module_external_length.subs(test_parameter_values).evalf()))
   print('   Nominal Glide Speed: {} m/s'.format(nominal_glide_speed.subs(test_parameter_values).evalf()))
   print('   Nominal Vertical Speed: {} m/s'.format(nominal_vertical_speed.subs(test_parameter_values).evalf()))
   print('   Peak Glide Speed: {} m/s'.format(peak_glide_speed.subs(test_parameter_values).evalf()))
   print('   Peak Vertical Speed: {} m/s'.format(peak_vertical_speed.subs(test_parameter_values).evalf()))
   print('   Mission Per-Dive Duration: {} s'.format(dive_duration.subs(test_parameter_values).evalf()))
   print('   Horizontal Distance Per Dive: {} m'.format(dive_horizontal_distance.subs(test_parameter_values).evalf()))
   print('   Required Total Number of Mission Dives: {}'.format(dives_per_mission.subs(test_parameter_values).evalf()))
   print('   Total Mission Duration: {} s'.format(mission_duration.subs(test_parameter_values).evalf()))
   print('   Total Hotel Energy Required: {} J'.format(total_hotel_energy_required.subs(test_parameter_values).evalf()))
   print('   Total Payload Energy Required: {} J'.format(total_payload_energy_required.subs(test_parameter_values).evalf()))
   print('   Total Propulsion Energy Required: {} J'.format(total_propulsion_energy_required.subs(test_parameter_values).evalf()))
   print('   Total Mission Energy Required: {} J'.format(total_mission_energy_required.subs(test_parameter_values).evalf()))
   print('   Required Battery Capacity: {} Wh'.format(battery_capacity_required.subs(test_parameter_values).evalf()))
   print('   Battery Pack Weight: {} kg'.format(battery_pack_weight.subs(test_parameter_values).evalf()))
   print('   Unpacked Battery Pack Volume: {} m^3'.format(unpacked_battery_pack_volume.subs(test_parameter_values).evalf()))
   print('   Packed Battery Pack Volume: {} m^3'.format(packed_battery_pack_volume.subs(test_parameter_values).evalf()))
   print('   Num Required Spherical Battery Packs: {}'.format(num_spherical_battery_packs_required.subs(test_parameter_values).evalf()))
   print('   Spherical Battery Pack Diameter Required: {} m'.format(spherical_battery_pack_required_diameter.subs(test_parameter_values).evalf()))
   print('   Spherical Battery Pack Unit Weight: {} kg'.format(spherical_battery_pack_unit_weight.subs(test_parameter_values).evalf()))
   print('   Cylindrical Battery Pack Diameter Required: {} m'.format(cylindrical_battery_pack_required_diameter.subs(test_parameter_values).evalf()))
   print('   Cylindrical Battery Pack Length Required: {} m'.format(cylindrical_battery_pack_required_length.subs(test_parameter_values).evalf()))
   print('   Battery Pack Module External Diameter: {} m'.format(battery_pack_module_external_diameter.subs(test_parameter_values).evalf()))
   print('   Battery Pack Module External Center Length: {} m'.format(battery_pack_module_external_center_length.subs(test_parameter_values).evalf()))
   print('   Battery Pack Module External Total Length: {} m'.format(battery_pack_module_external_length.subs(test_parameter_values).evalf()))
   print('   Electronics Module External Length: {} m'.format(electronics_module_external_length.subs(test_parameter_values).evalf()))
   print('   Payload Module External Length: {} m'.format(payload_module_external_length.subs(test_parameter_values).evalf()))
   print('   Syntactic Foam Density: {} kg/m^3'.format(syntactic_foam_density.subs(test_parameter_values).evalf()))
   print('   Single Wing Length: {} m'.format(single_wing_length.subs(test_parameter_values).evalf()))
   print('   Maximum L/D Ratio: {}'.format(maximum_lift_drag_ratio.subs(test_parameter_values).evalf()))
   print('   Minimum Wing Lift: {} N'.format(minimum_wing_lift.subs(test_parameter_values).evalf()))
   print('   Wing Surface Area: {} m^2'.format(wing_surface_area.subs(test_parameter_values).evalf()))
   print('   Wing Aspect Ratio: {}'.format(wing_aspect_ratio.subs(test_parameter_values).evalf()))
   print('   Wing Mean Chord: {} m'.format(wing_mean_chord.subs(test_parameter_values).evalf()))
   print('   Wing Root Chord: {} m'.format(wing_root_chord.subs(test_parameter_values).evalf()))
   print('   Wing Tip Chord: {} m'.format(wing_tip_chord.subs(test_parameter_values).evalf()))
   print()
