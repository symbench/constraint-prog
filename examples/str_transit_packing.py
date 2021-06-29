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

from re import fullmatch
import sympy
from sympy import pi, cos, sin, tan, sqrt, log, exp, ceiling

WATER_DENSITY_AT_SEA_LEVEL = 1027.0  # kg/m^3
OIL_DENSITY = 837.0  # kg/m^3
FOAM_DENSITY = 406.0  # kg/m^3
ALUMINIUM_DENSITY = 2700.0  # kg/m^3
LEAD_DENSITY = 11343.0  # kg/m^3

BATTERY_CAPACITY_PER_MASS = 621  # Wh / kg
BATTERY_CAPACITY_PER_VOLUME = 1211e3  # Wh / m^3
BATTERY_DERATING_FACTOR = 0.2
BATTERY_PACKING_FACTOR = 0.85

vehicle_fairing_wet_mass = 8.822  # kg
vehicle_inner_diameter = 0.35   # m
movable_pitch_diameter = 0.05   # m
movable_roll_length = 0.10  # m

pressure_vessel_outer_diameter = 0.30  # m
pressure_vessel_inner_diameter = 0.25  # m
pressure_vessel_outer_volume = pi / 6 \
    * (pressure_vessel_outer_diameter ** 3)  # m^3
pressure_vessel_inner_volume = pi / 6 \
    * (pressure_vessel_inner_diameter ** 3)  # m^3
pressure_vessel_dry_mass = (pressure_vessel_outer_volume - pressure_vessel_inner_volume) \
    * ALUMINIUM_DENSITY  # kg
pressure_vessel_wet_mass = pressure_vessel_dry_mass - \
    pressure_vessel_outer_volume * WATER_DENSITY_AT_SEA_LEVEL

allowable_pitch_error_at_neutral = 1.5  # degrees
allowable_roll_error_at_neutral = 0.5  # degrees

# ---------------

foam1_length = sympy.Symbol("foam1_length")
foam1_displacement = foam1_length * pi / 4 * \
    (pressure_vessel_outer_diameter ** 2)  # m^3
foam1_dry_mass = foam1_displacement * FOAM_DENSITY  # kg
foam1_wet_mass = foam1_displacement * \
    (FOAM_DENSITY - WATER_DENSITY_AT_SEA_LEVEL)  # kg
foam1_x_left = 0  # m
foam1_x_right = foam1_x_left + foam1_length  # m
foam1_x_center = (foam1_x_left + foam1_x_right) / 2  # m

vessel1_displacement = pressure_vessel_outer_volume
vessel1_dry_mass = pressure_vessel_dry_mass
vessel1_wet_mass = pressure_vessel_wet_mass
vessel1_x_left = foam1_x_right
vessel1_x_right = vessel1_x_left + pressure_vessel_outer_diameter
vessel1_x_center = (vessel1_x_left + vessel1_x_right) / 2

battery1_capacity = sympy.Symbol("battery1_capacity")
battery1_displacement = 0
battery1_dry_mass = battery1_capacity / BATTERY_CAPACITY_PER_MASS \
    * (1.0 + BATTERY_DERATING_FACTOR)
battery1_wet_mass = battery1_dry_mass
battery1_x_center = vessel1_x_center
battery1_packing_volume = battery1_capacity / BATTERY_CAPACITY_PER_VOLUME \
    * (1.0 + BATTERY_DERATING_FACTOR) / BATTERY_PACKING_FACTOR

vessel2_displacement = pressure_vessel_outer_volume
vessel2_dry_mass = pressure_vessel_dry_mass
vessel2_wet_mass = pressure_vessel_wet_mass
vessel2_x_left = vessel1_x_right
vessel2_x_right = vessel2_x_left + pressure_vessel_outer_diameter
vessel2_x_center = (vessel2_x_left + vessel2_x_right) / 2

reservoir1_oil_volume = pressure_vessel_inner_volume
reservoir1_displacement = 0
reservoir1_dry_mass_full = reservoir1_oil_volume * OIL_DENSITY
reservoir1_dry_mass_half = reservoir1_dry_mass_full / 2
reservoir1_dry_mass_empty = 0
reservoir1_wet_mass_full = reservoir1_dry_mass_full
reservoir1_wet_mass_half = reservoir1_dry_mass_half
reservoir1_wet_mass_empty = reservoir1_dry_mass_empty
reservoir1_x_center = vessel2_x_center

bladder1_displacement_full = reservoir1_oil_volume
bladder1_displacement_half = bladder1_displacement_full / 2
bladder1_displacement_empty = 0
bladder1_dry_mass_full = bladder1_displacement_full * OIL_DENSITY
bladder1_dry_mass_half = bladder1_displacement_half * OIL_DENSITY
bladder1_dry_mass_empty = bladder1_displacement_empty * OIL_DENSITY
bladder1_wet_mass_full = bladder1_displacement_full * \
    (OIL_DENSITY - WATER_DENSITY_AT_SEA_LEVEL)
bladder1_wet_mass_half = bladder1_displacement_half * \
    (OIL_DENSITY - WATER_DENSITY_AT_SEA_LEVEL)
bladder1_wet_mass_empty = bladder1_displacement_empty * \
    (OIL_DENSITY - WATER_DENSITY_AT_SEA_LEVEL)
bladder1_length_full = bladder1_displacement_full / \
    (pi / 4 * pressure_vessel_outer_diameter ** 2)
bladder1_x_left = vessel2_x_right
bladder1_x_right = bladder1_x_left + bladder1_length_full
bladder1_x_center = (bladder1_x_left + bladder1_x_right) / 2

foam2_length = sympy.Symbol("foam2_length")
foam2_displacement = foam2_length * pi / 4 * \
    (pressure_vessel_outer_diameter ** 2)  # m^3
foam2_dry_mass = foam2_displacement * FOAM_DENSITY  # kg
foam2_wet_mass = foam2_displacement * \
    (FOAM_DENSITY - WATER_DENSITY_AT_SEA_LEVEL)  # kg
foam2_x_left = bladder1_x_right  # m
foam2_x_right = foam2_x_left + foam2_length  # m
foam2_x_center = (foam2_x_left + foam2_x_right) / 2  # m

# TODO: fix these
wing_dry_mass = 1.718  # kg
wing_wet_mass = 1.698  # kg
wing_length = 0.248  # m
wing_displacement = 0.00001950063317  # m^3
wing_x_left = foam2_x_right
wing_x_right = wing_x_left + wing_length
wing_x_center = (wing_x_left + wing_x_right) / 2
wing_z_center = 0.1

movable_roll_diameter = sympy.Symbol("movable_roll_diameter")
movable_roll_volume = movable_roll_length * pi / 4 * movable_roll_diameter ** 2
movable_roll_displacement = movable_roll_volume
movable_roll_dry_mass = movable_roll_volume * LEAD_DENSITY
movable_roll_wet_mass = movable_roll_volume * (LEAD_DENSITY - WATER_DENSITY_AT_SEA_LEVEL)
movable_roll_x_left = wing_x_right
movable_roll_x_right = movable_roll_x_left + movable_roll_length
movable_roll_x_center = (movable_roll_x_left + movable_roll_x_right) / 2
movable_roll_y_center_stb = (pressure_vessel_outer_diameter - movable_roll_diameter) / 2
movable_roll_y_center_mid = 0
movable_roll_y_center_prt = -movable_roll_y_center_stb
movable_roll_z_center = 0

vessel3_displacement = pressure_vessel_outer_volume
vessel3_dry_mass = pressure_vessel_dry_mass
vessel3_wet_mass = pressure_vessel_wet_mass
vessel3_x_left = movable_roll_x_right
vessel3_x_right = vessel3_x_left + pressure_vessel_outer_diameter
vessel3_x_center = (vessel3_x_left + vessel3_x_right) / 2

electronics_displacement = 0
electronics_dry_mass = 5.5 + 6.0  # kg (includes pump and eletronics)
electronics_wet_mass = electronics_dry_mass
electronics_x_center = vessel3_x_center

foam3_length = sympy.Symbol("foam3_length")
foam3_displacement = foam3_length * pi / 4 * \
    (pressure_vessel_outer_diameter ** 2)  # m^3
foam3_dry_mass = foam3_displacement * FOAM_DENSITY  # kg
foam3_wet_mass = foam3_displacement * \
    (FOAM_DENSITY - WATER_DENSITY_AT_SEA_LEVEL)  # kg
foam3_x_left = vessel3_x_right  # m
foam3_x_right = foam3_x_left + foam3_length  # m
foam3_x_center = (foam3_x_left + foam3_x_right) / 2  # m

vessel4_displacement = pressure_vessel_outer_volume
vessel4_dry_mass = pressure_vessel_dry_mass
vessel4_wet_mass = pressure_vessel_wet_mass
vessel4_x_left = foam3_x_right
vessel4_x_right = vessel4_x_left + pressure_vessel_outer_diameter
vessel4_x_center = (vessel4_x_left + vessel4_x_right) / 2

battery2_capacity = sympy.Symbol("battery2_capacity")
battery2_displacement = 0
battery2_dry_mass = battery2_capacity / BATTERY_CAPACITY_PER_MASS \
    * (1.0 + BATTERY_DERATING_FACTOR)
battery2_wet_mass = battery2_dry_mass
battery2_x_center = vessel4_x_center
battery2_packing_volume = battery2_capacity / BATTERY_CAPACITY_PER_VOLUME \
    * (1.0 + BATTERY_DERATING_FACTOR) / BATTERY_PACKING_FACTOR

vehicle_total_length = vessel4_x_right
vehicle_fairing_x_center = vehicle_total_length / 2

# movable pitch is completeley below the other modules
movable_pitch_length = sympy.Symbol("movable_pitch_length")
movable_pitch_volume = movable_pitch_length * pi / 4 * movable_pitch_diameter ** 2
movable_pitch_displacement = movable_pitch_volume
movable_pitch_dry_mass = movable_pitch_volume * LEAD_DENSITY
movable_pitch_wet_mass = movable_pitch_volume * (LEAD_DENSITY - WATER_DENSITY_AT_SEA_LEVEL)
movable_pitch_x_center_fwd = movable_pitch_length / 2
movable_pitch_x_center_aft = vehicle_total_length - movable_pitch_length / 2
movable_pitch_x_center_mid = vehicle_total_length / 2
movable_pitch_y_center = 0
movable_pitch_z_center = (movable_pitch_diameter - vehicle_inner_diameter) / 2

vehicle_total_mass = foam1_dry_mass + vessel1_dry_mass + battery1_dry_mass + \
    vessel2_dry_mass + reservoir1_dry_mass_half + bladder1_dry_mass_half + \
    foam2_dry_mass + wing_dry_mass  + movable_roll_dry_mass + vessel3_dry_mass + \
    electronics_dry_mass + foam3_dry_mass + vessel4_dry_mass + battery2_dry_mass + \
    movable_pitch_dry_mass
vehicle_total_displacement_bladder_full = foam1_displacement + vessel1_displacement + \
    battery1_displacement + vessel2_displacement + reservoir1_displacement + \
    bladder1_displacement_full + foam2_displacement + wing_displacement  + \
    movable_roll_displacement + vessel3_displacement + electronics_displacement + \
    foam3_displacement + vessel4_displacement + battery2_displacement + \
    movable_pitch_displacement
vehicle_total_displacement_bladder_half = foam1_displacement + vessel1_displacement + \
    battery1_displacement + vessel2_displacement + reservoir1_displacement + \
    bladder1_displacement_half + foam2_displacement + wing_displacement  + \
    movable_roll_displacement + vessel3_displacement + electronics_displacement + \
    foam3_displacement + vessel4_displacement + battery2_displacement + \
    movable_pitch_displacement
vehicle_total_displacement_bladder_empty = foam1_displacement + vessel1_displacement + \
    battery1_displacement + vessel2_displacement + reservoir1_displacement + \
    bladder1_displacement_empty + foam2_displacement + wing_displacement  + \
    movable_roll_displacement + vessel3_displacement + electronics_displacement + \
    foam3_displacement + vessel4_displacement + battery2_displacement + \
    movable_pitch_displacement

center_of_gravity_invariable_sum_x = foam1_x_center * foam1_dry_mass + \
    vessel1_x_center * vessel1_dry_mass + \
    battery1_x_center * battery1_dry_mass + \
    vessel2_x_center * vessel2_dry_mass + \
    foam2_x_center * foam2_dry_mass + \
    wing_x_center * wing_dry_mass + \
    vessel3_x_center * vessel3_dry_mass + \
    electronics_x_center * electronics_dry_mass + \
    foam3_x_center * foam3_dry_mass + \
    vessel4_x_center * vessel4_dry_mass + \
    battery2_x_center * battery2_dry_mass
center_of_gravity_invariable_sum_y = 0
center_of_gravity_invariable_sum_z = wing_z_center * wing_dry_mass

center_of_gravity = {
    'bladder_full_pitch_forward_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_fwd * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_full + \
            reservoir1_x_center * reservoir1_dry_mass_empty) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_full_pitch_middle_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_mid * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_full + \
            reservoir1_x_center * reservoir1_dry_mass_empty) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_full_pitch_aft_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_aft * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_full + \
            reservoir1_x_center * reservoir1_dry_mass_empty) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_half_pitch_forward_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_fwd * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_half + \
            reservoir1_x_center * reservoir1_dry_mass_half) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_half_pitch_middle_roll_starboard': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_left * movable_roll_dry_mass + \
            movable_pitch_x_center_mid * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_half + \
            reservoir1_x_center * reservoir1_dry_mass_half) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_stb * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_half_pitch_middle_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_mid * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_half + \
            reservoir1_x_center * reservoir1_dry_mass_half) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_half_pitch_middle_roll_port': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_right * movable_roll_dry_mass + \
            movable_pitch_x_center_mid * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_half + \
            reservoir1_x_center * reservoir1_dry_mass_half) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_prt * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_half_pitch_aft_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_aft * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_half + \
            reservoir1_x_center * reservoir1_dry_mass_half) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_empty_pitch_forward_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_fwd * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_empty + \
            reservoir1_x_center * reservoir1_dry_mass_full) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_empty_pitch_middle_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_mid * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_empty + \
            reservoir1_x_center * reservoir1_dry_mass_full) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass },
    'bladder_empty_pitch_aft_roll_center': {
        'x': (center_of_gravity_invariable_sum_x + \
            movable_roll_x_center * movable_roll_dry_mass + \
            movable_pitch_x_center_aft * movable_pitch_dry_mass + \
            bladder1_x_center * bladder1_dry_mass_empty + \
            reservoir1_x_center * reservoir1_dry_mass_full) / vehicle_total_mass,
        'y': (center_of_gravity_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_dry_mass + \
            movable_roll_y_center_mid * movable_roll_dry_mass) / vehicle_total_mass,
        'z': (center_of_gravity_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_dry_mass + \
            movable_roll_z_center * movable_roll_dry_mass) / vehicle_total_mass }
}

center_of_buoyancy_invariable_sum_x = \
    foam1_x_center * foam1_displacement + \
    vessel1_x_center * vessel1_displacement + \
    battery1_x_center * battery1_displacement + \
    vessel2_x_center * vessel2_displacement + \
    foam2_x_center * foam2_displacement + \
    wing_x_center * wing_displacement + \
    vessel3_x_center * vessel3_displacement + \
    electronics_x_center * electronics_displacement + \
    foam3_x_center * foam3_displacement + \
    vessel4_x_center * vessel4_displacement + \
    battery2_x_center * battery2_displacement
center_of_buoyancy_invariable_sum_y = 0
center_of_buoyancy_invariable_sum_z = 0

center_of_buoyancy = {
    'bladder_full_pitch_forward_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_fwd * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_full) / vehicle_total_displacement_bladder_full,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_full,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_full },
    'bladder_full_pitch_middle_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_mid * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_full) / vehicle_total_displacement_bladder_full,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_full,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_full },
    'bladder_full_pitch_aft_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_aft * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_full) / vehicle_total_displacement_bladder_full,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_full,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_full },
    'bladder_half_pitch_forward_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_fwd * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_half) / vehicle_total_displacement_bladder_half,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_half,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_half },
    'bladder_half_pitch_middle_roll_starboard': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_left * movable_roll_displacement + \
            movable_pitch_x_center_mid * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_half) / vehicle_total_displacement_bladder_half,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_stb * movable_roll_displacement) / vehicle_total_displacement_bladder_half,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_half },
    'bladder_half_pitch_middle_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_mid * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_half) / vehicle_total_displacement_bladder_half,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_half,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_half },
    'bladder_half_pitch_middle_roll_port': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_right * movable_roll_displacement + \
            movable_pitch_x_center_mid * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_half) / vehicle_total_displacement_bladder_half,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_prt * movable_roll_displacement) / vehicle_total_displacement_bladder_half,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_half },
    'bladder_half_pitch_aft_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_aft * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_half) / vehicle_total_displacement_bladder_half,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_half,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_half },
    'bladder_empty_pitch_forward_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_fwd * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_empty) / vehicle_total_displacement_bladder_empty,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_empty,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_empty },
    'bladder_empty_pitch_middle_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_mid * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_empty) / vehicle_total_displacement_bladder_empty,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_empty,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_empty },
    'bladder_empty_pitch_aft_roll_center': {
        'x': (center_of_buoyancy_invariable_sum_x + \
            movable_roll_x_center * movable_roll_displacement + \
            movable_pitch_x_center_aft * movable_pitch_displacement + \
            bladder1_x_center * bladder1_displacement_empty) / vehicle_total_displacement_bladder_empty,
        'y': (center_of_buoyancy_invariable_sum_y + \
            movable_pitch_y_center * movable_pitch_displacement + \
            movable_roll_y_center_mid * movable_roll_displacement) / vehicle_total_displacement_bladder_empty,
        'z': (center_of_buoyancy_invariable_sum_z + \
            movable_pitch_z_center * movable_pitch_displacement + \
            movable_roll_z_center * movable_roll_displacement) / vehicle_total_displacement_bladder_empty }
}

pitch_angle_minimum = sympy.atan((center_of_gravity['bladder_empty_pitch_forward_roll_center']['x'] - \
    center_of_buoyancy['bladder_empty_pitch_forward_roll_center']['x']) / \
    (center_of_buoyancy['bladder_empty_pitch_forward_roll_center']['z'] - \
        center_of_gravity['bladder_empty_pitch_forward_roll_center']['z'])) * 180.0 / pi
pitch_angle_neutral = sympy.atan((center_of_gravity['bladder_half_pitch_middle_roll_center']['x'] - \
    center_of_buoyancy['bladder_half_pitch_middle_roll_center']['x']) / \
    (center_of_buoyancy['bladder_half_pitch_middle_roll_center']['z'] - \
        center_of_gravity['bladder_half_pitch_middle_roll_center']['z'])) * 180.0 / pi
pitch_angle_maximum = sympy.atan((center_of_gravity['bladder_full_pitch_aft_roll_center']['x'] - \
    center_of_buoyancy['bladder_full_pitch_aft_roll_center']['x']) / \
    (center_of_buoyancy['bladder_full_pitch_aft_roll_center']['z'] - \
        center_of_gravity['bladder_full_pitch_aft_roll_center']['z'])) * 180.0 / pi
roll_angle_minimum = sympy.atan((center_of_gravity['bladder_half_pitch_middle_roll_port']['y'] - \
    center_of_buoyancy['bladder_half_pitch_middle_roll_port']['y']) / \
    (center_of_buoyancy['bladder_half_pitch_middle_roll_port']['z'] - 
        center_of_gravity['bladder_half_pitch_middle_roll_port']['z'])) * 180.0 / pi
roll_angle_neutral = sympy.atan((center_of_gravity['bladder_half_pitch_middle_roll_center']['y'] - \
    center_of_buoyancy['bladder_half_pitch_middle_roll_center']['y']) / \
    (center_of_buoyancy['bladder_half_pitch_middle_roll_center']['z'] - 
        center_of_gravity['bladder_half_pitch_middle_roll_center']['z'])) * 180.0 / pi
roll_angle_maximum = sympy.atan((center_of_gravity['bladder_half_pitch_middle_roll_starboard']['y'] - \
    center_of_buoyancy['bladder_half_pitch_middle_roll_starboard']['y']) / \
    (center_of_buoyancy['bladder_half_pitch_middle_roll_starboard']['z'] - 
        center_of_gravity['bladder_half_pitch_middle_roll_starboard']['z'])) * 180.0 / pi

# ---------------

battery1_packing_equation = battery1_packing_volume <= pressure_vessel_inner_volume
battery2_packing_equation = battery2_packing_volume <= pressure_vessel_inner_volume
pitch_angle_minimum_equation = pitch_angle_minimum <= -60
pitch_angle_maximum_equation = pitch_angle_maximum >= 60
pitch_angle_neutral_equation = sympy.abs(pitch_angle_neutral) <= allowable_pitch_error_at_neutral
roll_angle_minimum_equation = roll_angle_minimum <= -20
roll_angle_maximum_equation = roll_angle_maximum >= 20
roll_angle_neutral_equation = sympy.abs(roll_angle_neutral) <= allowable_roll_error_at_neutral


#Assumptions:
#   movable_roll_displacement = volume of entire possible location for the mass
#   movable_pitch_displacement = volume of entire possible location for the mass
#   z-placements of all components are 0 (not true in the spreadsheet for Battery Pack 1)