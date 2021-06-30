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
import sympy
from sympy import pi, cos, sin, tan, sqrt, log, exp, ceiling, atan
from typing import Tuple

from constraint_prog.point_cloud import PointCloud, PointFunc

# lets use floating points
pi = 3.14159265359

WATER_DENSITY_AT_SEA_LEVEL = 1027.0  # kg/m^3
WATER_DENSITY_AT_DIVE_DEPTH = 1041.02  # kg/m^3
OIL_DENSITY = 837.0  # kg/m^3
FOAM_DENSITY = 406.0  # kg/m^3
LEAD_DENSITY = 11343.0  # kg/m^3

ALUMINIUM_DENSITY = 2700.0  # kg/m^3
ALUMINIUM_YOUNG_MODULUS = 6.89e10  # Pa
ALUMINIUM_YIELD_STRESS = 2.76e8  # Pa
ALUMINIUM_POISSON_RATIO = 0.33

BATTERY_CAPACITY_PER_MASS = 621  # Wh / kg
BATTERY_CAPACITY_PER_VOLUME = 1211e3  # Wh / m^3
BATTERY_PACKING_FACTOR = 0.85

DEPTH_RATING_SAFETY_FACTOR = 1.25
GRAVITATIONAL_CONSTANT = 9.806  # m/s^2

# design constants
vehicle_fairing_dry_mass = 8.822  # kg
vehicle_fairing_displacement = 0.0  # kg
vehicle_inner_diameter = 0.43   # m
movable_pitch_diameter = 0.05   # m
movable_roll_length = 0.50  # m
wing_dry_mass = 1.718  # kg
wing_displacement = 0.019  # kg
wing_length = 0.248  # m
wing_z_offset = 0.1  # m
required_battery_capacity = 28000  # Wh
glider_depth_rating = 3000  # m

# calculated automatically
pressure_vessel_outer_diameter = vehicle_inner_diameter - movable_pitch_diameter  # m
glider_crush_pressure = WATER_DENSITY_AT_DIVE_DEPTH * \
    glider_depth_rating * DEPTH_RATING_SAFETY_FACTOR * GRAVITATIONAL_CONSTANT
# aluminum_buckling_failure_cylinder = ((glider_crush_pressure * 0.5 / ALUMINIUM_YOUNG_MODULUS) *
#                                        (1.0 - ALUMINIUM_POISSON_RATIO ** 2)) ** (1.0/3.0) * \
#                                       pressure_vessel_outer_diameter
# aluminum_stress_failure_cylinder = 0.5 * (1.0 - math.sqrt(1.0 - 2 * glider_crush_pressure /
#                                                            ALUMINIUM_YIELD_STRESS)) * \
#                                     pressure_vessel_outer_diameter
aluminum_buckling_failure_sphere = sqrt(glider_crush_pressure *
                                        (0.5 * pressure_vessel_outer_diameter)**2 /
                                        (0.365 * ALUMINIUM_YOUNG_MODULUS))
aluminum_stress_failure_sphere = (glider_crush_pressure * 0.5 * pressure_vessel_outer_diameter) / \
    (2.0 * ALUMINIUM_YIELD_STRESS)

# pressure_vessel_thickness_cylinder = max(aluminum_buckling_failure_cylinder, aluminum_stress_failure_cylinder) * \
#     pressure_vessel_outer_diameter
pressure_vessel_thickness_sphere = max(aluminum_buckling_failure_sphere, aluminum_stress_failure_sphere)
pressure_vessel_inner_diameter = pressure_vessel_outer_diameter - 2 * pressure_vessel_thickness_sphere
pressure_vessel_outer_volume = pi / 6 \
    * (pressure_vessel_outer_diameter ** 3)  # m^3
pressure_vessel_inner_volume = pi / 6 \
    * (pressure_vessel_inner_diameter ** 3)  # m^3
pressure_vessel_dry_mass = (pressure_vessel_outer_volume - pressure_vessel_inner_volume) \
    * ALUMINIUM_DENSITY  # kg
pressure_vessel_displacement = pressure_vessel_outer_volume * WATER_DENSITY_AT_SEA_LEVEL

print("pressure_vessel_outer_diameter:", pressure_vessel_outer_diameter)
print("pressure_vessel_inner_diameter:", pressure_vessel_inner_diameter)
print("pressure_vessel_crush_pressure: ", glider_crush_pressure)
print("pressure_vessel_thickness:", pressure_vessel_thickness_sphere)
print("pressure_vessel_inner_volume:", pressure_vessel_inner_volume)
print("pressure_vessel_dry_mass:", pressure_vessel_dry_mass)
print("pressure_vessel_displacement:", pressure_vessel_displacement)
print()

allowable_pitch_error_at_neutral = 1.5  # degrees
allowable_roll_error_at_neutral = 0.5  # degrees

# ---------------
# center line (z = 0) is along the pressure vessels and foams axis

foam1_length = sympy.Symbol("foam1_length")
foam1_volume = foam1_length * pi / 4 * \
    (pressure_vessel_outer_diameter ** 2)  # m^3
foam1_dry_mass = foam1_volume * FOAM_DENSITY  # kg
foam1_displacement = foam1_volume * WATER_DENSITY_AT_SEA_LEVEL  # kg
foam1_x_left = 0  # m
foam1_x_right = foam1_x_left + foam1_length  # m
foam1_x_center = (foam1_x_left + foam1_x_right) / 2  # m

vessel1_dry_mass = pressure_vessel_dry_mass
vessel1_displacement = pressure_vessel_displacement
vessel1_x_left = foam1_x_right
vessel1_x_right = vessel1_x_left + pressure_vessel_outer_diameter
vessel1_x_center = (vessel1_x_left + vessel1_x_right) / 2

battery1_capacity = sympy.Symbol("battery1_capacity")
battery1_dry_mass = battery1_capacity / BATTERY_CAPACITY_PER_MASS
battery1_x_center = vessel1_x_center
battery1_packing_volume = battery1_capacity / BATTERY_CAPACITY_PER_VOLUME / BATTERY_PACKING_FACTOR

vessel2_dry_mass = pressure_vessel_dry_mass
vessel2_displacement = pressure_vessel_displacement
vessel2_x_left = vessel1_x_right
vessel2_x_right = vessel2_x_left + pressure_vessel_outer_diameter
vessel2_x_center = (vessel2_x_left + vessel2_x_right) / 2

reservoir1_oil_volume = pressure_vessel_inner_volume
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
bladder1_length_full = reservoir1_oil_volume / \
    (pi / 4 * pressure_vessel_outer_diameter ** 2)
bladder1_x_left = vessel2_x_right
bladder1_x_right = bladder1_x_left + bladder1_length_full
bladder1_x_center = (bladder1_x_left + bladder1_x_right) / 2

foam2_length = sympy.Symbol("foam2_length")
foam2_volume = foam2_length * pi / 4 * \
    (pressure_vessel_outer_diameter ** 2)  # m^3
foam2_dry_mass = foam2_volume * FOAM_DENSITY  # kg
foam2_displacement = foam2_volume * WATER_DENSITY_AT_SEA_LEVEL  # kg
foam2_x_left = bladder1_x_right  # m
foam2_x_right = foam2_x_left + foam2_length  # m
foam2_x_center = (foam2_x_left + foam2_x_right) / 2  # m

wing_x_left = foam2_x_right
wing_x_right = wing_x_left + wing_length
wing_x_center = (wing_x_left + wing_x_right) / 2
wing_z_center = (pressure_vessel_outer_diameter - vehicle_inner_diameter) / 2 + wing_z_offset

movable_roll_diameter = sympy.Symbol("movable_roll_diameter")
movable_roll_volume = movable_roll_length * pi / 4 * movable_roll_diameter ** 2
movable_roll_dry_mass = movable_roll_volume * LEAD_DENSITY
movable_roll_displacement = movable_roll_volume * WATER_DENSITY_AT_SEA_LEVEL
movable_roll_x_left = wing_x_right
movable_roll_x_right = movable_roll_x_left + movable_roll_length
movable_roll_x_center = (movable_roll_x_left + movable_roll_x_right) / 2
movable_roll_y_center_stb = (pressure_vessel_outer_diameter - movable_roll_diameter) / 2
movable_roll_y_center_mid = 0
movable_roll_y_center_prt = -movable_roll_y_center_stb
movable_roll_z_center = (pressure_vessel_outer_diameter - vehicle_inner_diameter) / 2

vessel3_dry_mass = pressure_vessel_dry_mass
vessel3_displacement = pressure_vessel_displacement
vessel3_x_left = movable_roll_x_right
vessel3_x_right = vessel3_x_left + pressure_vessel_outer_diameter
vessel3_x_center = (vessel3_x_left + vessel3_x_right) / 2

electronics_dry_mass = 5.5 + 6.0  # kg (includes pump and electronics)
electronics_x_center = vessel3_x_center

foam3_length = sympy.Symbol("foam3_length")
foam3_volume = foam3_length * pi / 4 * \
    (pressure_vessel_outer_diameter ** 2)  # m^3
foam3_dry_mass = foam3_volume * FOAM_DENSITY  # kg
foam3_displacement = foam3_volume * WATER_DENSITY_AT_SEA_LEVEL  # kg
foam3_x_left = vessel3_x_right  # m
foam3_x_right = foam3_x_left + foam3_length  # m
foam3_x_center = (foam3_x_left + foam3_x_right) / 2  # m

vessel4_dry_mass = pressure_vessel_dry_mass
vessel4_displacement = pressure_vessel_displacement
vessel4_x_left = foam3_x_right
vessel4_x_right = vessel4_x_left + pressure_vessel_outer_diameter
vessel4_x_center = (vessel4_x_left + vessel4_x_right) / 2

battery2_capacity = sympy.Symbol("battery2_capacity")
battery2_dry_mass = battery2_capacity / BATTERY_CAPACITY_PER_MASS
battery2_x_center = vessel4_x_center
battery2_packing_volume = battery2_capacity / BATTERY_CAPACITY_PER_VOLUME / BATTERY_PACKING_FACTOR

vehicle_inner_length = vessel4_x_right
vehicle_fairing_x_center = vehicle_inner_length / 2

# movable pitch is completeley below the other modules
movable_pitch_length = sympy.Symbol("movable_pitch_length")
movable_pitch_volume = movable_pitch_length * pi / 4 * movable_pitch_diameter ** 2
movable_pitch_dry_mass = movable_pitch_volume * LEAD_DENSITY
movable_pitch_displacement = movable_pitch_volume * WATER_DENSITY_AT_SEA_LEVEL
movable_pitch_x_center_fwd = movable_pitch_length / 2
movable_pitch_x_center_aft = vehicle_inner_length - movable_pitch_length / 2
movable_pitch_x_center_mid = vehicle_inner_length / 2
movable_pitch_y_center = 0
movable_pitch_z_center = -(pressure_vessel_outer_diameter + movable_pitch_diameter) / 2

vehicle_dry_mass = foam1_dry_mass + vessel1_dry_mass + battery1_dry_mass + \
    vessel2_dry_mass + reservoir1_dry_mass_half + bladder1_dry_mass_half + \
    foam2_dry_mass + wing_dry_mass + movable_roll_dry_mass + vessel3_dry_mass + \
    electronics_dry_mass + foam3_dry_mass + vessel4_dry_mass + battery2_dry_mass + \
    movable_pitch_dry_mass


def get_center_of_gravity(bladder: str, pitch: str, roll: str) \
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

    total_mass += movable_pitch_dry_mass
    total_y_sum += movable_pitch_dry_mass * movable_pitch_y_center
    total_z_sum += movable_pitch_dry_mass * movable_pitch_z_center
    if pitch == "forward":
        total_x_sum += movable_pitch_dry_mass * movable_pitch_x_center_fwd
    elif pitch == "middle":
        total_x_sum += movable_pitch_dry_mass * movable_pitch_x_center_mid
    else:
        assert pitch == "aft"
        total_x_sum += movable_pitch_dry_mass * movable_pitch_x_center_aft

    return (total_mass, total_x_sum / total_mass, total_y_sum / total_mass, total_z_sum / total_mass)


def get_center_of_buoyancy(bladder: str, pitch: str, roll: str) \
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

    total_mass += movable_pitch_displacement
    total_y_sum += movable_pitch_displacement * movable_pitch_y_center
    total_z_sum += movable_pitch_displacement * movable_pitch_z_center
    if pitch == "forward":
        total_x_sum += movable_pitch_displacement * movable_pitch_x_center_fwd
    elif pitch == "middle":
        total_x_sum += movable_pitch_displacement * movable_pitch_x_center_mid
    else:
        assert pitch == "aft"
        total_x_sum += movable_pitch_displacement * movable_pitch_x_center_aft

    return (total_mass, total_x_sum / total_mass, total_y_sum / total_mass, total_z_sum / total_mass)


def get_buoyancy_minus_gravity(bladder: str, pitch: str, roll: str) \
        -> Tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]:
    cb_mass, cb_x, cb_y, cb_z = get_center_of_buoyancy(bladder, pitch, roll)
    cg_mass, cg_x, cg_y, cg_z = get_center_of_gravity(bladder, pitch, roll)
    return cb_mass - cg_mass, cb_x - cg_x, cb_y - cg_y, cb_z - cg_z


battery1_packing_equation = battery1_packing_volume <= pressure_vessel_inner_volume
battery2_packing_equation = battery2_packing_volume <= pressure_vessel_inner_volume
battery_capacity_equation = battery1_capacity + battery2_capacity >= required_battery_capacity

pitch_minimum_cbmg_buoyancy, pitch_minimum_cbmg_x, pitch_minimum_cbmg_y, pitch_minimum_cbmg_z = \
    get_buoyancy_minus_gravity(bladder="empty", pitch="forward", roll="center")
pitch_minimum_equation1 = -pitch_minimum_cbmg_x / pitch_minimum_cbmg_z <= math.tan(-60 * pi / 180)
pitch_minimum_equation2 = pitch_minimum_cbmg_buoyancy <= -10.0  # sinks to bottom

pitch_maximum_cbmg_buoyancy, pitch_maximum_cbmg_x, pitch_maximum_cbmg_y, pitch_maximum_cbmg_z = \
    get_buoyancy_minus_gravity(bladder="full", pitch="aft", roll="center")
pitch_maximum_equation = -pitch_maximum_cbmg_x / pitch_maximum_cbmg_z >= math.tan(60 * pi / 180)

pitch_neutral_cbmg_buoyancy, pitch_neutral_cbmg_x, pitch_neutral_cbmg_y, pitch_neutral_cbmg_z = \
    get_buoyancy_minus_gravity(bladder="half", pitch="middle", roll="center")
pitch_neutral_equation1 = -pitch_neutral_cbmg_x / pitch_neutral_cbmg_z <= math.tan(1.5 * pi / 180)
pitch_neutral_equation2 = -pitch_neutral_cbmg_x / pitch_neutral_cbmg_z >= math.tan(-1.5 * pi / 180)
pitch_neutral_equation3 = pitch_neutral_cbmg_buoyancy >= 2.0  # floats to surface

roll_minimum_cbmg_buoyancy, roll_minimum_cbmg_x, roll_minimum_cbmg_y, roll_minimum_cbmg_z = \
    get_buoyancy_minus_gravity(bladder="half", pitch="middle", roll="port")
roll_minimum_equation = -roll_minimum_cbmg_y / roll_minimum_cbmg_z <= math.tan(-20 * pi / 180)

constraints = PointFunc({
    "battery1_packing_equation": battery1_packing_equation,
    "battery2_packing_equation": battery2_packing_equation,
    "battery_capacity_equation": battery_capacity_equation,
    "pitch_minimum_equation1": pitch_minimum_equation1,
    # "pitch_minimum_equation2": pitch_minimum_equation2,
    "pitch_neutral_equation1": pitch_neutral_equation1,
    "pitch_neutral_equation2": pitch_neutral_equation2,
    "pitch_neutral_equation3": pitch_neutral_equation3,
    "pitch_maximum_equation": pitch_maximum_equation,
    "roll_minimum_equation": roll_minimum_equation,
    "finess_ratio_equation": vehicle_inner_length <= 8 * vehicle_inner_diameter,
})

print(constraints.input_names)
print(constraints.output_names)

derived_values = PointFunc({
    "battery1_packing_volume": battery1_packing_volume,
    "battery2_packing_volume": battery2_packing_volume,
    "battery1_dry_mass": battery1_dry_mass,
    "battery2_dry_mass": battery2_dry_mass,
    "foam1_dry_mass": foam1_dry_mass,
    "foam2_dry_mass": foam2_dry_mass,
    "foam3_dry_mass": foam3_dry_mass,
    "movable_pitch_dry_mass": movable_pitch_dry_mass,
    "movable_roll_dry_mass": movable_roll_dry_mass,
    "pitch_minimum_buoyancy": pitch_minimum_cbmg_buoyancy,
    "pitch_minimum_angle": atan(-pitch_minimum_cbmg_x / pitch_minimum_cbmg_z) * 180 / pi,
    "pitch_neutral_buoyancy": pitch_neutral_cbmg_buoyancy,
    "pitch_neutral_angle": atan(-pitch_neutral_cbmg_x / pitch_neutral_cbmg_z) * 180 / pi,
    "pitch_maximum_buoyancy": pitch_maximum_cbmg_buoyancy,
    "pitch_maximum_angle": atan(-pitch_maximum_cbmg_x / pitch_maximum_cbmg_z) * 180 / pi,
    "roll_minimum_mass": roll_minimum_cbmg_buoyancy,
    "roll_minimum_angle": atan(-roll_minimum_cbmg_y / roll_minimum_cbmg_z) * 180 / pi,
    "total_battery_capacity": battery1_capacity + battery2_capacity,
    "vehicle_dry_mass": vehicle_dry_mass,
    "vehicle_inner_length": vehicle_inner_length,
    "vehicle_finess_ratio": vehicle_inner_length / vehicle_inner_diameter,
    "wing_x_center": wing_x_center,
    "wing_z_center": wing_z_center,
})


def print_solutions(points, num=None):
    points = points.extend(derived_values(points, equs_as_float=False))
    if num is None:
        num = points.num_points
    else:
        num = min(num, points.num_points)
    for sol in range(num):
        print("solution #:", sol)
        for idx, var in enumerate(points.float_vars):
            print(var + ":", points.float_data[sol, idx].item())
        print()


bounds = {
    "battery1_capacity": (0.0, required_battery_capacity),
    "battery2_capacity": (0.0, required_battery_capacity),
    "foam1_length": (0.0, 2.0),
    "foam2_length": (0.0, 2.0),
    "foam3_length": (0.0, 2.0),
    "movable_pitch_length": (0.10, 1.0),
    "movable_roll_diameter": (0.05, 0.5),
}

assert list(bounds.keys()) == list(constraints.input_names)

# generate random points
points = PointCloud.generate(bounds, 5000)

# minimize errors with newton raphson
points = points.newton_raphson(constraints, bounds)
# points.print_info()

# calculate remaining errors
errors = constraints(points)
# errors.print_info()

# check constraints with loose tolerance
points = points.prune_by_tolerances(errors, 0.5)
print(points.num_points)

for _ in range(5):
    if not points.num_points:
        print("no points left")
        break
    points.add_mutations([1000, 1000, 0.1, 0.1, 0.1, 0.1, 0.1], 5000)
    points = points.newton_raphson(constraints, bounds)
    points = points.prune_by_tolerances(constraints(points), 0.1)
    points = points.prune_close_points([500, 500, 0.05, 0.05, 0.05, 0.05, 0.05])
    print(points.num_points)

if points.num_points:
    points.plot2d(2, 3)
    print_solutions(points, 10)
