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

from typing import Dict, Optional

import json
import math
import sympy

from constraint_prog.point_cloud import PointCloud, PointFunc
from constraint_prog.sympy_func import SympyFunc


def run_analysis(motor_prop: Dict[str, float],
                 batt: Dict[str, float],
                 wing: Dict[str, float],
                 forward_count: Optional[int] = None):
    assert forward_count is None or isinstance(forward_count, int)

    series_count = int(math.floor(motor_prop["max_voltage"] / batt["voltage"]))
    parallel_count = sympy.Symbol("parallel_count")
    battery_pack = {
        "weight": batt["weight"] * series_count * parallel_count,
        "volume": batt["volume"] * series_count * parallel_count,
        "energy": batt["energy"] * series_count * parallel_count,
        "voltage": batt["voltage"] * series_count,
        "current": batt["current"] * parallel_count,
    }

    lifting_count = sympy.Symbol("lifting_count")
    lifting_motor_prop = {
        "weight": motor_prop["weight"] * lifting_count,
        "thrust": motor_prop["thrust"] * lifting_count,
        "power": motor_prop["power"] * lifting_count,
        "current": motor_prop["current"] * lifting_count,
    }

    if not isinstance(forward_count, int):
        forward_count = sympy.Symbol("forward_count")
    forward_motor_prop = {
        "weight": motor_prop["weight"] * forward_count,
        "thrust": motor_prop["thrust"] * forward_count,
        "power": motor_prop["power"] * forward_count,
        "current": motor_prop["current"] * forward_count,
    }

    wing_count = 2
    flying_speed = sympy.Symbol("flying_speed")  # m/s
    moving_wing = {
        "weight": wing["weight"] * wing_count,
        "available_volume": wing["available_volume"] * wing_count,
        "lift_force": wing_count * wing["lift_50mps"] * (flying_speed / 50.0) ** 2,
        "drag_force": wing_count * wing["drag_50mps"] * (flying_speed / 50.0) ** 2,
    }

    air_density = 1.225                # kg/m^3
    frontal_area = 2012345 * 1e-6      # m^2
    fuselage = {
        "weight": 400,
        "drag_force": 0.5 * air_density * frontal_area * flying_speed ** 2,
    }

    aircraft_weight = fuselage["weight"] + battery_pack["weight"] + \
        forward_motor_prop["weight"] + \
        lifting_motor_prop["weight"] + moving_wing["weight"]
    hower_time = battery_pack["energy"] / lifting_motor_prop["power"] * 3600.0
    flying_time = battery_pack["energy"] / forward_motor_prop["power"] * 3600.0
    flying_distance = flying_time * flying_speed

    gravitation = 9.81                 # m/s^2
    constraints = {
        "available_volume_equ": battery_pack["volume"] <= moving_wing["available_volume"],
        "hower_current_equ": battery_pack["current"] >= lifting_motor_prop["current"],
        "hower_thrust_equ": lifting_motor_prop["thrust"] >= aircraft_weight * gravitation,
        "flying_current_equ": battery_pack["current"] >= forward_motor_prop["current"],
        "flying_lift_equ": moving_wing["lift_force"] >= aircraft_weight * gravitation,
        "flying_thrust_equ": forward_motor_prop["thrust"] >= fuselage["drag_force"] + moving_wing["drag_force"],
    }

    bounds = {
        "parallel_count": (1.0, 1e6),
        "lifting_count": (1.0, 1e6),
        "forward_count": (1.0, 1e6),
        "flying_speed": (0.0, 50.0),
    }

    if isinstance(forward_count, int):
        del bounds["forward_count"]

    report_func = PointFunc({
        "series_count": series_count,
        "aircraft_weight": aircraft_weight,
        "hower_time": hower_time,
        "flying_time": flying_time,
        "flying_distance": flying_distance,
        "battery_pack_voltage": battery_pack["voltage"],
        "battery_pack_volume": battery_pack["volume"],
        "battery_pack_current": battery_pack["current"],
        "battery_pack_energy": battery_pack["energy"],
        "lifting_motor_current": lifting_motor_prop["current"],
        "lifting_motor_power": lifting_motor_prop["power"],
        "lifting_motor_thrust": lifting_motor_prop["thrust"],
        "forward_motor_current": forward_motor_prop["current"],
        "forward_motor_power": forward_motor_prop["power"],
        "forward_motor_thrust": forward_motor_prop["thrust"],
        "wing_available_volume": moving_wing["available_volume"],
    })

    # generate random points
    num = 5000
    points = PointCloud.generate(bounds, num)
    constraints_func = PointFunc(constraints)

    for step in range(5):
        points.add_mutations(2.0, num)

        points = points.newton_raphson(constraints_func, bounds, num_iter=10)
        points = points.prune_by_tolerances(constraints_func(points), {
            "available_volume_equ": 0.01,
            "hower_current_equ": 0.1,
            "hower_thrust_equ": 1.0,
            "flying_current_equ": 0.1,
            "flying_lift_equ": 1.0,
            "flying_thrust_equ": 1.0,
        })
        if False:
            points = points.prune_close_points2(resolutions={
                "parallel_count": 0.1,
                "lifting_count": 0.1,
                "forward_count": 0.1,
                "flying_speed": 0.1,
            })

        points = points.extend(report_func(points))
        if True:
            points = points.prune_pareto_front2({
                "flying_distance": 1.0,
                # "flying_speed": 1.0,
            })

        if points.num_points:
            print(json.dumps(points.row(0), indent=2))

    # points.plot2d("flying_distance", "flying_speed")


def run1():
    motor_prop = {
        "motor_name": "KDE13218XF-105",
        "propeller_name": "34x3_2_4600_41_250",
        "weight": 2.313,
        "propeller_diameter": 0.8636,
        "voltage": 25.0,
        "thrust": 106.04,
        "power": 1743.54,
        "current": 69.74,
        "max_voltage": 38.29,
        "max_thrust": 40.39,
        "max_power": 6045.95,
        "max_current": 157.89,
    }

    batt = {
        "weight": 0.004,
        "volume": 8.75e-05,
        "energy": 11.1,
        "voltage": 11.1,
        "current": 25.0,
        "name": "Vitaly Beta"
    }

    wing = {
        "weight": 29.487522,
        "lift_50mps": 9287.95,
        "drag_50mps": 440.14,
        "available_volume": 0.360,
        "profile_name": "NACA 2418",
        "chord": 1.0,
        "span": 10.0
    }

    run_analysis(motor_prop, batt, wing)


def run2():
    motor_prop = {
        "motor_name": "MAGiDRIVE150",
        "propeller_name": "62x5_2_3200_46_1150",
        "weight": 35.831,
        "propeller_diameter": 1.5748,
        "voltage": 700.0,
        "thrust": 1444.34,
        "power": 74098.07,
        "current": 105.85,
        "max_voltage": 845.1,
        "max_thrust": 1422.26,
        "max_power": 126315.79,
        "max_current": 149.47,
    }

    batt = {
        "weight": 0.004,
        "volume": 8.75e-05,
        "energy": 11.1,
        "voltage": 11.1,
        "current": 25.0,
        "name": "Vitaly Beta"
    }

    wing = {
        "weight": 29.487522,
        "lift_50mps": 9287.95,
        "drag_50mps": 440.14,
        "available_volume": 0.360,
        "profile_name": "NACA 2418",
        "chord": 1.0,
        "span": 10.0
    }

    run_analysis(motor_prop, batt, wing, forward_count=2)


if __name__ == '__main__':
    run1()
