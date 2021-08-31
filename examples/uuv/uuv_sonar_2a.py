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

import math
import sympy


class Sonar(object):
    # (kHz, log(kHz), log(range))
    TABLE = [
        (70.0, 1.84509804001426, 2.77815125038364),
        (120.0, 2.07918124604762, 2.39794000867204),
        (270.0, 2.43136376415899, 2.17609125905568),
        (410.0, 2.61278385671974, 2, 11394335230684),
        (540.0, 2.73239375982297, 2.0),
        (850.0, 2.92941892571429, 1, 69897000433602),
    ]

    def __init__(self, name: str, band: int):
        assert 0 <= band < len(Sonar.TABLE) - 1
        self.name = name
        self.band = band

        self.coefficient = sympy.Symbol(name + '_coefficient')
        self.resolution = sympy.Symbol(name + '_resolution')  # m

    @property
    def bounds(self):
        return {
            self.name + '_coefficient': (0.0, 1.0),
            self.name + '_resolution': (0.0, math.inf),
        }

    @property
    def frequency(self):
        return 10.0 ** (Sonar.TABLE[self.band][1] * (1.0 - self.coefficient) +
                        Sonar.TABLE[self.band + 1][1] * self.coefficient)  # kHz

    @property
    def range(self):
        return 10.0 ** (Sonar.TABLE[self.band][2] * (1.0 - self.coefficient) +
                        Sonar.TABLE[self.band + 1][2] * self.coefficient)  # m

    @property
    def power(self):
        return 15.0 + 0.05 * self.range  # W

    @property
    def length(self):
        return 0.0254 * 3000 / self.frequency / (sympy.atan2(self.resolution, self.range) * (180.0 / math.pi))

    @property
    def wet_weight(self):
        return 2 * 1.5 * self.length / 0.5


class Vessel(object):
    # (material, young_modulus, yield_stress, poisson_ratio, density)
    TABLE = [
        ("Aluminum 6061", 6.89e10, 3.96e8, 0.33, 2.7),
        ("Titanium Ti6Al4V", 1.138e11, 1.48e9, 0.342, 4.429),
        ("Stainless Steel 304", 1.93e11, 2.15e8, 0.29, 8.0),
        ("Aluminum Oxide", 3.4e11, 2.5e9, 0.22, 3.95),
    ]

    def __init__(self, name: str, material: int):
        assert 0 <= material < len(Vessel.TABLE)
        self.name = name
        self.material = Vessel.TABLE[material][0]
        self.young_modulus = Vessel.TABLE[material][1]
        self.yield_stress = Vessel.TABLE[material][2]
        self.poisson_ratio = Vessel.TABLE[material][3]
        self.density = Vessel.TABLE[material][4]

        self.crush_depth = sympy.Symbol(name + '_chrush_depth')
        self.diameter = sympy.Symbol(name + '_diameter')
        self.cylinder_length = sympy.Symbol(name + '_cylinder_length')

    @property
    def bounds(self):
        return {
            self.name + '_chrush_depth': (0.0, math.inf),
            self.name + '_diameter': (0.0, math.inf),
            self.name + '_cylinder_length': (0.0, math.inf),
        }


vessel = Vessel("vessel", 0)
print(vessel.bounds)

# mission parameters
total_system_weight = 5000.0  # kg
reference_displacement = 130.32383  # kg
reference_diameter = 0.3048  # m
reference_length = 2.2  # m

# parameters
vehicle_depth_rating = 1000.0  # m
number_of_vehicles = 2

vehicle_length = 7.22

# derived values

vehicle_fairing_displacement = total_system_weight / number_of_vehicles

vehicle_diameter = 0.3048 * (vehicle_fairing_displacement/130.32383) ** (1/3)
vehicle_length = 2.2 * (vehicle_fairing_displacement/130.32383) ** (1/3)
