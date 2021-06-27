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

from typing import Dict, Tuple
import math
import sympy


class Material():
    def __init__(self, name: str,
                 density: float):  # kg/m^3
        self.name = name
        self.density = density


SEA_WATER = Material("Sea_Water", 1027.57516)
OIL_DTE10 = Material("Mobil_DTE_10_Excel_15", 837.0)
LEAD = Material("Lead",  11343.0)


class VesselMaterial(Material):
    def __init__(self,
                 name: str,
                 density: float,        # kg/m^3
                 young_modulus: float,  # Pa
                 yield_stress: float,   # Pa
                 poisson_ratio: float):
        super(VesselMaterial, self).__init__(name, density)
        self.young_modulus = young_modulus
        self.yield_stress = yield_stress
        self.poisson_ration = poisson_ratio


ALUMINIUM_6061_T6 = VesselMaterial(
    "Aluminium_6061_T6",
    2.7e3,
    6.89e10,
    2.76e8,
    0.33)

TITANIUM_TI6AL4V = VesselMaterial(
    "Titanium_Ti6Al4V",
    4.429e3,
    1.138e10,
    1.1e9,
    0.33)


class FoamMaterial(Material):
    def __init__(self,
                 name: str,
                 density: float,        # kg/m^3
                 depth_rating: float):  # m
        super(FoamMaterial, self).__init__(name, density)
        self.depth_rating = depth_rating


FOAM_MZ_22 = FoamMaterial("Foam_MZ_22", 350, 1000)
FOAM_BZ_24 = FoamMaterial("Foam_BZ_24", 385, 2000)
FOAM_BZ_26 = FoamMaterial("Foam_BZ_26", 416, 3000)
