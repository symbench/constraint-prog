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

    def mass_to_volume(self, mass: sympy.Expr) -> sympy.Expr:
        return mass / self.density

    def volume_to_mass(self, volume: sympy.Expr) -> sympy.Expr:
        return volume * self.density


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


# TODO: Do a better representation for individual cells
class Battery(Material):
    def __init__(self,
                 name: str,
                 capacity_per_mass: float,    # Wh/kg
                 capacity_per_volume: float,  # Wh/m^3
                 derating_factor: float,
                 packing_factor: float):
        super(Battery, self).__init__(
            name, capacity_per_volume / capacity_per_mass)
        self.capacity_per_mass = capacity_per_mass / (1.0 + derating_factor)
        self.capacity_per_volume = capacity_per_volume / (1.0 + derating_factor) * packing_factor

    def capacity_to_mass(self, capacity: sympy.Expr) -> sympy.Expr:
        """Already taking the derating factor into account."""
        return capacity / self.capacity_per_mass

    def mass_to_capacity(self, volume: sympy.Expr) -> sympy.Expr:
        """Already taking the derating factor into account."""
        return volume * self.capacity_per_mass

    def capacity_to_packed_volume(self, capacity: sympy.Expr) -> sympy.Expr:
        """Already taking the derating and packing factors into account."""
        return capacity / (self.capacity_per_volume)

    def packed_volume_to_capacity(self, volume: sympy.Expr) -> sympy.Expr:
        """Already taking the derating and packing factors into account."""
        return volume * self.capacity_per_volume


BATTERY_LI_ION = Battery("Batter_Lithium_Ion", 500, 615e3, 0.2, 0.85)
BATTERY_LI_THC = Battery("Batter_Lithium_Thionyl_Chloride", 621, 1211e3, 0.2, 0.85)
