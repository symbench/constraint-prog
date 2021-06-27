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

import shapes
import materials


class Module():
    def __init__(self, name: str, shape: 'Shape'):
        self.name = name
        self.shape = shape

    @property
    def volume(self):
        return self.shape.volume

    @property
    def displacement(self):
        return self.volume * materials.SEA_WATER.density


class BuoyancyModule(Module):
    def __init__(self, name: str, shape: 'Cylinder', material: 'Material'):
        super(BuoyancyModule, self).__init__(name, shape)
        self.material = material

    @property
    def dry_mass(self):
        return self.volume * self.material.density


class FloatModule(Module):
    def __init__(self, name: str, shape: 'Cylinder', material: 'FoamMaterial'):
        super(FloatModule, self).__init__(name, shape)
        self.material = material

    @property
    def dry_mass(self):
        return self.volume * self.material.density


class WetModule(Module):
    def __init__(self, name: str, dry_mass: sympy.Expr):
        super(WetModule, self).__init__(name, shapes.Point())
        self.dry_mass = dry_mass


class DryModule(Module):
    def __init__(self, name: str, dry_mass: sympy.Expr):
        super(DryModule, self).__init__(name, shapes.Point())
        self.dry_mass = dry_mass


class PressureVessel(Module):
    def __init__(self, name: str, dry_mass: sympy.Expr,
                 outer_radius: sympy.Expr, inner_radius: sympy.Expr):
        super(PressureVessel, self).__init__(name, )
        self.dry_mass = dry_mass


class SimpleVehicle(Module):
    def __init__(self, name: str, shape: 'Cylinder'):

        self.float1 = FloatModule(
            "float1",
            shapes.CylinderSymb("float1"),
            materials.FOAM_BZ_26)

        self.battery1 = PressureVessel(
            "battery1")

        self.buoyancy1 = BuoyancyModule(
            "bouyancy1",
            shapes.CylinderSymb("float1"))


if __name__ == '__main__':
    flo1 = FloatModule(
        "FLO1",
        shapes.Cylinder("FLO1", 0.15, 0.25),
        materials.FOAM_BZ_26)
    print(float(flo1.dry_mass))
