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

from shapes import *
from materials import *


class Module():
    def __init__(self, name: str, shape: 'Shape'):
        self.name = name
        self.shape = shape
        self.bounds = dict()

    def add_param(self, name: str, spec: Union[sympy.Expr, Tuple[float, float]]) \
            -> sympy.Expr:
        if not isinstance(spec, tuple):
            return spec

        name = self.name + "_" + name
        assert name not in self.bounds and len(spec) == 2
        self.bounds[name] = (float(spec[0]), float(spec[1]))
        return sympy.Symbol(name)

    @property
    def volume(self):
        return self.shape.volume

    @property
    def displacement(self):
        return self.volume * SEA_WATER.density


class BuoyancyModule(Module):
    def __init__(self, name: str, shape: 'Cylinder', material: 'Material'):
        super(BuoyancyModule, self).__init__(name, shape)
        self.material = material

    @property
    def dry_mass(self):
        return self.volume * self.material.density


class FoamModule(Module):
    def __init__(self, name: str, shape: 'Cylinder', material: 'FoamMaterial'):
        super(FoamModule, self).__init__(name, shape)
        self.material = material

    @property
    def dry_mass(self):
        return self.volume * self.material.density


class WetModule(Module):
    def __init__(self, name: str, dry_mass: sympy.Expr):
        super(WetModule, self).__init__(name, Point())
        self.dry_mass = dry_mass


class DryModule(Module):
    def __init__(self, name: str):
        super(DryModule, self).__init__(name, Point())

    @property
    def displacement(self):
        return 0


class BatteryPack(DryModule):
    def __init__(self, name: str,
                 capacity:  Union[sympy.Expr, Tuple[float, float]],
                 battery: 'Battery'):
        super(BatteryPack, self).__init__(name, Point())
        self.capacity = self.add_param("capacity", capacity)
        self.battery = battery

    @property
    def dry_mass(self):
        return self.battery.capacity_to_mass(self.capacity)

    @property
    def dry_volume(self):
        return self.battery.capacity_to_packed_volume(self.capacity)


class PressureVessel(Module):
    def __init__(self, name: str, shape: 'Shape',
                 inner_shape: 'Shape', material: 'VesselMaterial'):
        super(PressureVessel, self).__init__(name, shape)
        self.inner_shape = inner_shape
        self.dry_mass = (shape.volume - inner_shape.volume) * material.density


class SimpleVehicle(Module):
    def __init__(self, name: str, shape: 'Cylinder'):
        super(SimpleVehicle, self).__init__(name, shape)

        self.foam1 = FoamModule(
            "foam1",
            Cylinder("foam1", radius=0.0),
            FOAM_BZ_26)

        self.battery1 = BatteryPack("battery1", (0.0, math.inf), BATTERY_LI_THC)
        self.pvessel1 = PressureVessel(
            "pv1",
            Sphere("pv1_outer", 0.338 / 2),
            Sphere("pv1_inner", 0.3148755601 / 2),
            ALUMINIUM_6061_T6)

        self.buoyancy1 = BuoyancyModule(
            "bouyancy1",
            Cylinder("float1"))


if __name__ == '__main__':
    pv1 = PressureVessel(
        "pv1",
        Sphere("pv1_outer", 0.338 / 2),
        Sphere("pv1_inner", 0.3148755601 / 2),
        ALUMINIUM_6061_T6)
    print(float(pv1.dry_mass))
