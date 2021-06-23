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


class RelativePos:
    """
    This represents a relative position of one component inside another.
    Each component has its own natural origin point (which is usually
    not the center of its gravity).
    """

    def __init__(self, name: str):
        self.name = name

        self.pos_x = sympy.Symbol(name + "_pos_x")  # in m
        self.pos_y = sympy.Symbol(name + "_pos_y")  # in m
        self.pos_z = sympy.Symbol(name + "_pos_z")  # in m

        self.pos_x_bounds = (-1e30, 1e30)
        self.pos_y_bounds = (-1e30, 1e30)
        self.pos_z_bounds = (-1e30, 1e30)

    @property
    def variables(self) -> Dict[str, sympy.Expr]:
        return {
            self.name + "_pos_x": self.pos_x,
            self.name + "_pos_y": self.pos_y,
            self.name + "_pos_z": self.pos_z,
        }

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            self.name + "_pos_x": self.pos_x_bounds,
            self.name + "_pos_y": self.pos_y_bounds,
            self.name + "_pos_z": self.pos_z_bounds,
        }


class CapsuleShape:
    """
    This represents a capsule shaped object with two semispheres connected by
    a cylinder. The class maintains the two variables describing this capsule,
    its radius and length in meters. If the length is zero, then the capsule
    is a sphere. The capsule is always oriented along the x-axis.
    """

    def __init__(self, name: str):
        self.name = name

        self.radius = sympy.Symbol(name + "_radius")  # in m
        self.length = sympy.Symbol(name + "_length")  # in m

        self.radius_bounds = (0.0, 1e30)
        self.length_bounds = (0.0, 1e30)

    @property
    def variables(self) -> Dict[str, sympy.Expr]:
        return {
            self.name + "_radius": self.radius,
            self.name + "_length": self.length,
        }

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            self.name + "_radius": self.radius_bounds,
            self.name + "_length": self.length_bounds,
        }


def is_capsule_inside_another(capsule1: 'CapsuleShape', capsule2: 'CapsuleShape',
                              relpos2: 'RelativePos') -> Dict[str, sympy.Expr]:
    """
    Returns a dictionary of differentiable constraints expressing the fact
    capsule1 contains capsule2.
    """
    pass


if __name__ == "__main__":
    capsule1 = CapsuleShape("capsule1")
    capsule2 = CapsuleShape("capsule2")
    relpos2 = RelativePos("capsule2")
    print(capsule1.bounds)
    print(relpos2.variables)
