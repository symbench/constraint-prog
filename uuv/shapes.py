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


class Shape():
    def __init__(self, name: str):
        self.name = name


class Point(Shape):
    def __init__(self, name: str):
        super(Point, self).__init__(name)

    @property
    def volume(self) -> sympy.Expr:
        return 0

    @property
    def external_length(self):
        return 0

    @property
    def external_diameter(self):
        return 0


class Cylinder(Shape):
    def __init__(self, name: str):
        super(Cylinder, self).__init__(name)

    @property
    def volume(self) -> sympy.Expr:
        return sympy.pi * (self.radius ** 2) * self.length

    @property
    def external_length(self):
        return self.length

    @property
    def external_diameter(self):
        return 2 * self.radius

    def enlarge(self, thickness: sympy.Expr) -> 'CylinderExpr':
        return CylinderExpr(self.name + "_enlarged",
                            self.radius + thickness,
                            self.length + thickness)


class CylinderSymb(Cylinder):
    def __init__(self, name: str,
                 radius_min: float = 0.0,
                 radius_max: float = math.inf,
                 length_min: float = 0.0,
                 length_max: float = math.inf):
        super(CylinderSymb, self).__init__(name)

        assert radius_min <= radius_max
        self.radius = sympy.Symbol(name + "_radius")
        self.radius_min = radius_min
        self.radius_max = radius_max

        assert length_min <= length_max
        self.length = sympy.Symbol(name + "_length")
        self.length_min = length_min
        self.length_max = length_max

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            self.name + "_radius": (self.radius_min, self.radius_max),
            self.name + "_length": (self.length_min, self.length_max),
        }


class CylinderExpr(Cylinder):
    def __init__(self, name: str,
                 radius: sympy.Expr,
                 length: sympy.Expr):
        super(CylinderExpr, self).__init__(name)
        self.radius = radius
        self.length = length

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return dict()


if __name__ == '__main__':
    c1 = CylinderSymb("hihi")
    c2 = CylinderExpr("haha", c1.radius, c1.length)
    print(c2.volume)


class Capsule(Shape):
    def __init__(self, name: str):
        super(Capsule, self).__init__(name)

    @property
    def volume(self) -> sympy.Expr:
        return sympy.pi * (self.radius ** 2) * (4 * self.radius / 3 + self.length)

    @property
    def external_length(self):
        return 2 * self.radius + self.length

    @property
    def external_diameter(self):
        return 2 * self.radius

    def enlarge(self, thickness: sympy.Expr) -> 'CapsuleExpr':
        return CapsuleExpr(self.name + "_enlarged",
                           self.radius + thickness,
                           self.length)


class CapsuleSymb(Capsule):
    def __init__(self, name: str,
                 radius_min: float = 0.0,
                 radius_max: float = math.inf,
                 length_min: float = 0.0,
                 length_max: float = math.inf):
        super(CapsuleSymb, self).__init__(name)

        assert radius_min <= radius_max
        self.radius = sympy.Symbol(name + "_radius")
        self.radius_min = radius_min
        self.radius_max = radius_max

        assert length_min <= length_max
        self.length = sympy.Symbol(name + "_length")
        self.length_min = length_min
        self.length_max = length_max

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            self.name + "_radius": (self.radius_min, self.radius_max),
            self.name + "_length": (self.length_min, self.length_max),
        }


class CapsuleExpr(Capsule):
    def __init__(self, name: str,
                 radius: sympy.Expr,
                 length: sympy.Expr):
        super(CapsuleExpr, self).__init__(name)
        self.radius = radius
        self.length = length

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return dict()
