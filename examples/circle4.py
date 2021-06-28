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

import sympy


# variables
x1 = sympy.Symbol("x1")
y1 = sympy.Symbol("y1")
x2 = sympy.Symbol("x2")
y2 = sympy.Symbol("y2")
x3 = sympy.Symbol("x3")
y3 = sympy.Symbol("y3")
r = sympy.Symbol("r")

# expressions
norm1 = x1 ** 2 + y1 ** 2 - (r-1) ** 2
norm2 = x2 ** 2 + y2 ** 2 - (r-1) ** 2
norm3 = x3 ** 2 + y3 ** 2 - (r-1) ** 2
dist1 = x1 ** 2 + (y1-r+1) ** 2
dist2 = x2 ** 2 + (y2-r+1) ** 2
dist3 = x3 ** 2 + (y3-r+1) ** 2
dist4 = (x1-x2) ** 2 + (y1-y2) ** 2
dist5 = (x1-x3) ** 2 + (y1-y3) ** 2
dist6 = (x2-x3) ** 2 + (y2-y3) ** 2

# equations
equ1 = sympy.LessThan(norm1, 0)
equ2 = sympy.LessThan(norm2, 0)
equ3 = sympy.LessThan(norm3, 0)
equ4 = sympy.GreaterThan(dist1, 4)
equ5 = sympy.GreaterThan(dist2, 4)
equ6 = sympy.GreaterThan(dist3, 4)
equ7 = sympy.GreaterThan(dist4, 4)
equ8 = sympy.GreaterThan(dist5, 4)
equ9 = sympy.GreaterThan(dist6, 4)
equ10 = sympy.LessThan(r, 2.4143)
