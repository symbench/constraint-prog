#!/usr/bin/env python3
# Copyright (C) 2022, Miklos Maroti
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

from typing import Any, Optional, Set

import os
import re
import sympy
import tempfile

import pypeg2
from OMPython import OMCSessionZMQ


class Name(str):
    gramar = re.compile('[a-zA-Z-\.]*')

def run2():
    value = pypeg2.parse("Hello", Name)
    print(value)


class ModelicaSession:
    def __init__(self):
        self.session = OMCSessionZMQ()
        print("Opening", self.send("getVersion()"))
        print("Loading standard library:", self.send("loadModel(Modelica)"))

    def send(self, expr: str) -> Any:
        return self.session.sendExpression(expr)

    def load_library(self, filename: Optional[str] = None):
        if filename is None:
            filename = os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                '..',
                '..',
                'modelica-symbench',
                'Symbench',
                'package.mo'))
        if not os.path.exists(filename):
            raise FileNotFoundError(
                "Library does not exists: " + filename)

        print("Loading", filename, "library:",
              self.send("loadFile(\"" + filename + "\")"))

    def instantiate_model(self, model: str) -> 'Model':
        print("Instantiating model:", model)
        result = self.send("instantiateModel(" + model + ")")
        model = Model()
        model.parse_model(result)
        return model


class Model:
    def __init__(self):
        self.model_name = None

    RE_FUNC = re.compile(
        '^(?:impure\s+)?function\s+([A-Za-z0-9_\.]+)(?:\s+"[^"]*")*$')
    RE_CLASS = re.compile('^class\s+([A-Za-z0-9_\.]+)$')
    RE_END = re.compile('^end\s+([A-Za-z0-9_\.]+);$')
    RE_EQU_START = re.compile('^equation$')

    def parse_model(self, result: str):
        current_function = None
        current_model = None
        in_equation = False
        for line in result.splitlines():
            # print(">", line)

            res = Model.RE_FUNC.match(line)
            if res:
                assert current_function is None
                current_function = res.group(1)
                continue

            res = Model.RE_CLASS.match(line)
            if res:
                assert self.model_name is None
                current_model = res.group(1)
                self.model_name = current_model
                continue

            res = Model.RE_END.match(line)
            if res:
                if res.group(1) == current_function:
                    current_function = None
                elif res.group(1) == self.model_name:
                    current_model = None
                    in_equation = False
                else:
                    raise ValueError("Unexpected end statement")
                continue

            res = Model.RE_EQU_START.match(line)
            if res:
                assert current_model is not None
                assert not in_equation
                in_equation = True
                continue

            if current_model and not in_equation:
                self.parse_component(set(), line)
            if current_model and in_equation:
                self.parse_equation(line)

    RE_SIMPLE_EQU = re.compile('^([^"=]*) = ([^"=]*);$')
    RE_ASSERT_LEQ = re.compile(
        '^\s*assert\(([^"=]*) <= ([^"=]*),\s*\"[^"]*\"\);$')
    RE_ASSERT_GEQ = re.compile(
        '^\s*assert\(([^"=]*) >= ([^"=]*),\s*\"[^"]*\"\);$')
    RE_ASSERT_SKIP = re.compile('^\s*assert\(.*,\s*\"([^"]*)\"\);$')
    ASSERT_SKIP_NAMES = [
        "tableOnFile = true and no table name given",
    ]

    def parse_equation(self, line):
        res = Model.RE_SIMPLE_EQU.match(line)
        if res:
            # print(res.group(1), '=', res.group(2))
            return

        res = Model.RE_ASSERT_LEQ.match(line)
        if res:
            # print(res.group(1), '<=', res.group(2))
            return

        res = Model.RE_ASSERT_GEQ.match(line)
        if res:
            # print(res.group(1), '>=', res.group(2))
            return

        res = Model.RE_ASSERT_SKIP.match(line)
        if res and res.group(1) in Model.ASSERT_SKIP_NAMES:
            return

        raise ValueError("Unknown line: " + line)

    RE_FLAGS_PREFIX = re.compile('^\s*(parameter)\s+(.*)$')
    RE_TYPES_PREFIX = re.compile('^\s*(Real)\s+(.*)$')

    def parse_component(self, flags: Set[str], string: str):
        res = Model.RE_FLAGS_PREFIX.match(string)
        if res:
            self.parse_component(flags.union({res.group(1)}), res.group(2))
        else:
            print(flags, string)

    def parse_real(self, string) -> sympy.Expr:
        pass

    def print(self):
        print("Model name:", self.model_name)


def run():
    session = ModelicaSession()
    session.load_library()
    model = session.instantiate_model("Symbench.FDM.Test")
    model.print()


if __name__ == '__main__':
    run()
    # run2()
