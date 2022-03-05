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

import os
import re
import sympy
import tempfile
from typing import Any, Optional, Set

import pyparsing as pp
from OMPython import OMCSessionZMQ

# https://build.openmodelica.org/Documentation/ModelicaReference.ModelicaGrammar.html
# https://pyparsing-docs.readthedocs.io/en/latest/HowToUsePyparsing.html

IDENTIFIER = pp.NotAny(pp.Keyword("end") | pp.Keyword("parameter")) \
    + pp.Word(pp.alphas, pp.alphanums + "_").setName("identifier")

NAME = pp.delimitedList(IDENTIFIER, delim=".", combine=True).setResultsName("name")

STRING_COMMENT = pp.QuotedString('"', '\\').setResultsName("comment")

MODIFICATION = pp.Forward().setResultsName("modification")

ELEMENT_MODIFICATION = NAME + pp.Optional(MODIFICATION) + pp.Optional(STRING_COMMENT)

ARGUMENT = pp.Group(ELEMENT_MODIFICATION)

CLASS_MODIFICATION = pp.Group(
    pp.Suppress("(") +
    pp.Optional(pp.delimitedList(ARGUMENT)) +
    pp.Suppress(")")).setResultsName("arguments")

EXPRESSION = (pp.dblQuotedString | NAME | pp.pyparsing_common.real).setResultsName("expression")

MODIFICATION <<= CLASS_MODIFICATION + pp.Optional('=' + EXPRESSION) | '=' + EXPRESSION

DECLARATION = NAME + pp.Optional(MODIFICATION)

TYPE_PREFIX = (
    pp.Optional(pp.Keyword("flow") | pp.Keyword("stream"))
    + pp.Optional(pp.Keyword("discrete") | pp.Keyword("parameter") | pp.Keyword("constant"))
    + pp.Optional(pp.Keyword("input") | pp.Keyword("output"))
).setResultsName("type_prefix")

TYPE_SPECIFIER = NAME("type_specifier")

COMPONENT_LIST = pp.delimitedList(
    DECLARATION + pp.Optional(STRING_COMMENT)).setResultsName("component_list")

COMPONENT_CLAUSE = TYPE_PREFIX + TYPE_SPECIFIER + COMPONENT_LIST

ELEMENT = pp.Group(COMPONENT_CLAUSE)

ELEMENT_LIST = pp.ZeroOrMore(ELEMENT + pp.Suppress(";")).setResultsName("element_list")

CLASS_PREFIXES = pp.Keyword("class") | pp.Keyword("function")

CLASS_DEFINITION = pp.Group(
    CLASS_PREFIXES + NAME + pp.Optional(STRING_COMMENT)
    + ELEMENT_LIST
    + pp.Keyword("end") + NAME("end_name"))

STORED_DEFINITION = pp.ZeroOrMore(CLASS_DEFINITION + pp.Suppress(";"))


def run3():
    print(NAME.parseString("Hihi3.Hehe.A3", parseAll=True))
    data = ELEMENT_MODIFICATION.parseString("Haha.Hehe = Hihi3 \"something\"", parseAll=True)
    print(data.modification, data.comment)
    data = "(a=b \"haha\", c(x=y)=d \"hehe\")=f"
    data = MODIFICATION.parseString(data, parseAll=True)
    print(data)
    print(len(data.arguments))
    print(data.arguments[0].comment)
    print(data.expression)
    print(data.arguments[1].arguments)

    data = ELEMENT.parseString("input Modelica.Blocks.Types.ExternalCombiTable2D tableID ;")
    print(data)

    data = """
function Modelica.Blocks.Tables.Internal.getDerTable2DValue "Derivative of interpolated 2-dim. table defined by matrix"
  input Modelica.Blocks.Types.ExternalCombiTable2D tableID;
  input Real u1;
  input Real u2;
  input Real der_u1;
  input Real der_u2;
  output Real der_y;
end Modelica.Blocks.Tables.Internal.getDerTable2DValue;

class Symbench.FDM.Test
  parameter Real body_rotation.phi0(quantity = "Angle", unit = "rad", displayUnit = "deg") = 0.0 "Fixed offset angle of housing";
  Real body_rotation.flange.phi(quantity = "Angle", unit = "rad", displayUnit = "deg") "Absolute rotation angle of flange";
  Real body_rotation.flange.tau(quantity = "Torque", unit = "N.m") "Cut torque in the flange";
  Real eletric_ground.p.v(quantity = "ElectricPotential", unit = "V") "Potential at the pin";
  Real eletric_ground.p.i(quantity = "ElectricCurrent", unit = "A") "Current flowing into the pin";
  parameter Real air_rotation.phi0(quantity = "Angle", unit = "rad", displayUnit = "deg") = 0.0 "Fixed offset angle of housing";
end Symbench.FDM.Test;
"""
    data = STORED_DEFINITION.parseString(data, parseAll=False)
    print(data)
    print(len(data))
    print(data[0])
    print(len(data[0]))
    print(data[1].name)
    print(data[1].end_name)
    print(data[1].comment)
    print(data[1].element_list)
    print(len(data[1].element_list))


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

    def save_model(self, model: str, filename: str):
        print("Instantiating model:", model)
        result = self.send("instantiateModel(" + model + ")")
        with open(filename, "w") as f:
            f.write(result)

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
    session.save_model('Symbench.FDM.Test', 'test.mo')
    # model = session.instantiate_model("Symbench.FDM.Test")
    # model.print()


if __name__ == '__main__':
    # run()
    run3()
