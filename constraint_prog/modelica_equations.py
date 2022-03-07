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
from torch import exp

# https://doc.modelica.org/Modelica%203.2.3/Resources/helpWSM/ModelicaReference/ModelicaReference.ModelicaGrammar.html
# https://pyparsing-docs.readthedocs.io/en/latest/HowToUsePyparsing.html

KEYWORD = pp.Keyword("end") | pp.Keyword("function") | pp.Keyword("class") \
    | pp.Keyword("model") | pp.Keyword("if") | pp.Keyword("else") \
    | pp.Keyword("elseif") | pp.Keyword("then") | pp.Keyword("initial") \
    | pp.Keyword("algorithm") | pp.Keyword("equation") | pp.Keyword("true") \
    | pp.Keyword("false") | pp.Keyword("not") | pp.Keyword("and") \
    | pp.Keyword("or") | pp.Keyword("public") | pp.Keyword("protected") \
    | pp.Keyword("flow") | pp.Keyword("constanr") | pp.Keyword("input") \
    | pp.Keyword("output") | pp.Keyword("break") | pp.Keyword("return") \
    | pp.Keyword("partial") | pp.Keyword("final") | pp.Keyword("block") \
    | pp.Keyword("record") | pp.Keyword("connector") | pp.Keyword("package") \
    | pp.Keyword("type") | pp.Keyword("enumeration") | pp.Keyword("pure") \
    | pp.Keyword("impure") | pp.Keyword("operator")

# partial
IDENT = pp.NotAny(KEYWORD) + \
    pp.Word(pp.alphas, pp.alphanums + "_").setName("identifier")

# complete
NAME = pp.delimitedList(IDENT, delim=".", combine=True).setResultsName("name")

# partial
STRING_COMMENT = pp.Optional(pp.QuotedString(
    '"', '\\')).setResultsName("comment")

# new
MULTIPLE_STRING_COMMENT = pp.ZeroOrMore(
    pp.QuotedString('"', '\\')).setResultsName("comment")

# partial
COMMENT = STRING_COMMENT

MODIFICATION = pp.Forward().setResultsName("modification")

ELEMENT_MODIFICATION = NAME + \
    pp.Optional(MODIFICATION) + pp.Optional(STRING_COMMENT)

ARGUMENT = pp.Group(ELEMENT_MODIFICATION)

CLASS_MODIFICATION = pp.Group(
    pp.Suppress("(") +
    pp.Optional(pp.delimitedList(ARGUMENT)) +
    pp.Suppress(")")).setResultsName("arguments")

# partial
COMPONENT_REFERENCE = NAME

# forward
EXPRESSION = pp.Forward()

# partial and modified
FUNCTION_ARGUMENTS = pp.delimitedList(EXPRESSION)

# complete
FUNCTION_CALL_ARGS = "(" + pp.Optional(FUNCTION_ARGUMENTS) + ")"

# complete
OUTPUT_EXPRESSION_LIST = pp.delimitedList(EXPRESSION)

# partial
PRIMARY = pp.dblQuotedString() \
    | pp.pyparsing_common.number \
    | pp.Keyword("true") \
    | pp.Keyword("false") \
    | (COMPONENT_REFERENCE + FUNCTION_CALL_ARGS) \
    | COMPONENT_REFERENCE \
    | "(" + OUTPUT_EXPRESSION_LIST + ")" \
    | "{}"

# complete
FACTOR = PRIMARY + pp.Optional(pp.oneOf(("^", ".^")) + PRIMARY)

# complete
MUL_OPERATOR = pp.oneOf(("*", "/", ".*", "./"))

# complete
TERM = FACTOR + pp.ZeroOrMore(MUL_OPERATOR + FACTOR)

# complete
ADD_OPERATOR = pp.oneOf(("+", "-", ".+", ".-"))

# complete
ARITHMETIC_EXPRESSION = pp.Optional(
    ADD_OPERATOR) + TERM + pp.ZeroOrMore(ADD_OPERATOR + TERM)

# complete
RELATIONAL_OPERATOR = pp.oneOf(("<", "<=", ">", ">=", "==", "<>"))

# complete
RELATION = ARITHMETIC_EXPRESSION + \
    pp.Optional(RELATIONAL_OPERATOR + ARITHMETIC_EXPRESSION)

# complete
LOGICAL_FACTOR = pp.Optional(pp.Keyword("not")) + RELATION

# complete
LOGICAL_TERM = LOGICAL_FACTOR + \
    pp.ZeroOrMore(pp.Keyword("and") + LOGICAL_FACTOR)

# complete
LOGICAL_EXPRESSION = LOGICAL_TERM + \
    pp.ZeroOrMore(pp.Keyword("or") + LOGICAL_TERM)

# complete
SIMPLE_EXPRESSION = LOGICAL_EXPRESSION + \
    pp.Optional(":" + LOGICAL_EXPRESSION +
                pp.Optional(":" + LOGICAL_EXPRESSION))

# complete
EXPRESSION <<= (
    SIMPLE_EXPRESSION |
    pp.Keyword("if") + EXPRESSION + pp.Keyword("then") + EXPRESSION
    + pp.ZeroOrMore(pp.Keyword("elseif") +
                    EXPRESSION + pp.Keyword("then") + EXPRESSION)
    + pp.Keyword("else") + EXPRESSION
)

# forward declaration
STATEMENT = pp.Forward()

# complete
IF_STATEMENT = pp.Keyword("if") + EXPRESSION \
    + pp.Keyword("then") + pp.ZeroOrMore(STATEMENT + ";") \
    + pp.ZeroOrMore(pp.Keyword("elseif") + EXPRESSION +
                    pp.Keyword("then") + pp.ZeroOrMore(STATEMENT + ";")) \
    + pp.Optional(pp.Keyword("else") + pp.ZeroOrMore(STATEMENT + ";")) \
    + pp.Keyword("end") + pp.Keyword("if")

# incomplete
STATEMENT <<= pp.Keyword("break") \
    | pp.Keyword("return") \
    | IF_STATEMENT \
    | COMPONENT_REFERENCE + ":=" + EXPRESSION

# partial
EQUATION = (
    SIMPLE_EXPRESSION + "=" + EXPRESSION
    | NAME + FUNCTION_CALL_ARGS
) + COMMENT

MODIFICATION <<= CLASS_MODIFICATION + \
    pp.Optional('=' + EXPRESSION) | '=' + EXPRESSION

DECLARATION = NAME + pp.Optional(MODIFICATION)

# complete
SUBSCRIPT = ":" | EXPRESSION

# complete
ARRAY_SUBSCRIPTS = "[" + pp.delimitedList(SUBSCRIPT) + "]"

# modified
TYPE_PREFIX = pp.Group(
    pp.Optional(pp.Keyword("public") | pp.Keyword("protected"))
    + pp.Optional(pp.Keyword("flow") | pp.Keyword("stream"))
    + pp.Optional(pp.Keyword("discrete") |
                  pp.Keyword("parameter") | pp.Keyword("constant"))
    + pp.Optional(pp.Keyword("input") | pp.Keyword("output"))
).setResultsName("type_prefix")

# complete new
ENUMERATION_TYPE = pp.Keyword("enumeration") + \
    "(" + pp.delimitedList(NAME) + ")"

# partial and modified
TYPE_SPECIFIER = (ENUMERATION_TYPE | NAME).setResultsName("type_specifier")

# partial
COMPONENT_DECLARATION = DECLARATION + COMMENT

# complete
COMPONENT_LIST = pp.delimitedList(
    COMPONENT_DECLARATION).setResultsName("component_list")

# complete
COMPONENT_CLAUSE = TYPE_PREFIX + TYPE_SPECIFIER + \
    pp.Optional(ARRAY_SUBSCRIPTS) + COMPONENT_LIST

# partial
ELEMENT = pp.Group(pp.Optional(pp.Keyword("final")) + COMPONENT_CLAUSE)

# complete
ELEMENT_LIST = pp.ZeroOrMore(
    ELEMENT + pp.Suppress(";")).setResultsName("element_list")

# complete
EXPRESSION_LIST = pp.delimitedList(EXPRESSION)

# complete
EXTERNAL_FUNCTION_CALL = pp.Optional(COMPONENT_REFERENCE + "=") \
    + IDENT + "(" + pp.Optional(EXPRESSION_LIST) + ")"

# conplete
ALGORITHM_SECTION = pp.Optional(pp.Keyword("initial")) \
    + pp.Keyword("algorithm") + pp.ZeroOrMore(STATEMENT + ";")

EQUATION_SECTION = pp.Optional(pp.Keyword("initial")) \
    + pp.Keyword("equation") + pp.ZeroOrMore(EQUATION + ";")

# complete
LANGUAGE_SPECIFICATION = pp.dblQuotedString("language_sepcification")

# partial and new
EXTERNAL_SECTION = pp.Group(
    pp.Keyword("external") + pp.Optional(LANGUAGE_SPECIFICATION)
    + pp.Optional(EXTERNAL_FUNCTION_CALL) + ";"
)

# partial
COMPOSITION = ELEMENT_LIST \
    + pp.ZeroOrMore(ALGORITHM_SECTION) \
    + pp.ZeroOrMore(EQUATION_SECTION) \
    + pp.Optional(EXTERNAL_SECTION)

# partial and modified
LONG_CLASS_SPECIFIER = NAME + MULTIPLE_STRING_COMMENT + \
    COMPOSITION + pp.Keyword("end") + NAME

# partial
CLASS_SPECIFIER = LONG_CLASS_SPECIFIER

# complete
CLASS_PREFIXES = pp.Group(pp.Optional(pp.Keyword("partial")) + (
    pp.Keyword("class") |
    pp.Keyword("model") |
    pp.Optional(pp.Keyword("operator")) + pp.Keyword("record") |
    pp.Keyword("block") |
    pp.Optional(pp.Keyword("expendable")) + pp.Keyword("connector") |
    pp.Keyword("type") |
    pp.Keyword("package") |
    pp.Optional(pp.Keyword("pure") | pp.Keyword("impure")) + pp.Optional("operator") + pp.Keyword("function") |
    pp.Keyword("operator")
))

# complete
CLASS_DEFINITION = pp.Optional(pp.Keyword("encapsulated")) \
    + CLASS_PREFIXES + CLASS_SPECIFIER

# complete
STORED_DEFINITION = pp.Optional(pp.Keyword("within") + pp.Optional(NAME) + ";") \
    + pp.ZeroOrMore(pp.Optional(pp.Keyword("final")) + CLASS_DEFINITION + ";")


def run2():
    with open('test.mo', 'r') as f:
        data = STORED_DEFINITION.parseString(f.read())
        print(data)


def run3():
    data = """
  parameter Real battery.constantVoltage.V(quantity = "ElectricPotential", unit = "V", start = 1.0) = battery.V "Value of constant voltage";
    """
    data = COMPONENT_CLAUSE.parseString(data)
    print(data)


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
    run2()
    # run3()
