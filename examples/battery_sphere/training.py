#!/usr/bin/env python3
# Copyright (C) 2021, Symbench Szeged team
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

from collections import Counter
import csv
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.distutils.lib2def import output_def
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import os
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sympy import *
from constraint_prog.point_cloud import PointCloud


def plot_3D(x, y, z, xlabel="x", ylabel="y", zlabel="z", title="3D plot"):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x, y, z, marker='.')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()


# X is 3D
# Y is the output
def fit_poly(X, Y, deg):
    # generate a model of polynomial features
    poly_features = PolynomialFeatures(degree=deg)
    # transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
    X_ = poly_features.fit_transform(X)

    # generate the regression object
    p = linear_model.LinearRegression()
    # preform the actual regression
    p.fit(X_, Y)

    return p, poly_features


if __name__ == '__main__':
    raw_points = PointCloud.load(filename='sphere_battery_packing.csv', delimiter=';')
    pruned_points = raw_points.prune_pareto_front(directions=[-1, 0, 0, 0, 0, 1, -1])

    raw_points = raw_points.projection(["sphere_D", "total_CAP", "total_MASS"])
    pruned_points = pruned_points.projection(["sphere_D", "total_CAP", "total_MASS"])

    raw_points.float_data[:, 1] /= np.max(raw_points.float_data.numpy()[:, 1])
    raw_points.float_data[:, 2] /= np.max(raw_points.float_data.numpy()[:, 2])
    pruned_points.float_data[:, 1] /= np.max(pruned_points.float_data.numpy()[:, 1])
    pruned_points.float_data[:, 2] /= np.max(pruned_points.float_data.numpy()[:, 2])

    plot_3D(raw_points.float_data[:, 0],
            raw_points.float_data[:, 1],
            raw_points.float_data[:, 2],
            "Sphere diameter",
            "Total capacity",
            "Total mass",
            "Original point cloud")

    plot_3D(pruned_points.float_data[:, 0],
            pruned_points.float_data[:, 1],
            pruned_points.float_data[:, 2],
            "Sphere diameter",
            "Total capacity",
            "Total mass",
            "Pruned point cloud")

    diams = torch.linspace(0.0, 1.0, 40)
    cap = torch.linspace(0.0, 1.0, 40)
    masses = torch.linspace(0.0, 1.0, 40)
    mesh = torch.stack(torch.meshgrid(diams, cap, masses), dim=-1)  # 40,40,40,3
    mesh.double()
    dist = pruned_points.get_pareto_distance(directions=[-1, 1, -1], points=mesh)  # 40,40,40

    X = mesh.numpy()
    X = np.reshape(X, (-1, 3))
    Y = dist.numpy()
    Y = np.reshape(Y, (-1, 1))

    # fit a higher order multivariate polynomial
    p, poly_features = fit_poly(X, Y, 5)

    # produce sympy expression from the fitted poly
    feature_names = poly_features.get_feature_names()
    coefficients = p.coef_.tolist()
    sympy_expr = ""
    for c,f in zip(feature_names, coefficients[0]):
        if(f < 0):
            sympy_expr += str(f)+"*"+c
            #print(str(f)+"*"+c, end='')
        else:
            sympy_expr += "+"+str(f) + "*" + c
            #print("+"+str(f) + "*" + c, end='')
    #print()
    sympy_expr = sympy_expr.replace("^", "**")
    sympy_expr = sympy_expr.replace(" ", "*")
    expr = sympify(sympy_expr)
    x0, x1, x2 = symbols("x0 x1 x2")
    print("Sympy expression: ", expr)
    print("Subs in point: [0.5, 0.5, 0.5]: ",
          expr.subs( [(x0, 0.5), (x1, 0.5), (x2, 0.5)]))
    print("Evalf in point: [0.5, 0.5, 0.5]: ",
          expr.evalf(subs={x0:0.5, x1:0.5, x2:0.5}))
    x = np.asarray([0.5, 0.5, 0.5])
    x = np.reshape(x, (1, -1))
    print("The poly in this point: ", p.predict(poly_features.fit_transform(x)))

    # use the fitted polynomial
    predicted_surface = np.zeros_like(dist)
    real_surface = np.zeros_like(dist)
    for i in range(mesh.numpy().shape[0]):
        for j in range(mesh.numpy().shape[1]):
            for k in range(mesh.numpy().shape[2]):
                x = mesh[i, j, k]
                x = np.reshape(x, (1, -1))
                predicted_surface[i, j, k] = p.predict(poly_features.fit_transform(x))
                real_surface[i, j, k] = pruned_points.get_pareto_distance(directions=[-1, 1, -1], points=mesh[i, j, k])

    # checking
    selected_weigth_id = 0
    plot_3D(mesh[:, :, selected_weigth_id, 0].numpy(),
            mesh[:, :, selected_weigth_id, 1].numpy(),
            real_surface[:, :, 0],
            "Sphere diameter",
            "Total capacity",
            "Pareto distance",
            "Real surface")

    selected_weigth_id = 0
    plot_3D(mesh[:, :, selected_weigth_id, 0].numpy(),
            mesh[:, :, selected_weigth_id, 1].numpy(),
            predicted_surface[:, :, 0],
            "Sphere diameter",
            "Total capacity",
            "Pareto distance",
            "Predicted surface")
