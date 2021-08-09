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

# X is 3D: diameter, capacity, mass
# Y is the output: Pareto-distance
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
    # read data
    raw_points = PointCloud.load(filename='sphere_battery_packing.csv', delimiter=';')
    direction_pattern = [-1, 0, 0, 0, 0, 1, -1]
    # prune points
    pruned_points = raw_points.prune_pareto_front(directions=direction_pattern)
    # project points down to 3D
    raw_points = raw_points.projection(["sphere_D", "total_CAP", "total_MASS"])
    pruned_points = pruned_points.projection(["sphere_D", "total_CAP", "total_MASS"])
    # normalize data
    max_cap = np.max(raw_points.float_data.numpy()[:, 1])
    max_mass = np.max(raw_points.float_data.numpy()[:, 2])
    raw_points.float_data[:, 1] /= max_cap
    raw_points.float_data[:, 2] /= max_mass
    pruned_points.float_data[:, 1] /= max_cap
    pruned_points.float_data[:, 2] /= max_mass
    # visualize and compare raw and pruned points
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(raw_points.float_data[:, 0],
               raw_points.float_data[:, 1],
               raw_points.float_data[:, 2], s=0.5, marker='.')
    ax.scatter(pruned_points.float_data[:, 0],
               pruned_points.float_data[:, 1],
               pruned_points.float_data[:, 2], marker='.', c="r")
    ax.set_xlabel("Sphere diameter")
    ax.set_ylabel("Total capacity")
    ax.set_zlabel("Total mass")
    ax.set_title("Battery packing problem Point cloud")
    ax.legend(["Raw", "Pruned"])
    plt.show()
    # generate a mesh for training
    mesh_gran = 20
    diams = torch.linspace(0.0, 1.0, mesh_gran)
    cap = torch.linspace(0.0, 1.0, mesh_gran)
    masses = torch.linspace(0.0, 1.0, mesh_gran)
    mesh = torch.stack(torch.meshgrid(diams, cap, masses), dim=-1)  # 40,40,40,3
    mesh.double()
    # set optimization directions
    direction_pattern = [-1, 1, -1]
    dist = pruned_points.get_pareto_distance(directions=direction_pattern, points=mesh)  # 40,40,40

    # prepare training data
    Y = dist.numpy()
    Y = np.reshape(Y, (-1, 1))

    X = mesh.numpy()
    X = np.reshape(X, (-1, 3))
    # only positive distances are kept
    X = X[np.where(Y>0)[0]]
    Y = Y[np.where(Y>0)[0]]


    # fit a higher order multivariate polynomial
    poly_deg = 5
    p, poly_features = fit_poly(X, Y, poly_deg)

    # produce sympy expression from the fitted poly
    feature_names = poly_features.get_feature_names()
    #print("features: ", feature_names)
    coefficients = p.coef_.tolist()
    #print("Coeffs: ", coefficients)
    sympy_expr = str(float(p.intercept_))
    for c, f in zip(feature_names, coefficients[0]):
        if (f < 0):
            sympy_expr += str(f) + "*" + c
            # print(str(f)+"*"+c, end='')
        else:
            sympy_expr += "+" + str(f) + "*" + c
            # print("+"+str(f) + "*" + c, end='')
    # print()
    sympy_expr = sympy_expr.replace("^", "**")
    sympy_expr = sympy_expr.replace(" ", "*")
    expr = sympify(sympy_expr)
    x0, x1, x2 = symbols("x0 x1 x2")
    print("Sympy expression: ", expr)
    print("Evalf in point: [0.5, 0.5, 0.5]: ",
          expr.evalf(subs={x0: 0.5, x1: 0.5, x2: 0.5}))
    x = np.asarray([0.5, 0.5, 0.5])
    x = np.reshape(x, (1, -1))
    x = poly_features.fit_transform(x)
    print("The poly in this point: ", float(p.predict(x)[0]))

    # compare results
    predicted_surface = np.zeros_like(dist)
    real_surface = np.zeros_like(dist)
    sympy_surface = np.zeros_like(dist)
    for i in range(mesh.numpy().shape[0]):
        for j in range(mesh.numpy().shape[1]):
            for k in range(mesh.numpy().shape[2]):
                x = mesh[i, j, k]
                x = np.reshape(x, (1, -1))
                predicted_surface[i, j, k] = p.predict(poly_features.fit_transform(x))
                real_surface[i, j, k] = pruned_points.get_pareto_distance(directions=direction_pattern, points=mesh[i, j, k])
                sympy_surface[i, j, k] = max(expr.evalf(subs={x0: float(x[0][0]),
                                                          x1: float(x[0][1]),
                                                          x2: float(x[0][2])}), 0)
    # checking
    selected_weigth_id = 4
    X = mesh[:, :, selected_weigth_id, 0].numpy()
    Y = mesh[:, :, selected_weigth_id, 1].numpy()
    plot_3D(X,
            Y,
            real_surface[:, :, selected_weigth_id],
            "Sphere diameter",
            "Total capacity",
            "Pareto distance",
            "Real surface")
    plot_3D(X,
            Y,
            predicted_surface[:, :, selected_weigth_id],
            "Sphere diameter",
            "Total capacity",
            "Pareto distance",
            "Predicted surface")
    plot_3D(X,
            Y,
            sympy_surface[:, :, selected_weigth_id],
            "Sphere diameter",
            "Total capacity",
            "Pareto distance",
            "Sympy surface")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(X,
               Y,
               real_surface[:, :, selected_weigth_id], marker='.', c="b")
    ax.scatter(X,
               Y,
               sympy_surface[:, :, selected_weigth_id], marker='+', c="r")
    ax.set_xlabel("Sphere diameter")
    ax.set_ylabel("Total capacity")
    ax.set_zlabel("Pareto distance")
    ax.set_title("Real VS Sympy pareto fronts" +
                 " (mass:" +
                 str(float(masses[selected_weigth_id]) * max_mass) + ")")
    ax.legend(["Real", "Sympy"])
    plt.show()