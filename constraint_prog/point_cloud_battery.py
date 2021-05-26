#!/usr/bin/env python3
# Copyright (C) 2021, Miklos Maroti, Zsolt Vizi
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
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from constraint_prog.point_cloud import PointCloud

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x1 = F.relu(self.hidden(x))     # activation function for hidden layer
        x2 = F.relu(self.hidden2(x1))
        x3 = F.relu(self.hidden3(x2))
        x = self.predict(x3)             # linear output
        return x

if __name__ == '__main__':

    points = PointCloud.load(filename='..\\notebooks\\battery_packing.csv', delimiter=';')
    points = points.prune_pareto_front(directions=[-1, -1, 0, 0, 0, 0, 1, 0])
    points.print_info()
    points = points.projection(["hull_D", "hull_L", "total_CAP"])
    points.print_info()

    xpos = torch.linspace(0.0, 1.0, 50)
    ypos = torch.linspace(0.0, 1.0, 50)
    cap = torch.linspace(5000.0, 5000.0, 50)
    mesh = torch.stack(torch.meshgrid(xpos, ypos, cap), dim=-1) #50,50,50,3

    # dist = points.get_pareto_distance(directions=[-1, -1, 1], points=mesh) #50,50,50
    # fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    # ax1.plot_surface(
    #     mesh[:, :, 1, 0].numpy(),
    #     mesh[:, :, 1, 1].numpy(),
    #     dist[:,:,40].numpy())
    # plt.show()

    net = Net(n_feature=3, n_hidden=100, n_output=1)  # define the network
    net.double()
    # print(net)  # net architecture
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss()

    EPOCH = 1000000

    # start training
    for epoch in range(EPOCH):
        d_r = 2.0*np.random.rand()
        l_r = 2.0*np.random.rand()
        cap_r = np.random.uniform(0,10000)
        input = Variable(torch.tensor(np.array([d_r,l_r,cap_r])))
        input2 = Variable(torch.tensor(np.array([d_r, l_r, cap_r/1000])))
        output = Variable(torch.tensor(np.array([points.get_pareto_distance(directions=[-1, -1, 1], points=input)])))
        prediction = net(input2)  # input x and predict based on x
        #prediction = torch.maximum(prediction, torch.tensor(0.0))

        loss = loss_func(prediction, output)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if epoch % (EPOCH//10) == 0:
            print(loss)


    dist = points.get_pareto_distance(directions=[-1, -1, 1], points=mesh) #50,50,50
    fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    ax1.plot_surface(
        mesh[:, :, 1, 0].numpy(),
        mesh[:, :, 1, 1].numpy(),
        dist[:,:,40].numpy())
    plt.show()

    predicted_surface = np.zeros_like(dist)
    for i in range(mesh.numpy().shape[0]):
        for j in range(mesh.numpy().shape[1]):
            input = Variable(torch.tensor(np.array([mesh[i, j, 1, 0].numpy(),mesh[i, j, 1, 1].numpy(),5.0])))
            prediction = net(input)
            predicted_surface[i,j,0] = prediction

    fig, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    ax2.plot_surface(
        mesh[:, :, 1, 0].numpy(),
        mesh[:, :, 1, 1].numpy(),
        predicted_surface[:,:,0])
    plt.show()

