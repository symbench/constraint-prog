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
from numpy.distutils.lib2def import output_def
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import os
import sys

from constraint_prog.point_cloud import PointCloud

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x1 = torch.tanh(self.hidden(x))     # activation function for hidden layer
        x2 = torch.tanh(self.hidden2(x1))
        x3 = torch.tanh(self.hidden3(x2))
        x = self.predict(x3)             # linear output
        return x

if __name__ == '__main__':

    points = PointCloud.load(filename=os.path.join('..', 'notebooks', 'battery_packing.csv'), delimiter=';')
    points = points.prune_pareto_front(directions=[-1, -1, 0, 0, 0, 0, 1, 0])
    points.print_info()
    points = points.projection(["hull_D", "hull_L", "total_CAP"])
    points.print_info()

    xpos = torch.linspace(0.0, 1.0, 50)
    ypos = torch.linspace(0.0, 1.0, 50)
    cap = torch.linspace(5000.0, 5000.0, 50)
    mesh = torch.stack(torch.meshgrid(xpos, ypos, cap), dim=-1) #50,50,50,3

    net = torch.nn.Sequential(
        torch.nn.Linear(3, 20),
        torch.nn.Sigmoid(),
        torch.nn.Linear(20, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 1),
    )

    #net = Net(n_feature=3, n_hidden=100, n_output=1)  # define the network
    net.double()
    print(net)  # net architecture
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    #optimizer = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss()

    if(len(sys.argv)>1):
        EPOCH = int(sys.argv[1])
        BATCH_SIZE = int(sys.argv[2])
        N = int(sys.argv[3])
    else:
        EPOCH = 100
        BATCH_SIZE = 128
        N = BATCH_SIZE*10

    d_r = np.random.rand(N)
    l_r = np.random.rand(N)
    cap_r = np.random.uniform(0, 10000, N)
    input = Variable(torch.tensor(np.array([d_r, l_r, cap_r]).T))
    input2 = Variable(torch.tensor(np.array([d_r, l_r, cap_r / 1000]).T))
    output = points.get_pareto_distance(directions=[-1, -1, 1], points=input)
    output = torch.unsqueeze(output, 1)
    output = Variable(output)

    torch_dataset = torch.utils.data.TensorDataset(input2, output)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, )

    # start training
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step

            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            prediction = net(b_x)  # input x and predict based on x
            # prediction = torch.maximum(prediction, torch.tensor(0.0))

            loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        if epoch % 10 == 0:
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

