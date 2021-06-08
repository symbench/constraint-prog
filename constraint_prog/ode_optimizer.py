#!/usr/bin/env python3
# Copyright (C) 2021, Zsolt Vizi
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

from typing import Callable

import matplotlib.pyplot as plt
import torch


class ODEOptimizer:
    def __init__(self, f: Callable, fourier_order: int, u_dim: int,
                 t_0: float, x_0: torch.Tensor, t_1: float, x_1: torch.Tensor,
                 dt: float, device: torch.device) -> None:
        self.f = f
        self.fourier_order = fourier_order
        self.u_dim = u_dim
        assert t_1 >= t_0
        self.t_0, self.t_1, self.x_0, self.x_1 = t_0, t_1, x_0, x_1
        self.dt = dt
        self.device = device

        self.x_dim = self.x_0.shape[0]
        self.n_coeff_fourier = 2 * self.fourier_order + 1
        self.t = torch.linspace(start=t_0, end=t_1, steps=int((t_1 - t_0) / self.dt),
                                requires_grad=True, device=self.device).view((-1, 1))
        self.n_t = self.t.shape[0]

    def run(self, coeff: torch.Tensor) -> torch.Tensor:
        # Reshape input coefficient tensor
        coeff_ = coeff.view((self.n_coeff_fourier, self.x_dim + self.u_dim))
        # Get Fourier polynomial for x(t) and u(t)
        x_t = self.fourier(coeff=coeff_[:, :self.x_dim])
        u_t = self.fourier(coeff=coeff_[:, self.x_dim:])

        # Get x'(t)
        dx_dt_list = []
        for idx in range(self.x_dim):
            dx_dt_list.append(
                torch.autograd.grad(inputs=self.t,
                                    outputs=x_t[:, idx],
                                    grad_outputs=torch.ones(self.t.shape[0],
                                                            device=self.device),
                                    retain_graph=True)[0]
                              )
        dx_dt = torch.stack(tensors=dx_dt_list, dim=-1).squeeze(dim=1)

        # Return with equalities in a lhs - rhs form
        return torch.cat(
            tensors=[(x_t[0] - self.x_0).flatten(),
                     (x_t[-1] - self.x_1).flatten(),
                     (dx_dt - self.f(x_t, u_t)).flatten()]
        ).unsqueeze(dim=0)

    def fourier(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        Calculates Fourier approximation using input coefficients with shape (2 * order + 1, *)
        and returns tensor with shape (t, *)
        """
        v_dim = coeff.shape[1]
        # Initialize zero tensor with shape (self.n_t, v_dim)
        result = torch.zeros(size=(self.n_t, v_dim), device=self.device)
        # Add constant term
        # Broadcasting is used: (self.n_t, v_dim) * (v_dim, ) = (self.n_t, v_dim)
        result = result + torch.ones_like(input=result, device=self.device) * coeff[0, :]

        for j in range(1, self.fourier_order + 1):
            # Broadcasting is used: (self.n_t, 1) * (v_dim, ) * (v_dim, ) = (self.n_t, v_dim)
            # Add cosine term
            result = result + \
                torch.cos(input=j * self.t) * \
                torch.ones(v_dim, device=self.device) * \
                coeff[j, :]
            # Add sine term
            result = result + \
                torch.sin(input=j * self.t) * \
                torch.ones(v_dim, device=self.device) * \
                coeff[self.fourier_order + j, :]

        return result

    def fourier_2(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        Calculates Fourier approximation using input coefficients with shape (2 * order + 1, *)
        and returns tensor with shape (t, *)
        """
        v_dim = coeff.shape[1]
        # Calculate zeroth order part
        # Broadcasting: (self.n_t, v_dim) * (1, v_dim)
        const_part = torch.ones(size=(self.n_t, v_dim), device=self.device) * coeff[0, :]
        # Get tensor of time points with shape (self.fourier_order, self.n_t, v_dim)
        # tt[i, j, k] = i * self.t[j]
        tt = \
            torch.tile(input=self.t * torch.ones(v_dim, device=self.device),
                       dims=(self.fourier_order, 1, 1)
                       ) * \
            torch.reshape(
                input=torch.arange(start=1, end=self.fourier_order + 1, device=self.device),
                shape=(self.fourier_order, 1, 1)
            )
        # Get cosine and sine part
        # Broadcasting: (self.fourier_order, self.n_t, v_dim) * (self.fourier_order, 1, v_dim)
        # Summation along dim=0: (self.fourier_order, self.n_t, v_dim) -> (self.n_t, v_dim)
        cos_part = \
            torch.sum(
                input=torch.cos(input=tt) * torch.reshape(input=coeff[1:self.fourier_order + 1, :],
                                                          shape=(self.fourier_order, 1, v_dim)),
                dim=0)
        sin_part = \
            torch.sum(
                input=torch.sin(input=tt) * torch.reshape(input=coeff[self.fourier_order + 1:, :],
                                                          shape=(self.fourier_order, 1, v_dim)),
                dim=0)
        # Get result as a sum of constant, cosine and sine parts
        result = const_part + cos_part + sin_part
        return result


def main():
    # Function generating the flow (dynamical system)
    def func(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            tensors=[-3 * x[:, 1] + u[:, 0],
                     2 * x[:, 0] + u[:, 1]],
            dim=-1
        )

    # Get variables needed to create an ODEOptimizer object
    device = torch.device('cpu')
    u_dim = 2
    fourier_order = 5
    t_0, t_1 = 0.0, 4.0
    x_0 = torch.tensor(data=[1.0, 1.0], device=device)
    x_1 = torch.tensor(data=[-1.17613, 4.21177], device=device)
    dt = 0.01

    ode_opt = ODEOptimizer(f=func, fourier_order=fourier_order, u_dim=u_dim,
                           t_0=t_0, x_0=x_0, t_1=t_1, x_1=x_1,
                           dt=dt, device=device)
    coeff = torch.randn(size=(1, ode_opt.n_coeff_fourier * (ode_opt.x_dim + ode_opt.u_dim)),
                        device=device)

    # Get solution from optimization
    # from constraint_prog.gradient_descent import gradient_descent
    # output = gradient_descent(f=ode_opt.run, in_data=coeff,
    #                           lrate=0.001, it=10000, device=device)
    from constraint_prog.newton_raphson import newton_raphson
    output = newton_raphson(func=ode_opt.run, input_data=coeff)

    coeff = output.view((ode_opt.n_coeff_fourier, ode_opt.x_dim + ode_opt.u_dim))
    x_t = ode_opt.fourier(coeff=coeff[:, :ode_opt.x_dim]).cpu().detach().numpy()
    run_output = ode_opt.run(coeff=output)
    val = (run_output.pow(2.0).sum(dim=-1)).pow(0.5)

    print("Number of points (flow):", run_output.shape[1] - 4)
    print("Squared error:", val.cpu().detach().numpy()[0])
    plt.plot(x_t[:, 0], x_t[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
