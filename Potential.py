__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan (not erik lol)
import numpy as np
from matplotlib import pyplot as plt

class Potential:
    def __init__(self, V0, dq, n_max):
        self.dq = dq    #   dx and dy. This should be small.
        self.N = int(1/dq + 1)   #  Number of discretization points.
        self.n_max = n_max  # Upper sum boundary for Fourier coefficients. This should be large.
        self.xi = np.linspace(0, 1, self.N)

        self.prod2 = 0  # A product that will be used a lot in calculations. See calc_fCoeffs

        self.V0_space = V0(self.xi)

        self.fCoeffs = np.zeros(self.n_max)
        self.V = np.zeros((self.N, self.N))

    def calc_fCoeffs(self):
        #   1D matrices
        n_space = np.arange(self.n_max) + 1
        prod1 = np.pi * n_space

        #   2D matrices - PRODUCT NUMBER 2
        self.prod2 = prod1.reshape(-1,1) * self.xi

        integrands = self.V0_space * np.sin(self.prod2)
        self.fCoeffs = 2 / np.sinh(prod1) * np.trapz(integrands, dx = self.dq)

    def calc_potential(self):
        sinh_prod2 = np.sinh(self.prod2)
        sin_prod2 = np.sin(self.prod2)

        for i in range(self.n_max):
            #   TODO: Make an equivalent sum function that utilizes numpy more efficiently.
            self.V +=  self.fCoeffs[i] * sin_prod2[i].reshape(-1, 1) * sinh_prod2[i]
        self.V = self.V.T

    def plot_potential_contour(self):
        fig, ax = plt.subplots()
        ax.contourf(self.xi, self.xi, self.V, 500)
        #plt.savefig('figqontour')
        plt.show()

    #   TODO: Make a function that plots V(x,y) in 3D.

def sine_func(x, Vc=1, m=2):
    return Vc*np.sin(m*x*np.pi)

def const(x, Vc=1):
    return np.zeros(len(x))

def e(x):
    return np.exp(x)

pot1 = Potential(const, 0.01, 100)
pot1.calc_fCoeffs()
pot1.calc_potential()
pot1.plot_potential()