__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan (not erik lol)
import numpy as np
from matplotlib import pyplot as plt

class Potential:
    def __init__(self, V0, dq, n_max, label=""):
        self.label = label
        self.dq = dq    #   dx and dy. This should be small.
        self.N = int(1/dq + 1)   #  Number of discretization points.
        self.n_max = n_max  # Upper sum boundary for Fourier coefficients. This should be large.
        self.xi = np.linspace(0, 1, self.N)

        self.prod2 = 0  # A product that will be used a lot in calculations. See calc_fCoeffs

        self.V0_space = V0(self.xi)

        self.fCoeffs = np.zeros(self.n_max)
        self.V = np.zeros((self.N, self.N))
        self.calc_fCoeffs()
        self.calc_potential()

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
            self.V +=  self.fCoeffs[i] * sinh_prod2[i].reshape(-1, 1) * sin_prod2[i]

    def plot_V0(self):
        plt.plot(self.xi, self.V[-1], label="Numerical potential", color="r")
        plt.plot(self.xi, self.V0_space, label="Analytic potential", color="k")
        plt.xlabel(r"$\xi$", fontsize=16)
        plt.ylabel(r"$V(\xi, \eta=1)= V_0(\xi)$", fontsize=16)
        plt.legend()
        plt.grid()
        #plt.savefig('fourier')
        plt.show()

    def plot_potential_contour(self):
        fig, ax = plt.subplots()
        ax.contourf(self.xi, self.xi, self.V, 500)
        plt.xlabel(r"$\xi$", fontsize=16)
        plt.ylabel(r"$\eta$", fontsize=16)
        #plt.savefig('figqontour')
        plt.show()

    def plot_potential_quiver(self, slice):
        fig, ax = plt.subplots()
        n_quiv = int(len(self.V)/slice + 1/slice)
        xq = np.linspace(0, 1, n_quiv)
        [B, A] = - np.array(np.gradient(self.V[::slice,::slice]))

        ax.quiver(xq, xq, A, B, A ** 2 + B ** 2)
        #plt.savefig('figquiver')
        plt.show()

    def plot_fourier_convergence(self):
        self.V = np.zeros((self.N, self.N))
        sinh_prod2 = np.sinh(self.prod2)
        sin_prod2 = np.sin(self.prod2)

        n_space = np.arange(self.n_max) + 1
        diff = np.zeros(self.n_max)

        for i in range(self.n_max):
            self.V += self.fCoeffs[i] * sinh_prod2[i].reshape(-1, 1) * sin_prod2[i]
            diff[i] = np.absolute(self.V[-1][0:-1] - self.V0_space[0:-1]).max()
            print(diff[i], ", n =", i)

        plt.plot(n_space, diff, color="k")
        plt.xlabel(r"$n$", fontsize=16)
        plt.ylabel(r"Error", fontsize=16)
        plt.legend()
        plt.grid()
        # plt.savefig('fourier')
        plt.show()

#   TODO: Make a function that plots V(x,y) in 3D.

def sine_func(x, Vc=1, m=2):
    return Vc*np.sin(m*x*np.pi)

def const(x, Vc=1):
    return Vc*np.ones(len(x))

def linear_func(x, a=4, b=0):
    return a*x + b

def quadratic_func(x, c=0.5):
    return 2*(x-c)**2

def exp_func(x):
    return np.exp(x)

def heaviside(x, Vc=1):
    return Vc*np.heaviside(x-0.5, 1)*np.heaviside(0.75 - x, 1)

pot1 = Potential(sine_func, 0.005, 226)
pot2 = Potential(const, 0.005, 226)
pot3 = Potential(linear_func, 0.005, 226)
pot4 = Potential(quadratic_func, 0.005, 226)
pot5 = Potential(exp_func, 0.005, 226)
pot6 = Potential(heaviside, 0.005, 226)

slic = 3
pot1.plot_potential_contour()
pot1.plot_potential_quiver(slic)
pot1.plot_fourier_convergence()

pot2.plot_potential_contour()
pot2.plot_potential_quiver(slic)
pot2.plot_fourier_convergence()


pot3.plot_potential_contour()
pot3.plot_potential_quiver(slic)
pot3.plot_fourier_convergence()

pot4.plot_potential_contour()
pot4.plot_potential_quiver(slic)
pot4.plot_fourier_convergence()

pot5.plot_potential_contour()
pot5.plot_potential_quiver(slic)
pot5.plot_fourier_convergence()

pot6.plot_potential_contour()
pot6.plot_V0()
pot6.plot_potential_quiver(slic)
pot6.plot_fourier_convergence()