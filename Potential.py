__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan (not erik lol)
import numpy as np
import threading
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Potential(threading.Thread):
    def __init__(self, V0, dq, n_max, label=""):
        threading.Thread.__init__(self)
        self.label = label
        self.dq = dq    #   dx and dy. This should be small.
        self.N = int(1/dq + 1)   #  Number of discretization points.
        self.n_max = n_max  # Upper sum boundary for Fourier coefficients. This should be large.
        self.xi = np.linspace(0, 1, self.N)

        self.prod2 = 0  # A product that will be used a lot in calculations. See calc_fCoeffs

        self.V0_space = V0(self.xi)

        self.fCoeffs = np.zeros(self.n_max)
        self.V = np.zeros((self.N, self.N))

    def run(self):
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
        plt.figure()
        plt.plot(self.xi, self.V[-1], label="Numerical " + self.label, color="r")
        plt.plot(self.xi, self.V0_space, label="Analytic potential", color="k")
        plt.xlabel(r"$\xi$", fontsize=16)
        plt.ylabel(r"$V(\xi, \eta=1)= V_0(\xi)$", fontsize=16)
        plt.legend()
        plt.grid()
        plt.savefig('num_an_' + self.label +  '.pdf')
        plt.show()

    def plot_potential_contour(self):
        fig, ax = plt.subplots()
        ax.contourf(self.xi, self.xi, self.V, 500, cmap = cm.inferno)
        plt.xlabel(r"$\xi$", fontsize=16)
        plt.ylabel(r"$\eta$", fontsize=16)
        plt.savefig('contour_' + self.label +  '.pdf')
        plt.show()

    def plot_potential_quiver(self, slice):
        fig, ax = plt.subplots()
        n_quiv = int(len(self.V)/slice + 1/slice)
        xq = np.linspace(0, 1, n_quiv)
        [B, A] = - np.array(np.gradient(self.V[::slice,::slice]))

        ax.quiver(xq, xq, A, B, A ** 2 + B ** 2)
        plt.savefig('quiver_' + self.label +  '.pdf')
        plt.show()

    def plot_fourier_convergence(self):
        V0fourier = np.zeros(self.N)
        sinh_prod2 = np.sinh(self.prod2)
        sin_prod2 = np.sin(self.prod2)

        n_space = np.arange(self.n_max) + 1

        max_diff = np.zeros(self.n_max)
        av_diff = np.zeros(self.n_max)

        for i in range(self.n_max):
            print(self.label + ": " + str(i))
            V0fourier += self.fCoeffs[i] * sinh_prod2[i][-1] * sin_prod2[i]
            max_diff[i] = np.absolute(V0fourier[1:-1] - self.V0_space[1:-1]).max()
            #av_diff[i] = np.average(np.absolute(V0fourier[1:-1] - self.V0_space[1:-1]))
        plt.plot(n_space, max_diff, color="r", label=self.label + ", max error")
        #plt.plot(n_space, av_diff, color="b", label="Average")
        plt.xlabel(r"$n$", fontsize=16)
        plt.ylabel(r"Error", fontsize=16)
        plt.legend()
        plt.grid()
        plt.savefig('error_' + self.label +  '.pdf')
        plt.show()

    def plot_potential_3d(self):
        fig = plt.figure()
        X, Y = np.meshgrid(self.xi, self.xi)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')
        ax.set_zlabel(r'Potential, $V(\xi, \eta)$')
        surf = ax.plot_surface(X, Y, self.V, label=r"Potential surface", cmap=cm.inferno)

        #   Apparently there is a bug in the 3d plot library that is resolved by the following lines of code.
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d

        plt.legend()
        ax.view_init(azim=45, elev=45)
        plt.savefig("3D_" + self.label + ".pdf")
        plt.show()