__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
import threading
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import integrate


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
        self.fCoeffs = 2 / np.sinh(prod1) * integrate.simps(integrands, dx = self.dq)

    def calc_potential(self):
        sinh_prod2 = np.sinh(self.prod2)
        sin_prod2 = np.sin(self.prod2)

        for i in range(self.n_max):
            #   TODO: Make an equivalent sum function that utilizes numpy more efficiently.
            self.V +=  self.fCoeffs[i] * sinh_prod2[i].reshape(-1, 1) * sin_prod2[i]

    def plot_V0(self):
        plt.figure()
        plt.plot(self.xi, self.V0_space, label=r"Analytic $V_0(\xi)$", color="k", linewidth=2)
        plt.plot(self.xi, self.V[-1], label=r"Numerical $V(\xi, \eta=1)$", color="r",linewidth=1)
        #plt.plot(self.xi, self.V[0], label=r"Numerical $V(\xi, \eta=0)$", color="b", linewidth=2)
        #plt.plot(self.xi, self.V[:,0], label=r"Numerical $V(\xi=0, \eta)$", color="g",linewidth=1.25)
        #plt.plot(self.xi, self.V[:,-1], label=r"Numerical $V(\xi=1, \eta)$", color="y", linewidth=0.75)
        plt.xlabel(r"$\xi, \eta$", fontsize=16)
        plt.ylabel(r"$V(\xi, \eta)$", fontsize=16)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('num_an_' + self.label +  '.pdf')
        plt.show()

    def plot_potential_contour(self, slice):
        [B, A] = - np.asarray(np.gradient(self.V))

        A, B = A[::slice, ::slice], B[::slice, ::slice]

        fig, ax = plt.subplots(figsize=(9,6))

        cs1 = ax.contourf(self.xi, self.xi, self.V, 65, cmap = cm.magma)
        cs2 = ax.quiver(self.xi[::slice], self.xi[::slice], A, B, np.sqrt(A ** 2 + B ** 2), cmap = cm.winter, units='xy')
        cbar1 = fig.colorbar(cs1, ax=ax, shrink=0.9)
        cbar1.ax.set_ylabel(r'Potential, $V(\xi,\eta)$', fontsize=14)
        cbar2 = fig.colorbar(cs2, ax=ax, shrink=0.9)
        cbar2.ax.set_ylabel(r'Electric field, $|\vec{E}|$', fontsize=14)
        plt.xlabel(r"$\xi$", fontsize=16)
        plt.ylabel(r"$\eta$", fontsize=16)
        plt.tight_layout()
        plt.savefig('contour_' + self.label +  '.pdf')
        plt.show()

    # The following lines of code are deprecated.
    """
    def plot_potential_quiver(self, slice):
        fig, ax = plt.subplots()
        [B, A] = - np.array(np.gradient(self.V))

        A, B = A[::slice,::slice], B[::slice,::slice]

        cs = ax.quiver(self.xi[::slice], self.xi[::slice], A, B, np.sqrt(A ** 2 + B ** 2), cmap = cm.viridis, units='xy')
        cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
        cbar.ax.set_ylabel(r'$|\ vec{E}|$', fontsize=16)

        plt.xlabel(r"$\ xi$", fontsize=16)
        plt.ylabel(r"$\eta$", fontsize=16)
        plt.savefig('quiver_' + self.label +  '.pdf')
        plt.show()
    """

    def plot_fourier_convergence(self):
        V0fourier = np.zeros(self.N)
        sinh_prod2 = np.sinh(self.prod2)
        sin_prod2 = np.sin(self.prod2)

        n_space = np.arange(self.n_max) + 1

        max_diff = np.zeros(self.n_max)
        av_diff = np.zeros(self.n_max)

        plt.figure()
        plt.plot(self.xi, self.V0_space, label=r"Analytic $V_0(\xi)$", color="k", linewidth=2)
        for i in range(self.n_max):
            print(self.label + ": " + str(i))
            V0fourier += self.fCoeffs[i] * sinh_prod2[i][-1] * sin_prod2[i]
            max_diff[i] = np.absolute(V0fourier[1:-1] - self.V0_space[1:-1]).max()
            av_diff[i] = np.average(np.absolute(V0fourier[1:-1] - self.V0_space[1:-1]))
            if (i == 5 or i==50):
                plt.plot(self.xi, V0fourier, label=r"$V(\xi, \eta=1),\, n=$"+ str(i), linewidth=0.75)

        plt.plot(self.xi, self.V[-1], label=r"$V(\xi, \eta=1),\, n=$"+ str(self.n_max), linewidth=0.75)
        # plt.plot(self.xi, self.V[0], label=r"Numerical $V(\xi, \eta=0)$", color="b", linewidth=2)
        # plt.plot(self.xi, self.V[:,0], label=r"Numerical $V(\xi=0, \eta)$", color="g",linewidth=1.25)
        # plt.plot(self.xi, self.V[:,-1], label=r"Numerical $V(\xi=1, \eta)$", color="y", linewidth=0.75)
        plt.xlabel(r"$\xi$", fontsize=16)
        plt.ylabel(r"$V(\xi, \eta = 1)$", fontsize=16)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('V0_' + self.label + '.pdf')
        plt.show()



        plt.figure()
        plt.plot(n_space, max_diff, color="r", label="Max error")
        plt.plot(n_space, av_diff, color="b", label="Average error")
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
        surf = ax.plot_surface(X, Y, self.V, cmap=cm.inferno)

        #   Apparently there is a bug in the 3d plot library that is resolved by the following lines of code.
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d

        ax.view_init(azim=-45, elev=45)
        plt.tight_layout()
        plt.savefig("3D_" + self.label + ".pdf")
        plt.show()