__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
from Potential import Potential
import pot_functions as V0
import numpy as np

if (__name__ == "__main__"):
    pot1 = Potential(V0.sine_func, 0.0004, 2, label="Sinusoidal-potential")
    pot2 = Potential(V0.const, 0.0004, 226, label="Constant-potential")
    pot3 = Potential(V0.linear_func, 0.0004, 226, label="Linear-potential")
    pot4 = Potential(V0.quadratic_func, 0.0004, 226, label="Quadratic-potential")
    #pot5 = Potential(V0.exp_func, 0.0004, 226, label="Exponential-potential")
    pot6 = Potential(V0.heaviside, 0.0004, 226, label="Heaviside-potential")
    pot7 = Potential(V0.quartic_func, 0.0004, 226, label="Quartic-potential")

    pots = [pot1, pot2, pot3,
            pot4, pot6, pot7]
    print("Calculating V(x,y) (with the help of threading) based on the following V_0(x) functions:")
    for pot in pots:
        print("\t" + pot.label)
        pot.start()

    for pot in pots:
        pot.join()

    print("Calculation of V(x,y) potentials done!")

    slic = 110


    pot1.plot_potential_3d()
    #pot1.plot_V0()
    pot1.plot_potential_contour(slic)
    pot1.plot_fourier_convergence()

    pot2.plot_potential_contour(slic)
    pot2.plot_V0()
    pot2.plot_fourier_convergence()


    pot3.plot_potential_contour(slic)
    #pot3.plot_V0()
    pot3.plot_fourier_convergence()


    pot4.plot_potential_contour(slic)
    #pot4.plot_V0()

    pot4.plot_fourier_convergence()
    
    #pot5.plot_potential_contour(slic)
    #pot5.plot_V0()
    #pot5.plot_potential_quiver(slic)
    #pot5.plot_fourier_convergence()
    
    pot6.plot_potential_contour(slic)
    #pot6.plot_V0()
    pot6.plot_fourier_convergence()

    pot7.plot_potential_contour(slic)
    #pot7.plot_V0()
    pot7.plot_potential_3d()
    pot7.plot_fourier_convergence()