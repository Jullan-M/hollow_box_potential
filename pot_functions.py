__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np

def sine_func(x, Vc=1, m=4):
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

def quartic_func(x, Vc=1):
    return Vc * (1- (x - 1/2)**4)