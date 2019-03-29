__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np

def sine_func(x, Vc=1, m=2):
    return Vc*np.sin(m*x*np.pi)

def const(x, Vc=1):
    return Vc*np.ones(len(x))

def linear_func(x, a=4, b=-2):
    return a*x + b

def quadratic_func(x, c1=0, c2=1, Vc=1):
    return Vc*(x-c1)*(x-c2)

def exp_func(x, a=1):
    return np.exp(a*x)

def heaviside(x, Vc=1):
    return Vc*np.heaviside(x-0.5, 1)*np.heaviside(0.75 - x, 1)

def quartic_func(x, Vc=1):
    return Vc * (1- (x - 1/2)**4)