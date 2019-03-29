__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
import matplotlib.pyplot as plt

# Check out the matploblib documentation for more colormap references.
# http://matplotlib.org/examples/color/colormaps_reference.html

gradient = np.linspace(0, 1, 512)
gradient = np.vstack((gradient, gradient))

plt.figure(figsize=(10, 2))
plt.title('"Magma" colormap', fontsize=14)
plt.imshow(gradient, aspect='auto', cmap=plt.get_cmap('magma'))
plt.axis('off')
plt.savefig('magma_colormap.pdf')

plt.show()

plt.figure(figsize=(10, 2))
plt.title('"Plasma" colormap', fontsize=14)
plt.imshow(gradient, aspect='auto', cmap=plt.get_cmap('viridis'))
plt.axis('off')
plt.savefig('viridis_colormap.pdf')

plt.show()

plt.figure(figsize=(10, 2))
plt.title('"Winter" colormap', fontsize=14)
plt.imshow(gradient, aspect='auto', cmap=plt.get_cmap('winter'))
plt.axis('off')
plt.savefig('winter_colormap.pdf')

plt.show()
