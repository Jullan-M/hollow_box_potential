__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
import matplotlib.pyplot as plt

# Check out the matploblib documentation for more colormap references.
# http://matplotlib.org/examples/color/colormaps_reference.html

gradient = np.linspace(0, 1, 512)
gradient = np.vstack((gradient, gradient))
fig = plt.figure(figsize=(10, 2))
plt.title('"Magma" colormap', fontsize=14)
plt.imshow(gradient, aspect='auto', cmap=plt.get_cmap('magma'))
plt.axis('off')
plt.savefig('magma_colormap.pdf')

plt.show()
