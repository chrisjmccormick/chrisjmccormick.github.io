# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:57:47 2017

@author: Chris
"""

from matplotlib import pyplot as plt

import seaborn
import numpy as np

# Not sure what this does..
seaborn.set(style='ticks')

# Generate the data points.
x = np.random.normal(scale=0.5, size=(40,2))

fig, ax = plt.subplots()

plt.scatter(x[:, 0], x[:, 1])



ax.set_aspect('equal')
#ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0) # the important part here