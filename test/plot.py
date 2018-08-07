from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

X = [0,300,400,500,700]
Y = [0,2,5,8,10]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X,Y)
ax.grid()
plt.show()