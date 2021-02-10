import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

X, Y, Z = axes3d.get_test_data(0.05)

print(X.shape)
print(Y.shape)
print(Z.shape)

ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()