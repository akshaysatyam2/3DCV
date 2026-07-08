import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2) - 1.0

plt.figure()
cp = plt.contourf(X, Y, Z, levels=20, cmap='RdBu')
plt.colorbar(cp, label='Distance to surface')
plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
plt.title("Implicit Representation (Signed Distance Field)")
plt.savefig("output.png")
