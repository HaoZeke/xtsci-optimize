import numpy as np
import matplotlib.pyplot as plt
# Load data
data = np.load('rosenbrock.npz')
X, Y, Z = data['X'], data['Y'], data['Z']

# Set up logarithmically spaced levels to capture wide range of values
levels = np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 50)

# Plot contour
contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
plt.colorbar(contour)

# Labels and title
plt.title('Rosenbrock Function')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
