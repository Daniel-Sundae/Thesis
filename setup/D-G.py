import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def sample_discrete_gaussian(sigma, size=1):
    """Sample from the discrete Gaussian distribution."""
    x = np.arange(-3*sigma, 3*sigma+1)
    pmf = np.exp(-x**2 / (2 * sigma**2))
    pmf /= pmf.sum()
    return np.random.choice(x, size=size, p=pmf)

# Set parameters
n_samples = 1000000
alpha = 100
sigma = alpha / math.sqrt(2 * math.pi)

# Generate discrete Gaussian samples in Z^2
samples = np.array([sample_discrete_gaussian(sigma, size=n_samples) for _ in range(2)]).T

# Compute 2D histogram
H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=(30, 30))

# Get x, y, z for non-zero counts
x, y = np.nonzero(H)
z = H[x, y]
x = xedges[x]
y = yedges[y]

# Create a new 3D subplot
fig = plt.figure(figsize=(8, 6))  # Define the figure size
ax = fig.add_subplot(111, projection='3d')

# Plot a dot at the top of each line
scatter = ax.scatter(x, y, z, color='black', s=6, alpha=1)

# Set labels with larger font size
ax.set_zlabel('Frequency', fontsize=14)

# Set grid color to white and linestyle to dotted
ax.grid(color='white', linestyle=':', linewidth=0.75)

# Set the background color of the 3d plot to be transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Make the grid lines transparent
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.grid(True)
plt.tight_layout()
plt.show()
