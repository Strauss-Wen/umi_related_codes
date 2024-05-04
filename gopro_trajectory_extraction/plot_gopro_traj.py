import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pdb
import sys

gopro_csv_path = sys.argv[1]
# coordinates_list_1 = np.genfromtxt(gopro_csv_path, delimiter=',')
# coordinates_list_1 = coordinates_list_1[1:,5:8]

coordinates_list_1 = np.load(gopro_csv_path)
coordinates_list_1 = coordinates_list_1[:, :3, 3]

pdb.set_trace()

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Initialize empty scatter plots
scatter1 = ax.scatter([], [], [], color='red', label='List 1')

# Initialize empty lines
line1, = ax.plot([], [], [], color='red')

# Function to update scatter plots and lines
def update(frame):
    scatter1._offsets3d = (coordinates_list_1[:frame, 0], coordinates_list_1[:frame, 1], coordinates_list_1[:frame, 2])
    line1.set_data_3d(coordinates_list_1[:frame, 0], coordinates_list_1[:frame, 1], coordinates_list_1[:frame, 2])

    return scatter1, line1

# Create animation
ani = FuncAnimation(fig, update, frames=len(coordinates_list_1), interval=20, blit=True)

ax.legend()

plt.show()
