"""
Core Semantic Color Math Visualizations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Choose a pathway to get 2D visualizations of the 11 La Matriz Consulting
color sequences and their momentums.
'''

import numpy as np
import matplotlib.pyplot as plt

def calculate_2d_wave_interpolation(x_coords, y_coords, t_coords):
    """
    Calculate 2D wave momentum from (x, y) coordinates and corresponding time
    values using linear interpolation.
    """
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    t_coords = np.array(t_coords)

    direction = np.array([x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]])
    direction_norm = np.linalg.norm(direction)
    if direction_norm != 0:
        direction = direction / direction_norm

    wavelength = np.linalg.norm([x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]])
    k_mag = (2 * np.pi / wavelength)
    k = k_mag * direction

    momentums = []
    for x_coord, y_coord, t_coord in zip(x_coords, y_coords, t_coords):
        phase = np.dot([x_coord, y_coord], direction) - np.dot([x_coord, y_coord], direction) / wavelength * t_coord
        momentum = k * np.exp(1j * phase)
        momentums.append(momentum)

    return np.array(momentums)

# Pathways mapped to color sequences
pathways = {
    'plot': ['grey', 'pink', 'gold', 'nude', 'orange'],
    'knot': ['white', 'blue', 'green', 'red', 'black', 'brown', 'yellow', 'purple'],
    'pain': ['gold', 'orange'],
    'practical': ['yellow', 'green'],
    'spiritual': ['blue', 'brown'],
    'prayer': ['nude', 'white'],
    'sad': ['purple', 'grey', 'red'],
    'precise': ['pink', 'black'],
    'fem': ['brown', 'gold', 'orange', 'pink'],
    'masc': ['red', 'blue', 'orange'],
    'direct': ['red', 'orange']
}

# Each color
color_sequences = {
    'gold': {'x': [250], 'y': [200], 't': [0]},
    'orange': {'x': [250], 'y': [110], 't': [0]},
    'yellow': {'x': [255], 'y': [255], 't': [0]},
    'green': {'x': [0], 'y': [255], 't': [0]},
    'blue': {'x': [0], 'y': [0], 't': [255]},
    'brown': {'x': [180], 'y': [50], 't': [0]},
    'nude': {'x': [250], 'y': [180], 't': [120]},
    'white': {'x': [255], 'y': [255], 't': [255]},
    'purple': {'x': [180], 'y': [50], 't': [255]},
    'grey': {'x': [170], 'y': [170], 't': [170]},
    'red': {'x': [255], 'y': [0], 't': [0]},
    'pink': {'x': [250], 'y': [0], 't': [90]},
    'black': {'x': [0], 'y': [0], 't': [0]}
}

def plot_momentum_analysis(x_coords, y_coords, t_coords):
    momentums = calculate_2d_wave_interpolation(x_coords, y_coords, t_coords)

    real_parts = np.real(momentums)
    imaginary_parts = np.imag(momentums)
    magnitudes = np.abs(momentums)
    phases = np.angle(momentums)

    fig = plt.figure(figsize=(15, 10))

    # Plot momentum components
    plt.subplot(2, 3, 1)
    plt.plot(real_parts, 'bo-', label='Real Part')
    plt.title('Real Part of Momentum')
    plt.xlabel('Index')
    plt.ylabel('Real Part')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(imaginary_parts, 'ro-', label='Imaginary Part')
    plt.title('Imaginary Part of Momentum')
    plt.xlabel('Index')
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(magnitudes, 'go-', label='Magnitude')
    plt.title('Magnitude of Momentum')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(phases, 'mo-', label='Phase')
    plt.title('Phase of Momentum')
    plt.xlabel('Index')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)

    # Plot trajectory
    plt.subplot(2, 3, 5)
    plt.plot(x_coords, y_coords, 'ko-', label='Trajectory')
    plt.scatter(x_coords, y_coords, c=t_coords, cmap='viridis')
    plt.colorbar(label='Time')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory (color = time)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    sequence_name = f"{chosen_pathway}_{'_'.join(colors)}.png"
    plt.savefig(sequence_name)
    plt.show()

# User interface
print("Available pathways:")
for key in pathways:
    print(f"- {key}")

chosen_pathway = input("\nChoose a pathway: ").lower()

if chosen_pathway in pathways:
    colors = pathways[chosen_pathway]
    x_coords, y_coords, t_coords = [], [], []

    for color in colors:
        if color in color_sequences:
            seq = color_sequences[color]
            x_coords.extend(seq['x'])
            y_coords.extend(seq['y'])
            t_coords.extend(seq['t'])

    print(f"\nAnalyzing pathway '{chosen_pathway}' with colors {colors}...")
    plot_momentum_analysis(x_coords, y_coords, t_coords)
else:
    print("Invalid pathway choice. Please run the program again and select a valid pathway.")


