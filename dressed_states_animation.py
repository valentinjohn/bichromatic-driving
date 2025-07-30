# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 18:06:56 2023

@author: vjohn
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch, basis
from matplotlib.animation import FuncAnimation, PillowWriter


def dressed_states(theta):
    """
    Generate the dressed states |+> and |-> for a given theta.
    """
    g = basis(2, 0)  # Ground state
    e = basis(2, 1)  # Excited state

    plus = np.cos(theta) * g + np.sin(theta) * e  # Longitude-like rotation
    minus = np.sin(theta) * g - np.cos(theta) * e
    # Latitude-like rotation
    # minus = (1/np.sqrt(2)) * (g + np.exp(1j*theta) * e)

    return plus, minus


# Bloch sphere initialization
pipb = Bloch()
b.up = [r"$\uparrow$"]
b.down = [r"$\downarrow$"]
b.vector_color = ['r', 'b']

# Initial dressed states
theta = np.pi/4
plus, minus = dressed_states(theta)

b.add_states(plus)
b.add_states(minus)

b.make_sphere()  # This will create the Bloch sphere plot

fig = b.fig


def update(num):
    b.clear()
    theta = num * np.pi / 30  # Adjust the divisor for speed of rotation
    plus, minus = dressed_states(theta)

    b.add_states(plus)
    b.add_states(minus)
    b.make_sphere()

    # Extract the Axes instance
    ax = b.axes

    # Dummy lines for legend
    line_plus, = ax.plot([], [], lw=2, color='red', label=r'$|+\rangle$')
    line_minus, = ax.plot([], [], lw=2, color='blue', label=r'$|-\rangle$')

    # Create the legend using only our dummy lines
    # Adjust location if needed
    # ax.legend(handles=[line_plus, line_minus], loc='upper right')

    # Remove the dummy lines from the plot
    line_plus.remove()
    line_minus.remove()


# Ensure the figure canvas is drawn before creating the animation
fig.canvas.draw()

# Adjust frames for length of animation
ani = FuncAnimation(fig, update, frames=range(60), repeat=True)
ani.save("dressed_states_animation.gif", writer=PillowWriter(fps=10))

plt.show()
