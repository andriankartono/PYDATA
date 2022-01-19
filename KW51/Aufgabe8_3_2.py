'''
Calculate π the very primitive way: Using Monte Carlo integration.
For this you could for example just use the [0, 1] × [0, 1]-square,
draw random (numpy see here) numbers from two independent
distributions and then count how many points are in the
quarter-circle and how many are not.
This is trivially parallelizable, by just doing the aforementioned
procedure on N processors and then collecting the statistics. One
way to do this is pythons multiprocessing. Just replace the function
in the first example with a function doing x-samples of drawing
numbers (ensure that every RNG has a different initial “seed”).
The more adventurous can also try numba, which can make code
which runs with CUDA too! Or joblib... Just remember that
pythons own threads will not speed up anything, because one
python process only ever executes one line of python code (the so
called GIL (Global Interpreter Lock))!
(3pts - 1pts for the basic code, 1pts for a visulization, 1pts for
using multiple processes to speed it up)
'''

import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

# Set number of samples as 50000 to prevent overclustering in figure
throws = 50000


# The sampling function(ignored is just a placeholder)
def sampling(ignored):
    return np.random.rand(2)


# The circle function
def func(x, y):
    if (x**2 + y**2) <= 1:
        return 1
    else:
        return 0


# Calculate the end result
def MC_Integration(inside_circle):
    return 4*inside_circle/throws


def main():
    #Save the number of points in the circle(ask about scope!!)
    counter = 0

    # Multiprocessing
    with Pool(4) as p:
        x = p.map(sampling, range(throws))
        for array in x:
            counter += func(array[0], array[1])
    Pi_Value = MC_Integration(counter)
    print(Pi_Value)

    # Visualization
    complete_array = np.stack(x, axis=0)
    first_element = complete_array[:, 0]
    second_element = complete_array[:, 1]

    # Create Circle and only show a quarter of it.s
    circle1 = plt.Circle((0, 0), 1, fill=False)
    fig, axs = plt.subplots()
    axs.add_patch(circle1)
    plt.scatter(first_element, second_element, s=1)
    axs.set_ylim(0, 1)
    axs.set_xlim(0, 1)

    # Get current figure size
    current_figsize = fig.get_size_inches()
    # Set the figure size as the double of the original one
    # Otherwise the saved figure is shrinked
    fig.set_size_inches(current_figsize * 1.5)
    axs.set_title(
        f"Predicted Value of Pi({throws} samples): {Pi_Value}. There are {counter} samples inside the circle", fontsize="medium")
    plt.savefig("Aufgabe9_3.png")


if __name__ == "__main__":
    main()
