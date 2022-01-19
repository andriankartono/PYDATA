'''
Revisit the Mandelbrot-set. Plot it with imshow and choose a
suitable colormap (hint: any is ok). Then set the axis-ticklabels
such that the axes reflect the section of the complex plane, you
plotted the mandelbrot on (remember, your data coordinates are
pixels!). Mark the coordinate grid axes
'''

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec as gs

MAX_ITER = 150

def mandelbrot(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < MAX_ITER:
        z = z*z + c
        n += 1
    return n

# Image size (pixels)
WIDTH = 800
HEIGHT = 800

# Plot window
RE_START =-2
RE_END =1
IM_START = 2
IM_END = -1

im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
draw = ImageDraw.Draw(im)

for x in range(0, WIDTH):
    for y in range(0, HEIGHT):
        # Convert pixel coordinate to complex number
        c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                    IM_START + (y / HEIGHT) * (IM_END - IM_START))

        if(c.real==0 or c.imag==0):
            draw.point([x,y], (0,0,255))
        else:
        #Compute the number of iterations
            m = mandelbrot(c)
            # The color depends on the number of iterations
            hue = int(255 * m / MAX_ITER)
            saturation = 255
            value = 255 if m < MAX_ITER else 0
            # Plot the point
            draw.point([x, y], (hue, saturation, value))

x_pos=np.linspace(0,800,9)

x_tick=np.linspace(RE_START, RE_END, 9)
y_tick=np.linspace(IM_START, IM_END, 9)

fig=plt.figure()
plt.imshow(im.convert('RGB'))
plt.ylabel('Im')
plt.xlabel('Re')
plt.xticks(x_pos,x_tick)
plt.yticks(x_pos,y_tick)

plt.suptitle('Mandelbrot')
plt.savefig('Mandelbrot.png')