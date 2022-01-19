'''
Make a picture of the Mandelbrot set on the complex plane for real
parts [-1,0] and complex parts from [0, 1]. Feel free to lookup
algorithms or specific implementations on the internet (cite
appropriately in a comment!), the only requirement is that the
output picture is 800x800 pixels (so this is a lesson in adapting
foreign code or your algorithmic skills) and you save it into a png.
Color it if possible! (1.5 pts - 1 for a black and white Mandelbrot,
0.5 for showing the iterationcount!) Hint: python (and numpy) has
native support for complex numbers
'''
from PIL import Image, ImageDraw

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
RE_START = 0
RE_END =-1
IM_START = 0
IM_END = 1

im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
draw = ImageDraw.Draw(im)

for x in range(0, WIDTH):
    for y in range(0, HEIGHT):
        # Convert pixel coordinate to complex number
        c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                    IM_START + (y / HEIGHT) * (IM_END - IM_START))
        # Compute the number of iterations
        m = mandelbrot(c)
        # The color depends on the number of iterations
        hue = int(255 * m / MAX_ITER)
        saturation = 255
        value = 255 if m < MAX_ITER else 0
        # Plot the point
        draw.point([x, y], (hue, saturation, value))

im.convert('RGB').save('output.png', 'PNG')