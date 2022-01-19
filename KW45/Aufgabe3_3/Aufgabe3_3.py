'''
Read in the image supplied in the zip as TUM old.jpg. Subsample
by taking every second point and output it as TUM small.png.
Then apply a convolution, using scipy. signal .convolve2d. Make
sure to set the mode-kw-argument to ’same’ so that the output
has the same size as the image. How do you handle different
channels? One of the simplest convolution kernels you could use
would be a box blur. Choose an appropriate size, so that one sees
an effect. Hand in the code and the two resulting pictures
'''

import numpy as np
from PIL import Image
from scipy import signal

im= Image.open("TUM_old.jpg")
image_array=np.array(im)
#print(image_array)
#print(image_array.shape)


#code to sample the picture
image_array_converted= image_array[0::2,0::2]
print(image_array_converted)
#print(image_array_converted.shape)

channel_red=image_array[0:,0:,0]
channel_green=image_array[0:,0:,1]
channel_blue=image_array[0:,0:,2]
#print(channel_red.shape)

#convolution
box_blur=np.ones((6,6))/36

blurred_array_red= signal.convolve2d(channel_red, box_blur, mode='same')
blurred_array_green= signal.convolve2d(channel_green, box_blur, mode='same')
blurred_array_blue= signal.convolve2d(channel_blue, box_blur, mode='same')
blurred_array=np.stack([blurred_array_red,blurred_array_green,blurred_array_blue], axis=-1)
blurred_array=blurred_array.astype(np.uint8)
#print(blurred_array)

#output tum_small
outimg = Image . fromarray ( image_array_converted )
outimg . save ("TUM_small.png")

#output tum_convolution
outimg2 = Image . fromarray ( blurred_array )
outimg2 . save ("TUM_convolution.png")