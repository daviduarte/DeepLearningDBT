import os 
import pydicom
import numpy as np
from matplotlib import pyplot, cm
import pylab
import math
import tensorflow as tf

proj = np.load("mel_dels.npy")

ramlak_width = 2 * 2048 + 1
pixel_shape = 0.14
# pixel_shape = 0.664
pixel_shape = 1

print(ramlak_width)

def init_ramlak_1DD(  ):
	hw = int( ( ramlak_width-1 ) / 2 )

	f = []
	for i in range( -hw, hw+1 ):
		f.append(0)

	for i in range( -hw, hw+1 ):
		if i == 0:
			f[i] = 1/4 * math.pow( pixel_shape, 2 )
		else:
			if i%2 == 1:
				f[i] = 0
			else:
				f[i] = -1 / math.pow( i * math.pi * pixel_shape, 2 )
	return f


def init_ramlak_1D( ):
    assert( ramlak_width % 2 == 1 )
    hw = int( ( ramlak_width-1 ) / 2 )
    f = [
            -1 / math.pow( i * math.pi * pixel_shape, 2 ) if i%2 == 1 else 0
            for i in range( -hw, hw+1 )
        ]
    f[hw] = 1/4 * math.pow( pixel_shape, 2 )
    return f


print(proj.shape)
kernel = init_ramlak_1D()

print(kernel)
#oi = np.asarray(oi).reshape((1,1,2048,1792))