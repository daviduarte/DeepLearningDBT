import os 
import pydicom
import numpy as np
from matplotlib import pyplot, cm
import pylab
import math
import tensorflow as tf

pixel_shape = 0.14
#pixel_shape = 1
#pixel_shape = 1


N = 1
W = 5
H = 3

ramlak_width = 2 * W + 1


def init_ramlak_1D( ):
    assert( ramlak_width % 2 == 1 )
    hw = int( ( ramlak_width-1 ) / 2 )
    f = [
            -1 / math.pow( i * math.pi * pixel_shape, 2 ) if i%2 == 1 else 0
            for i in range( -hw, hw+1 )
        ]
    f[hw] = 1/4 * math.pow( pixel_shape, 2 )
    return f


graph = tf.Graph()
with graph.as_default():
	proj = np.ones((N, H, W), dtype='float32')
	proj_before = tf.reshape(proj, [N, 1, H, W])
	kernel = init_ramlak_1D()

	ramlak_1d = init_ramlak_1D()
	kernel = tf.Variable(initial_value = ramlak_1d, dtype = np.float32, trainable = False)
	kernel = tf.reshape(kernel, [1, ramlak_width, 1, 1])

	proj = tf.nn.conv2d(input = proj_before, filter = kernel, strides = [1,1,1,1], padding = 'SAME', data_format = 'NCHW', name='ramlak-filter')

with tf.Session(graph=graph) as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print("Projection: ")
	print(sess.run(proj_before))
	print("Kernel: ")
	print(sess.run(kernel))
	print("Resultado final")
	print(sess.run(proj))
