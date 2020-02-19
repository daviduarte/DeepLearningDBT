import os 
import pydicom
import numpy
from matplotlib import pyplot, cm
import pylab
import cv2
import tensorflow as tf

im1 = cv2.imread("test_label__.png")
im2 = cv2.imread("test_vol__.png")


print(im1)
print(im2)


with tf.Session() as sess:
	loss = tf.losses.mean_squared_error(im1, im2)
	print(sess.run(loss))







