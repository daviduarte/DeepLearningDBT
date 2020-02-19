"""

Utiliza algumas medidas de comparação de ruído e qualidade de imagem para comparar slices
reconstruídos a partir de métodos diferentes


"""

import numpy as np
import cv2
import os
import pydicom
import imageio
import math
import cv2

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def load_img(path):
	return imageio.imread(path)

def scaling(vet):
	#vet = (vet-np.amin(vet))/(np.amax(vet)-np.amin(vet)) * 255#65535
	vet = vet * (254 / np.amax(vet))
	return vet


def psnr_(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
    	return 100
    
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
	

inputdir = '../input/stage_2_test_images/'
outdir = './'
#os.mkdir(outdir)

#test_list = [ f for f in  os.listdir(inputdir)]

#for f in test_list[:10]:   # remove "[:10]" to convert all images 



# Faz a mesma normalização dos alemães
#img = img * ( 65534 / np.amax( img ) )
#img = img.astype('uint16')
#print(img.dtype)





"""
ssim_value = ssim(label, img2) # Near to 1 is better
print(ssim_value)

psnr = psnr(label, img2)	# Higher is better
print(psnr)

exit()



print(np.amax(img2))
print(np.amin(img2))

print(np.amax(img1))
print(np.amin(img1))

print(img1.dtype)
print(img2.dtype)
"""

base = 80
table = np.zeros((4, 20))

psnr_parker_medio = 0
psnr_rna_medio = 0
ssim_parker_medio = 0
ssim_rna_medio = 0
for i in range(0, 20):

	print("Calculando para imagem " + str(base+i))


	label = load_img("slices/slice_label_"+str(base+i)+".png")
	img1 = load_img("slices/slice_before_training_"+str(base+i)+".png")
	img2 = load_img("slices/slice_after_training_"+str(base+i)+".png")

	label = label[0:300, 700:1250]	
	#max_label = np.amax(label)
	label = scaling(label)
	label = label.astype('uint8')


	img1 = img1[0:300, 700:1250]		
	#max_img1 = np.amax(img1)
	img1 = scaling(img1)
	img1 = img1.astype('uint8')

	img2 = img2[0:300, 700:1250]		
	#max_img2 = np.amax(img2)
	img2 = scaling(img2)
	img2 = img2.astype('uint8')
	
	#max_label_img1 = np.amax([max_label, max_img1])
	#max_label_img2 = np.amax([max_label, max_img2])

	
	
	#img2 = scaling(img2, max_label_img2)
	#label_img1 = scaling(label, max_label_img1)
	#label_img2 = scaling(label, max_label_img2)

	#label_img2 = label_img2.astype('uint8')
	#label_img1 = label_img1.astype('uint8')
	
		

	from PIL import Image
	import numpy as np
	
	#img = Image.fromarray(img1)
	#img.save('my.png')
	#img.show()	

	ssim_value = ssim(img1, img1)
	print(ssim_value)
	exit()


	# SSIM_Ruidosa
	ssim_value = ssim(label, img1)
	table[0,i] = ssim_value
	ssim_parker_medio += ssim_value

	# SSIM_RNA
	ssim_value = ssim(label, img2)
	table[1,i] = ssim_value		
	ssim_rna_medio += ssim_value

	# PSNR ruidosa
	psnr_value = psnr(label, img1)
	table[2,i] = psnr_value
	psnr_parker_medio += psnr_value

	# PSNR RNA
	psnr_value = psnr(label, img2)
	table[3,i] = psnr_value	
	psnr_rna_medio += psnr_value

print(ssim_parker_medio/20)
print(ssim_rna_medio/20)
print(psnr_parker_medio/20)
print(psnr_rna_medio/20)
print(psnr_value)

