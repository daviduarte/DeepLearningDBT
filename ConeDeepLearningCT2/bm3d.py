import pybm3d
import numpy as np
import cv2
import skimage.data
from skimage.measure import compare_psnr
import anscombe
from PIL import Image

def scaling(proj):
	proj = proj * ( 255 / np.amax( proj ) )
	proj = proj.astype('uint8')
	return proj

proj = np.load("./tfcone/npy_projs/binary0.proj.bin.npy")
proj = proj[0]
#print(im)

"""
noise_std_dev = 40
img = skimage.data.astronaut()
noise = np.random.normal(scale=noise_std_dev,
                         size=img.shape).astype(img.dtype)

noisy_img = img + noise
"""

"""
Printei as duas e ambas possuem os mesmos detalhes das mamas, amboras os valores absolutos sejam 
diferentes
"""
anscombe_proj = anscombe.anscombe(proj)

anscombe_proj_filtred = pybm3d.bm3d.bm3d(anscombe_proj, 1)	# Anscombe deixa a imagem com média = 0 e sd = 1

proj_filtred = anscombe.inverse_anscombe(anscombe_proj_filtred)



#print(denoised_image)

#noise_psnr = compare_psnr(img, noisy_img)
#out_psnr = compare_psnr(img, denoised_image)

"""
Printando imagem
"""
img = Image.fromarray(scaling(proj))
img.save('my.png')
img.show()

img = Image.fromarray(scaling(proj_filtred))
img.save('my.png')
img.show()

exit()


print("PSNR of noisy image: ", noise_psnr)
print("PSNR of reconstructed image: ", out_psnr)

