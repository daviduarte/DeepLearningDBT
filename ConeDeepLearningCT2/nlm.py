import pybm3d
import numpy as np
import cv2
import skimage.data
from skimage.measure import compare_psnr
import anscombe
from PIL import Image
from skimage.restoration import denoise_nl_means
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.restoration import denoise_nl_means, estimate_sigma

def scaling(proj):
	#vet = (vet-np.amin(vet))/(np.amax(vet)-np.amin(vet)) * 255#65535
	#proj = proj / 256
	proj = proj * ( 255 / np.amax( proj ) )
	proj = proj.astype('uint8')
	return proj

def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

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

#proj = img_as_float(data.astronaut())
#proj = proj[30:180, 150:300]

#sigma = 0.08
#noisy = random_noise(proj, var=sigma**2)
#sigma = 1

print(np.amax(proj))
print(np.amin(proj))
print(proj)
anscombe_ = anscombe.anscombe(proj)


# estimate the noise standard deviation from the noisy image
sigma_est = np.mean(estimate_sigma(anscombe_, multichannel=False))
print("estimated noise standard deviation " + str(sigma_est))
#sigma_est = 1

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False)

# slow algorithm, sigma provided
denoise2 = denoise_nl_means(anscombe_, h=0.8 * sigma_est, sigma=sigma_est,
                            fast_mode=False, **patch_kw)

denoise2 = anscombe.inverse_anscombe(denoise2)

print(np.amax(denoise2))
print(np.amin(denoise2))
print(denoise2)

np.save("./tfcone/npy_projs/nlm/binary0.proj.bin.npy", denoise2)


"""
Printando imagem
"""

"""
denoise2 = scaling(denoise2)
img = Image.fromarray(denoise2)
img.save('my.tiff')
img.show()

proj = scaling(proj)
img = Image.fromarray(proj)
img.save('my_filtred.tiff')
img.show()
"""

