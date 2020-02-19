import os 
import pydicom
import numpy as np
from matplotlib import pyplot, cm
import pylab

ds = pydicom.read_file("/home/davi/ConeDeepLearningCT2/phantoms/lowdose/Phantom_teste/wNoise/01_Alvarado_888076.1.685169686552.20171116213109547/_1.dcm")

# Load dimensions based on the number of rows and columns
ConstPixelDims = (int(ds.Rows), int(ds.Columns))
print("Quantidade de Rows w Cols")
print(ConstPixelDims)

# Load spacing values (in mm)
ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
print("Valor do Pixel Spacing")
print(ConstPixelSpacing)

# Lista começando em 0, finalizando em
x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
#x = 2048
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
#y = 1792

# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)

ArrayDicom[:, :] = ds.pixel_array
ArrayDicom = ArrayDicom.astype(np.float32)

#COMENTAR DEPOIS!!!
# Para testes, vamos amostrar o array em 500 x 500 x 15
ArrayDicom = ArrayDicom[773:1273, 0:500]

print(np.max(ArrayDicom))
print(np.min(ArrayDicom))

#print(ArrayDicom)