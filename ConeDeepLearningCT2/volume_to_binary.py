import os 
import pydicom
import numpy
from matplotlib import pyplot, cm
import pylab


with open("filename", "wb") as f:
	header = numpy.asarray([1792, 2048, 15]).astype(numpy.uint16)
	#newFile = open("filename", "wb")
	header.tofile(f)	

	for i in range(0):
		# Get the file
		ds = pydicom.read_file("/home/davi/Documentos/Mestrado/Phantom Drake/projections/_"+str(i)+".dcm")

		# Load dimensions based on the number of rows and columns
		ConstPixelDims = (int(ds.Rows), int(ds.Columns))
		print("Quantidade de Rows w Cols")
		print(ConstPixelDims)

		# Load spacing values (in mm)
		ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
		print("Valor do Pixel Spacing")
		print(ConstPixelSpacing)



		# Lista começando em 0, finalizando em
		x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
		#x = 2048
		y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
		#y = 1792


		# The array is sized based on 'ConstPixelDims'
		ArrayDicom = numpy.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)

		ArrayDicom[:, :] = ds.pixel_array  
		ArrayDicom = ArrayDicom.astype(numpy.float32)
		print(ArrayDicom)

		print("Tipo do dado da imagem .dcm")
		print(ArrayDicom.dtype)

		# Primeiro vamos escrever o header
		#Hedader:
		# 2B   2B   2B
		#COLS#ROLS#DEPTH
		#File:
		#floatData	

		# Simulando 180 projeções
		ArrayDicom.tofile(f)
	
