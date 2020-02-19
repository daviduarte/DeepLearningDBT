import os 
import pydicom
import numpy
from matplotlib import pyplot, cm
import pylab


max_max = float("-Inf")


with open("filename", "wb") as f:

	for i in range(15):

                # Get the file
                ds = pydicom.read_file("/home/davi/Phantom Drake/projections/_"+str(i)+".dcm")

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
                # Transpoe a matriz


                # Vamos normalizar ;)
                max = numpy.amax(ArrayDicom)

                if max > max_max:
                        max_max = max




with open("filename", "wb") as f:
#	header = numpy.asarray([1792, 2048, 1]).astype(numpy.uint16)
#	Transposed:
	header = numpy.asarray([2048, 1792, 1]).astype(numpy.uint16)


	#newFile = open("filename", "wb")
	header.tofile(f)

	for i in range(15):
		# Get the file
		ds = pydicom.read_file("/home/davi/Phantom Drake/projections/_"+str(i)+".dcm")

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

		# Transpoe a matriz
		ArrayDicom = numpy.transpose(ArrayDicom)
		# Inverte para a mama ficar em cima


		# Vamos normalizar ;)
#		max = numpy.amax(ArrayDicom)
#		print("Valor máximo da PROJ: ")
   
    
#		print(max)

     
		div = lambda t: -numpy.log( t / max_max )
		vfunc = numpy.vectorize(div)
		ArrayDicom = vfunc(ArrayDicom)

		print("Projeção depois da divisão: ")
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
	

#numpy.save("teste_bin", ArrayDicom)
