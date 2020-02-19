import os
import pydicom
import numpy
from matplotlib import pyplot, cm
import pylab


max_max = float("-Inf")


max_novo = float("-Inf")
for i in reversed(range(15)):

	# Get the file
	# Let's test with a few examples
	ds = pydicom.read_file("/home/davi/Documentos/Phantoms/noNoise/00_Alvarado_888076.1.685169686552.20171116213109547/_"+str(i)+".dcm")

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

	print("Tipo do dado da imagem .dcm")
	print(ArrayDicom.dtype)

	ArrayDicom = numpy.transpose(ArrayDicom)
	# Inverte para a mama ficar em cima
	ArrayDicom = numpy.flip(ArrayDicom)


	#COMENTAR DEPOIS!!!
	# Para testes, vamos amostrar o array em 500 x 500 x 15
	ArrayDicom = ArrayDicom[773:1273, 0:500]

	print(ArrayDicom)
	# Transpoe a matriz


	# Vamos normalizar ;)
	for i in range(ArrayDicom.shape[0]):
		for j in range(ArrayDicom.shape[1]):
			if  ArrayDicom[i, j] > max_novo:
				max_novo = ArrayDicom[i,j]
	max = numpy.amax(ArrayDicom)

	if max > max_max:
		max_max = max


print("Valor máximo depois do primeiro LOOPING")
print(max_novo)
print(max_max)


with open("filename", "wb") as f:
#	header = numpy.asarray([1792, 2048, 1]).astype(numpy.uint16)
#	Transposed:

	header = numpy.asarray([500, 500, 15]).astype(numpy.uint16)
#	header = numpy.asarray([2048, 1792, 15]).astype(numpy.uint16)


	#newFile = open("filename", "wb")
	header.tofile(f)

	for i in reversed(range(15)):
		# Get the file
		ds = pydicom.read_file("/home/davi/Documentos/Phantoms/noNoise/00_Alvarado_888076.1.685169686552.20171116213109547/_"+str(i)+".dcm")

		# Load dimensions based on the number of rows and columns
		ConstPixelDims = (int(ds.Rows), int(ds.Columns))
#		print("Quantidade de Rows w Cols")
#		print(ConstPixelDims)

		# Load spacing values (in mm)
		ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
#		print("Valor do Pixel Spacing")
#		print(ConstPixelSpacing)



		# Lista começando em 0, finalizando em
		x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
		#x = 2048
		y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
		#y = 1792


		# The array is sized based on 'ConstPixelDims'
		ArrayDicom = numpy.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)

		ArrayDicom[:, :] = ds.pixel_array
		ArrayDicom = ArrayDicom.astype(numpy.float32)
#		print(ArrayDicom)

		# Transpoe a matriz
		ArrayDicom = numpy.transpose(ArrayDicom)
		# Inverte para a mama ficar em cima
		ArrayDicom = numpy.flip(ArrayDicom)


		# Vamos normalizar ;)
#		max = numpy.amax(ArrayDicom)
#		print("Valor máximo da PROJ: ")
   
    
#		print(max)


		# COMENTAR DEPOIS!!!
		# Para testes, vamos amostrar o array em 500 x 500 x 15
		ArrayDicom = ArrayDicom[773:1273, 0:500]



                # Normalização de Beer. Questões da física ;)

		print("\n\nValor máximo entre todas as projeções")
		print(max_max)
		div = lambda t: -1 * numpy.log( t / max_max )
		vfunc = numpy.vectorize(div)
		ArrayDicom = vfunc(ArrayDicom)


		# Vamos coloca runs valores aleatórios para testar se o pipeline do traning pega amostras diferentes
		# APAGAR DEPOIS!!!!!!
		ArrayDicom = numpy.random.rand(500,500)
		print("\n\nProjecao qualquer antes de normalizar")
		print(ArrayDicom)


                # Normaliza. Vamos ver se a NN funciona assim
#		ArrayDicom = ArrayDicom / max_max

#		print("Projeção depois da divisão: ")
#		print(ArrayDicom)                

		print("Tipo do dado da imagem .dcm")
		print(ArrayDicom.dtype)

		# Primeiro vamos escrever o header
		#Hedader:
		# 2B   2B   2B
		#COLS#ROLS#DEPTH
		#File:
		#floatData	

#		print("Normalizando valores entre 0->1")
#		ArrayDicom = ArrayDicom / numpy.amax(ArrayDicom)

		print("\n\nProjeção com o LOG e já normalizada")
		print(ArrayDicom)

		# Simulando 180 projeções
		ArrayDicom = ArrayDicom.astype(numpy.float32)
		ArrayDicom.tofile(f)


#numpy.save("teste_bin", ArrayDicom)
