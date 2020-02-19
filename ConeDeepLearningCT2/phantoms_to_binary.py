import os 
import pydicom
import numpy
from matplotlib import pyplot, cm
import pylab

dir = "/home/davi/Documentos/Phantoms/wNoise/"
phantoms_dir = "/home/davi/Documentos/ConeDeepLearningCT2/phantoms/lowdose/"

output = [dI for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir,dI))]
output.sort()

#print(output)
#exit()

conter = 33
output = output[conter:]
print(output)
#exit()
for folder in output:
	print("Copiando a folder: ")
	print(folder)
	print("\n")

	max_max = float("-Inf")
	for i in reversed(range(15)):

		# Get the file
		# Let's test with a few examples
		ds = pydicom.read_file(dir + "/" + folder + "/_"+str(i)+".dcm")

		# Load dimensions based on the number of rows and columns
		ConstPixelDims = (int(ds.Rows), int(ds.Columns))
		#print("Quantidade de Rows w Cols")
		#print(ConstPixelDims)

		# Load spacing values (in mm)
		ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
		#print("Valor do Pixel Spacing")
		#print(ConstPixelSpacing)



		# Lista começando em 0, finalizando em
		x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
		#x = 2048
		y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
		#y = 1792


		# The array is sized based on 'ConstPixelDims'
		ArrayDicom = numpy.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)

		ArrayDicom[:, :] = ds.pixel_array
		ArrayDicom = ArrayDicom.astype(numpy.float32)

		#COMENTAR DEPOIS!!!
		# Para testes, vamos amostrar o array em 500 x 500 x 15
		#ArrayDicom = ArrayDicom[773:1273, 0:500]
		# Transpoe a matriz


		# Vamos normalizar ;)
		max = numpy.amax(ArrayDicom)

		if max > max_max:
			max_max = max


	print("Movendo para: ")
	print(phantoms_dir + "binary" + str(conter) + ".proj.bin")
	print("\n")
	with open(phantoms_dir + "binary" + str(conter) + ".proj.bin", "wb") as f:
	#	header = numpy.asarray([1792, 2048, 1]).astype(numpy.uint16)
	#	Transposed:
		header = numpy.asarray([2048, 1792, 15]).astype(numpy.uint16)


		#newFile = open("filename", "wb")
		header.tofile(f)

		for i in reversed(range(15)):
			# Get the file
			ds = pydicom.read_file(dir + "/" + folder + "/_"+str(i)+".dcm")

			# Load dimensions based on the number of rows and columns
			ConstPixelDims = (int(ds.Rows), int(ds.Columns))
			#print("Quantidade de Rows w Cols")
			#print(ConstPixelDims)

			# Load spacing values (in mm)
			ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
			#print("Valor do Pixel Spacing")
			#print(ConstPixelSpacing)



			# Lista começando em 0, finalizando em
			x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
			#x = 2048
			y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
			#y = 1792


			# The array is sized based on 'ConstPixelDims'
			ArrayDicom = numpy.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)

			ArrayDicom[:, :] = ds.pixel_array
			ArrayDicom = ArrayDicom.astype(numpy.float32)
			#print(ArrayDicom)

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
			#ArrayDicom = ArrayDicom[773:1273, 0:500]

			div = lambda t: -1 * numpy.log( t / max_max )
			vfunc = numpy.vectorize(div)
			ArrayDicom = vfunc(ArrayDicom)

			# Garante que o ArrayDicom é Float32 (depois da normalizacao)
			ArrayDicom = ArrayDicom.astype(numpy.float32)

			#print("Projeção depois da divisão: ")
			#print(ArrayDicom)                

			#print("Tipo do dado da imagem .dcm")
			#print(ArrayDicom.dtype)

			# Primeiro vamos escrever o header
			#Hedader:
			# 2B   2B   2B
			#COLS#ROLS#DEPTH
			#File:
			#floatData	

			print("\n\nArrayDicom final que será copiado:")
			print(ArrayDicom)

			# Simulando 180 projeções
			ArrayDicom.tofile(f)
	conter += 1

	if conter == 34:
		exit()


#numpy.save("teste_bin", ArrayDicom)
