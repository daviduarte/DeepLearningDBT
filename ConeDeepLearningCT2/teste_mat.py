import re
import sys
import numpy as np
import math

file_handle = open( "phantoms/lowdose/projMat.txt", 'r' )
file_contents = file_handle.read()
file_handle.close()

regex = re.compile( r"@\s\d*\n([\d.]+)\s+([\d.]+)\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+" )


angles = []
proj = []
# Parece haver apenas 1 matriz no código, pois se houvesse mais, as matrizes seguintes iriam reescrever as variáveis no looping
for match in regex.finditer( file_contents ):        	
	d = [ float( x ) for x in match.groups() ]
	print(d)
	angles += d[0:2]    # Pega os 2 primeiros elementos
	proj += d[2:]   


print("\n\n\n\n\n\n")

assert( len(proj) % 12 == 0 )   # Certifica-se que a matriz tem 3 linhas e 4 colunas

print(angles)

angles = np.array( angles, dtype = np.float )
angles = np.reshape( angles, ( int( len(angles)/2 ), 2 ) )
angles_sum = np.sum( angles, 0 )
angles_i = np.argmax( angles_sum )

print(angles)
print(angles_sum)
print(angles_i)

# rotation axis needs to be parallel to world coordinate system
# Garante que o primeiro número sempre seja 0
assert( angles_sum[ (angles_i+1)%2 ] < 10e-7 )

proj = np.reshape( proj, ( int( len(proj)/12 ), 3, 4 ) )


# convert to radians
angles = angles[:,angles_i] * (math.pi/180)

print(angles)