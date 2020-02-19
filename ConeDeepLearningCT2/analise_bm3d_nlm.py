"""

1 - Importar os phantoms e aplicar a camada de coseno
2 - Passe para Anscombre e faça o BM3D / NLM
3 - Faça o backproject
4 - Salva como png

"""

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import os
from tfcone.inout import dennerlein, projtable, png
import tfcone.util.types as t
import copy
import math

_path = os.path.dirname(os.path.abspath(__file__))
libbackprojepath = os.path.join(_path, 'lib', 'libbackproject.so')
print(libbackprojepath)
_bp_module = tf.load_op_library( libbackprojepath )
backproject = _bp_module.backproject
project = _bp_module.project

proj_test = "/home/davi/Documentos/phantoms/lowdose/binary0.proj.bin"	# Server
#proj_test = "/home/davi/Documentos/Mestrado/phantoms/lowdose/binary0.proj.bin"	# local

g1 = tf.Graph()
g2 = tf.Graph()


# GEOMETRY
VOL_SHAPE           = t.Shape3D(
                        W = 2048, H = 1792, D = 64
#                        W = 500, H = 500, D = 1
#            W = 10, H = 10, D = 1

                    )
VOL_ORIG            = t.Coord3D(
                        X = -204.8, Y = 0, Z = 0    #Corret
#                        X = -200, Y = 0, Z = 31    # Just generates the 31º slice

                    )
VOXEL_DIMS          = t.Shape3D(
                        W = 0.2, H = 0.2, D = 1
#            W = 0.8, H = 0.7, D = 1
                    )
# NOTE: See todo at ct.init_ramlak_1D
PIXEL_DIMS          = t.Shape2D(
                    W = 0.14, H = 0.14
#                    W = 1, H = 1
)
SOURCE_DET_DISTANCE = 700
PROJ_SHAPE          = t.ShapeProj(
                        N = 15, W = 2048, H = 1792
#                    N = 15, W = 500, H = 500
                    )

# Place the breast in center of origin. Utilizing in user-ops/backproject.cu
DISPLACEMENT = VOL_SHAPE.W / 2
displacement = DISPLACEMENT

RAMLAK_WIDTH        = 2*PROJ_SHAPE.W + 1

LIMITED_ANGLE_SIZE  = 15   # #projections for limited angle


class ReconstructionConfiguration:

    '''
        proj_shape
            instance of ProjShape
        vol_shape
            shape of volume (instance of 3DShape)
        vol_origin
            volume origin in world coords (instance of 3DCoord)
        voxel_shape
            size of a voxel in mm (instance of 3DShape)
        pixel_shape
            size of a detector pixel in mm (instance of 2DShape)
        source_det_distance
            in mm
        ramlak_width
            in pixel
    '''
    def __init__(
            self,
            proj_shape,
            vol_shape,
            vol_origin,
            voxel_shape,
            pixel_shape,
            source_det_distance,
            ramlak_width
    ):
        assert( type( proj_shape ) is t.ShapeProj )
        assert( type( vol_shape ) is t.Shape3D )
        assert( type( vol_origin ) is t.Coord3D )
        assert( type( voxel_shape ) is t.Shape3D )
        assert( type( pixel_shape ) is t.Shape2D )

        self.proj_shape             = proj_shape
        self.vol_shape              = vol_shape
        self.vol_origin             = vol_origin
        self.voxel_shape            = voxel_shape
        self.pixel_shape            = pixel_shape
        self.source_det_distance    = source_det_distance
        self.ramlak_width           = ramlak_width

# full angle configuration
CONF = ReconstructionConfiguration(
        PROJ_SHAPE,
        VOL_SHAPE,
        VOL_ORIG,
        VOXEL_DIMS,
        PIXEL_DIMS,
        SOURCE_DET_DISTANCE,
        RAMLAK_WIDTH
)

# limited angle configuration
CONF_LA = copy.deepcopy( CONF )
CONF_LA.proj_shape.N = LIMITED_ANGLE_SIZE

config = CONF_LA


def init_cosine_3D( config ):
    cu = config.proj_shape.W/2 * config.pixel_shape.W
    cv = config.proj_shape.H/2 * config.pixel_shape.H
    sd2 = config.source_det_distance**2

    w = np.zeros( ( 1, config.proj_shape.H, config.proj_shape.W ), dtype =
            np.float32 )

    for v in range( 0, config.proj_shape.H ):
        dv = ( (v+0.5) * config.pixel_shape.H - cv )**2
        for u in range( 0, config.proj_shape.W ):
            du = ( (u+0.5) * config.pixel_shape.W - cu )**2
            w[0,v,u] = config.source_det_distance / math.sqrt( sd2 + dv + dv )

    return w

with g1.as_default():

    # init cosine weights
    cosine_w_np = init_cosine_3D( config )
    cosine_w = tf.Variable(
            initial_value = cosine_w_np,
            dtype = tf.float32,
            trainable = False
    )


    """
    	1 - Importar os phantoms e aplicar a camada de coseno
    """
    proj = dennerlein.read_noqueue(proj_test)
    proj = tf.multiply( proj, cosine_w)

with tf.Session(graph = g1) as sess:
    sess.run( tf.global_variables_initializer() )
    sess.run( tf.local_variables_initializer() )
    proj = sess.run(proj)

    sess.close()

print(proj)

#np.save("proj.npy", proj)
#exit()


"""
	3 - Faça o backproject
"""

#proj = np.load("proj.npy")

with g2.as_default():
    DATA_P = "/home/davi/Documentos/phantoms/lowdose/projMat.txt" # SERVER
    #DATA_P = "/home/davi/Documentos/Mestrado/phantoms/lowdose/projMat.txt" # LOCAL
    geom, angles = projtable.read( DATA_P )
    geom_proto = tf.contrib.util.make_tensor_proto( geom, tf.float32 )

    # initializations for backprojection op
    vol_origin_proto = tf.contrib.util.make_tensor_proto(
            config.vol_origin.toNCHW(), tf.float32 )
    voxel_dimen_proto = tf.contrib.util.make_tensor_proto(
            config.voxel_shape.toNCHW(), tf.float32 )
    pixel_dimen_proto = tf.contrib.util.make_tensor_proto(
            config.pixel_shape.toNCHW(), tf.float32 )

    with tf.device('/gpu:0'):
        vol = backproject(
                projections  = proj,
                geom         = geom_proto,
                vol_shape    = config.vol_shape.toNCHW(),
                vol_origin   = vol_origin_proto,
                voxel_dimen  = voxel_dimen_proto,
                proj_shape   = config.proj_shape.toNCHW(),
                displacement = displacement,
                pixel_dimen  = pixel_dimen_proto        )

        vol = tf.nn.relu( vol )

    """
    	4 - Salva como png
    """
    write_png = png.writeSlice( vol[0], 'testando_a_comparacao_com_outros_metodos.png' )
    #proj_test = dennerlein.read_noqueue(proj_test)
    #write_proj = png.writeSlice( proj_test[0], 'testando_a_comparacao_com_outros_metodos.png' )

GPU_FRACTION        = .9
SAVE_GPU_MEM        = False
GPU_OPTIONS         = tf.GPUOptions( per_process_gpu_memory_fraction = GPU_FRACTION )



with tf.Session(config = tf.ConfigProto( gpu_options = GPU_OPTIONS ), graph = g2 ) as sess:
#with tf.Session( ) as sess:
    sess.run( tf.global_variables_initializer() )
    sess.run( tf.local_variables_initializer() )

    #ref_reconstructor = ct.Reconstructor( CONF, angles, displacement, name = 'RefReconstructor')	
    #sess.run( create_label( proj_test, 'testando_a_comparacao_com_outros_metodos.png', ref_reconstructor, geom ) )

    oi = sess.run(write_png)
    sess.close()
    print("Done!")
