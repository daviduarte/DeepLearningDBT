import tensorflow as tf
import numpy as np
import os
import copy
import random
from inout import dennerlein, projtable, png, dataset_format
from algo import ct
from util import types as t
from tensorflow.python.client import timeline
import re
import math
import argparse

# CONFIGGG
#-------------------------------------------------------------------------------------------

# GEOMETRY
VOL_SHAPE           = t.Shape3D(
                        W = 2048, H = 1792, D = 64
#                        W = 500, H = 500, D = 1
#			 W = 10, H = 10, D = 1

                    )
VOL_ORIG            = t.Coord3D(
                        X = -204.8, Y = 0, Z = 0	#Corret
#                        X = -200, Y = 0, Z = 31	# Just generates the 31º slice

                    )
VOXEL_DIMS          = t.Shape3D(
                        W = 0.2, H = 0.2, D = 1
#			 W = 0.8, H = 0.7, D = 1
                    )
# NOTE: See todo at ct.init_ramlak_1D
PIXEL_DIMS          = t.Shape2D(
                    W = 0.14, H = 0.14
#                    W = 1, H = 1
)
SOURCE_DET_DISTANCE = 700
PROJ_SHAPE          = t.ShapeProj(
                        N = 15, W = 2048, H = 1792
#			         N = 15, W = 500, H = 500
                    )

# Place the breast in center of origin. Utilizing in user-ops/backproject.cu
DISPLACEMENT = VOL_SHAPE.W / 2

RAMLAK_WIDTH        = 2*PROJ_SHAPE.W + 1


# DATA HANDLING
DATA_P              = os.path.abspath(
                        os.path.dirname( os.path.abspath( __file__ ) )
                        + '/../../phantoms/lowdose/'
                    ) + '/'
PROJ_FILES          = [ DATA_P + f for f in os.listdir( DATA_P )
                        if f.endswith(".proj.bin") ]
VOL_FILES           = [ DATA_P + f for f in os.listdir( DATA_P )
                        if f.endswith(".vol.bin") ]
PROJ_FILES.sort()
VOL_FILES.sort()
LOG_DIR             = '/home/davi/Documentos/train/'
  

# TRAINING CONFIG
LIMITED_ANGLE_SIZE  = 15   # #projections for limited angle
#LEARNING_RATE       = 0.00002	# tentar essa. Provaelmente foi essa com os melhores resultados
#LEARNING_RATE       = 0.0000000002
#LEARNING_RATE       = 0.0000002	# Usando esta na base de dados normal, o treinamento vai se rmais lento
LEARNING_RATE       = 0.0002

EPOCHS              = None  # unlimited
TRACK_LOSS          = 5000    # number of models/losses to track for early stopping
TEST_EVERY          = 1     # number of training steps before test is run
WEIGHTS_TYPE        = 'parker'

# GPU RELATED STAFF
GPU_FRACTION        = .5
SAVE_GPU_MEM        = False
GPU_OPTIONS         = tf.GPUOptions( per_process_gpu_memory_fraction = GPU_FRACTION )


# SOME SETUP
#-------------------------------------------------------------------------------------------

# full angle configuration
CONF = ct.ReconstructionConfiguration(
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


# SETUP INPUT PIPELINE
#-------------------------------------------------------------------------------------------
def input_pipeline( train_proj_fns, train_vol_fns, test_proj_fns, test_vol_fns ):
    train_proj = tf.train.string_input_producer( train_proj_fns, num_epochs = EPOCHS,
            shuffle = False )
    train_label = tf.train.string_input_producer( train_vol_fns, num_epochs = EPOCHS,
            shuffle = False )
    test_proj  = tf.train.string_input_producer( test_proj_fns,
            shuffle = False )
    test_label  = tf.train.string_input_producer( test_vol_fns,
            shuffle = False )


    train_proj = dennerlein.read( train_proj, 'TrainProjReader',
            PROJ_SHAPE.toNCHW() )
    train_label = dennerlein.read( train_label, 'TrainLabelReader',
            VOL_SHAPE.toNCHW() )
    test_proj = dennerlein.read( test_proj, 'TestProjReader',
            PROJ_SHAPE.toNCHW() )
    test_label = dennerlein.read( test_label, 'TestLabelReader',
            VOL_SHAPE.toNCHW() )

    # pick LA slice
    train_proj = train_proj[0:LIMITED_ANGLE_SIZE]
    test_proj = test_proj[0:LIMITED_ANGLE_SIZE]

    train_proj_batch, train_label_batch = tf.train.shuffle_batch(
            [ train_proj, train_label ],
            batch_size = 1,
            capacity = 10,
            min_after_dequeue = 5
    )



    return train_proj_batch[0], train_label_batch[0], test_proj, test_label


# MODEL
#-------------------------------------------------------------------------------------------
class Model:

    def mean_error(self, y, y_):
        cost = tf.abs(tf.reduce_mean(tf.subtract(y,y_)))
        return cost

    def train_on_projections( self, train_proj, train_label, reconstructor, geom ):

        train_step = tf.no_op()

        volume_la = reconstructor.apply( train_proj, geom )

        if SAVE_GPU_MEM:
            # compute loss on cpu (avoid too many volume instances on gpu)
            with tf.device("/cpu:0"):
#                t1 = tf.math.reduce_max(train_label)
#                t2 = tf.math.reduce_max(volume_la)
#                max = tf.math.reduce_max([t1, t2])
#                train_label_ = train_label / max
#                volume_la_ = volume_la / max
#                loss = tf.losses.mean_squared_error( train_label_, volume_la_ )

                #loss = tf.losses.mean_squared_error( tf.math.scalar_mul(10e10, train_label), tf.math.scalar_mul(10e10, volume_la) )
                pass

#                loss = tf.losses.mean_squared_error( train_label, volume_la )
#                 loss = tf.losses.mean_squared_error( tf.math.scalar_mul(10e10, train_label), tf.math.scalar_mul(10e10, volume_la) )

        else:

            t1 = tf.math.reduce_max(train_label)
            t2 = tf.math.reduce_max(volume_la)
            #max = tf.math.reduce_max([t1, t2])
            #train_label_ = train_label * ( 254 / max )
            train_label_ = train_label * ( 254 / t1 )
            #volume_la_ = volume_la * (254 / max )
            volume_la_ = volume_la * (254 / t2 )
#            train_label = tf.Print(train_label, [train_label])
#            loss = tf.losses.mean_squared_error( train_label, volume_la )

#            train_label = tf.slice(train_label, [0,0,0], [1,50,50])
            loss = self.mean_error(volume_la, train_label)
#            loss = tf.losses.mean_squared_error(volume_la, train_label)
#            loss = tf.losses.log_loss(train_label, volume_la)

            #tf.summary.histogram("loss_treinemento", loss)
            loss_summary = tf.summary.scalar("training_loss", loss)

#            X = tf.placeholder(shape=[1],dtype=tf.float32)
 #           loss = tf.math.add(X, loss)

#            a = np.ones((1, 500, 500)) * 1000000000000
#            train_label_ = tf.add(train_label, a)
#            volume_la_ =tf.add(volume_la, a) 

#            loss = tf.losses.mean_squared_error( train_label_, volume_la_ )


#            loss = tf.losses.mean_squared_error( tf.math.scalar_mul(10e10, train_label), tf.math.scalar_mul(10e10, volume_la) )

        with tf.control_dependencies( [ train_step ] ):
            gstep = tf.train.get_global_step()

            train_step = tf.train.GradientDescentOptimizer( LEARNING_RATE ).minimize(
                    loss,
                    colocate_gradients_with_ops = True,
                    global_step = gstep
                )

        return train_step, loss_summary

    def setTest( self, test_proj, test_vol, sess ):
        sess.run( self.test_proj_fns.initializer, feed_dict = {
                    self.test_proj_fns_init: test_proj
                }
            )
        sess.run( self.test_vol_fns.initializer, feed_dict = {
                    self.test_vol_fns_init: test_vol
                }
            )

    def __init__( self, train_proj, train_vol, test_proj, test_vol, sess ):

        self.train_proj_fns_init = tf.placeholder( tf.string, shape = ( len(train_proj) ) )
        self.train_vol_fns_init = tf.placeholder( tf.string, shape = ( len(train_vol) ) )
        self.test_proj_fns_init = tf.placeholder( tf.string, shape = ( len(test_proj) ) )
        self.test_vol_fns_init = tf.placeholder( tf.string, shape = ( len(test_vol) ) )

        self.train_proj_fns = tf.Variable( self.train_proj_fns_init, trainable = False,
                collections = [] )
        self.train_vol_fns = tf.Variable( self.train_vol_fns_init, trainable = False,
                collections = [] )
        self.test_proj_fns = tf.Variable( self.test_proj_fns_init, trainable = False,
                collections = [] )
        self.test_vol_fns = tf.Variable( self.test_vol_fns_init, trainable = False,
                collections = [] )
        sess.run( self.train_proj_fns.initializer, feed_dict = {
                    self.train_proj_fns_init: train_proj
                }
            )
        sess.run( self.train_vol_fns.initializer, feed_dict = {
                    self.train_vol_fns_init: train_vol
                }
            )
        sess.run( self.test_proj_fns.initializer, feed_dict = {
                    self.test_proj_fns_init: test_proj
                }
            )
        sess.run( self.test_vol_fns.initializer, feed_dict = {
                    self.test_vol_fns_init: test_vol
                }
            )

        geom, angles = projtable.read( DATA_P + 'projMat.txt' )
        self.geom = geom

        re = ct.Reconstructor(
                CONF_LA, angles[0:LIMITED_ANGLE_SIZE], DISPLACEMENT,
                trainable = True,
                name = 'LAReconstructor',
                weights_type = WEIGHTS_TYPE        
            )
        geom_la = geom[0:LIMITED_ANGLE_SIZE]

        with tf.device("/cpu:0"):
            train, train_label, test, test_label = input_pipeline(
                    self.train_proj_fns,
                    self.train_vol_fns,
                    self.test_proj_fns,
                    self.test_vol_fns
                )
        self.test = test
        self.test_label = test_label
        self.train_proj = train
        self.train_label = train_label
        self.re = re
        

        # Salva as projeções como uma variável para eu tentar salvá-las como png e ver se está no formato correto
        self.train_data = train


        if not tf.train.get_global_step():
            tf.train.create_global_step()

        self.train_op = self.train_on_projections( train, train_label, re, geom_la )
        self.test_vol = re.apply( test, geom_la )

        with tf.device("/cpu:0"):
            #self.test_loss = png.writeSlice(test_label[0], '/media/davi/526E10CC6E10AAAD/mestrado_davi/train/test_model_0/test_label__.png')
            #self.test_loss = tf.Print(test_label, [test_label], summarize=300000)
            # Vamos normalizar os valores para não dar problema com a LOSS

            t1 = tf.math.reduce_max(test_label)
            t2 = tf.math.reduce_max(self.test_vol)
            #max = tf.math.reduce_max([t1, t2])
            test_label_ = test_label * (254 / t1)
            test_vol_ = self.test_vol * (254 / t2)

#            a = np.ones((1, 500, 500)) * 1000000000000
#            train_label_ = tf.add(train_label, a)
#            test_vol_ = tf.add(self.test_vol, a)

#            self.test_loss = tf.losses.mean_squared_error( train_label_, test_vol_ )
#            t1 = test_label #tf.slice(test_label, [0,0,225], [1,50,50])
#            t2 = self.test_vol #tf.slice(self.test_vol, [0, 0, 225], [1,50,50])

#            test_label_ = tf.Print(t1, [tf.reduce_max(t1)], message="\n\nValor maximo do TEST_LABEL: ", summarize=2500)
#            test_label_ = tf.Print(test_label_, [tf.reduce_min(t1)], message="Valor minimo do TEST_LABEL: ", summarize=2500)
#            test_label_ = tf.Print(test_label_, [tf.reduce_mean(t1)], message="Valor MEDIO do TEST_LABEL: \n\n", summarize=2500)


#            test_vol_ = tf.Print(t2, [tf.reduce_max(t2)], message="Valor maximo do TEST_VOL: ", summarize=2500)
#            test_vol_ = tf.Print(test_vol_, [tf.reduce_min(t2)], message="Valor minimo do TEST_VOL: ", summarize=2500)
#            test_vol_ = tf.Print(test_vol_, [tf.reduce_mean(t2)], message="Valor medio do TEST_VOL: ", summarize=2500)



#            x_as_string = tf.map_fn(lambda xi: tf.strings.format('{}', xi), test_label_, dtype=tf.string)
#            test_label_ = tf.io.write_file("ARQUIVO_CONTENDO_O_TEST_LABEL.txt", x_as_string[0], name=None)
#            np.savetxt('test.out', tf.Session().run(t1), delimiter=',')
#            self.test_loss = tf.losses.mean_squared_error( test_label, self.test_vol )
#            self.test_loss = self.mean_error(self.test_vol, test_label)

            #test_label = tf.Print(test_label, [test_label[0, 225:275, 0:50].shape], message="\n\n\nAmostra do test")
            self.test_loss = self.mean_error(test_label, self.test_vol)
#            self.test_loss = tf.losses.mean_squared_error(test_label, self.test_vol)
            self.test_loss_summary = tf.summary.scalar("validation_loss", self.test_loss)

#            self.test_loss = tf.losses.log_loss(test_label, self.test_vol)


#            self.test_loss = tf.losses.mean_squared_error( test_label, self.test_vol )
            #self.test_loss = tf.losses.mean_squared_error( tf.math.scalar_mul(10e10, test_label), tf.math.scalar_mul(10e10, self.test_vol) )
#            self.test_loss = tf.losses.mean_squared_error( np.ones((1,500,500)), self.test_vol )


def split_train_validation_set( offset ):
    N = len( PROJ_FILES ) - 1
    ntrain = N - 1

    validation_idx = (offset-1) % N

    # put  test file to end
    proj_files = copy.deepcopy( PROJ_FILES )
    del proj_files[offset]              # do not use test proj
    validation_proj = proj_files[validation_idx]
    del proj_files[validation_idx]

    vol_files = copy.deepcopy( VOL_FILES )
    del vol_files[ offset ]             # do not use test volume
    validation_vol = vol_files[validation_idx]
    del vol_files[validation_idx]

    return proj_files, vol_files, [validation_proj], [validation_vol]


# 20% de testes e 80% de validação
def my_split_train_test_set():
    pass


# Training set = 80% do set em geral, sendo que 1 exemplo é para validação, como fez os Alemães
def my_split_train_validation_set():
    N_TEST = int((20/100) * len( PROJ_FILES ))  # Quantidade de Phantom de teste = 20%
    N = len( PROJ_FILES ) - N_TEST

    # Tira o TEST SET (20 últimos elementos)
    
    training_proj = copy.deepcopy( PROJ_FILES )
    del training_proj[N: (N + N_TEST)]
    validation_proj = training_proj[0]
    del training_proj[0]

    
    training_vol = copy.deepcopy( VOL_FILES )
    del training_vol[N: (N + N_TEST)]
    validation_vol = training_vol[0]
    del training_vol[0]

    return training_proj, training_vol, [validation_proj], [validation_vol]



def my_train():

	EPOCHS = 10
	BATCH_SIZE = 16


	train_proj = ["/home/davi/Documentos/ConeDeepLearningCT2/phantoms/lowdose/binary0.proj.bin",

		      ]

	train_label = ["/home/davi/Documentos/ConeDeepLearningCT2/phantoms/lowdose/binary0.vol.bin",

		      ]

	save_path = '/home/davi/Documentos/train/model_%d/' % 0
	with tf.Session() as sess:
		
		train_list = []
		label_list = []
		for i in range(len(train_proj)):

			train_list_ = sess.run(dennerlein.read_noqueue(train_proj[i]))
			train_list.append(train_list_)

			label_list_ = sess.run(dennerlein.read_noqueue(train_label[i]))
			label_list.append(label_list_)

	geom, angles = projtable.read( DATA_P + 'projMat.txt' )
	# Beleza, temos as amostras em train_proj e os labels em train_label

	BATCH_SIZE = 1
	features, labels = (train_list, label_list)
	dataset = tf.data.Dataset.from_tensor_slices((np.asarray(features),np.asarray(labels))).repeat().batch(BATCH_SIZE)
	iter = dataset.make_one_shot_iterator()
	x, y = iter.get_next()

	EPOCHS = 100000
	with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:

		global LEARNING_RATE
		# Ajeita as coisas para printar um slice antes de comçar a treinar
		re = ct.Reconstructor(
			CONF_LA, angles[0:LIMITED_ANGLE_SIZE], DISPLACEMENT,
			name = 'LAReconstructor',
			weights_type = WEIGHTS_TYPE		
        )

		volume_la = re.apply( x, geom )

		# Esse reconstructor será utilizado para corrigir a imagem durante as EPOCHS do treinamento
		re = ct.Reconstructor(
			CONF_LA, angles[0:LIMITED_ANGLE_SIZE], DISPLACEMENT,
			trainable = True,
			name = 'LAReconstructor',
			weights_type = WEIGHTS_TYPE
		)


		if not tf.train.get_global_step():
		    tf.train.create_global_step()

		sess.run( tf.global_variables_initializer() )
		sess.run( tf.local_variables_initializer() )

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners( sess = sess, coord = coord )

		saver = tf.train.Saver( max_to_keep = TRACK_LOSS )

		# Printa
		write_png = png.writeSlice( volume_la[0], '/home/davi/Documentos/train/test_model_0/slice_qualquer_antes_treinamento.png' )
		sess.run(write_png)

		try:
			for i in range(EPOCHS):
				
#				if i % 30 == 0:
#					print("Abaixando a LEARNING RATE")
#					LEARNING_RATE /= 10

				proj = train_list[0]
				label = label_list[0]

				train_step = tf.no_op()
				

				volume_la = re.apply( x, geom )
				volume_la = tf.expand_dims(volume_la, axis=0)

				t1 = tf.math.reduce_max(y)
				t2 = tf.math.reduce_max(volume_la)
				label_ = y * ( 255 / t1 )
				volume_la_ = volume_la * (255 / t2 )


				# Add a extension to the tensor
				write_png = png.writeSlice( volume_la_[0][0], '/home/davi/Documentos/train/test_model_0/slice_qualquer_depois_treinamento.png' )
				sess.run(write_png)

				loss = tf.losses.mean_squared_error( label_, volume_la_ )

				with tf.control_dependencies( [ train_step ] ):
				    gstep = tf.train.get_global_step()

				    train_step = tf.train.GradientDescentOptimizer( LEARNING_RATE ).minimize(
					    loss,
					    colocate_gradients_with_ops = True,
					    global_step = gstep
					)



				step = sess.run(tf.train.get_global_step())
				if step % 10:
					print("Salvando o modelo")
					saver.save( sess, save_path + 'model', global_step = step )

				# Treinando
				print("Treinando")
				sess.run(train_step)


		except tf.errors.OutOfRangeError:
			print( 'Done.' )
		finally:



			coord.request_stop()


        


def train_model( offset, save_path, resume ):
    global LEARNING_RATE
    losses = []
    stop_crit = lambda l: np.median( l[:int(len(l)/2)] ) < np.median( l[int(len(l)/2):] )

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        #sets = split_train_validation_set( offset )
        sets = my_split_train_validation_set() # O primeiro elemento é 1 phantom para validação

        m = Model( *sets, sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        c = lambda l: stop_crit( l ) if len( l ) >= TRACK_LOSS else False

        saver = tf.train.Saver( max_to_keep = TRACK_LOSS )

        if resume:
            cp = tf.train.latest_checkpoint( save_path )
            if cp:
                print( 'Restoring session' )
                saver.restore( sess, cp )
        
        write_png = png.writeSlice( m.test_vol[0], '/home/davi/Documentos/train/test_model_0/test_label_antes_treinamento.png' )
        sess.run(write_png)


        train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)
        count_all = 0
        try:
            while 1: #not coord.should_stop() and not c( losses ):
                for i in range( TEST_EVERY ):
                    count_all += 1
                    sess.run( m.train_op[0] )
                    sess.run( m.train_op[1] )
                    

                lv, step = sess.run( [ m.test_loss, tf.train.get_global_step() ] )
                sess.run(m.test_loss_summary)

                #print("Step: " + str(step))
                #print("Loss: " + str(lv))
                print( 'Step %d; Loss: %f' % ( step, lv ), flush=True)
                print(lv, flush=True)
                
                #tf.summary.histogram("loss_validacao", lv)

                merge = tf.summary.merge_all()    
                summary = sess.run(merge) 
                train_writer.add_summary(summary, count_all)    
#                if step == 17:
#                    print("\nAbaixando a LARARNING_RATE \n")
#                    LEARNING_RATE /= 10


                #write_png = png.writeSlice( m.test_vol[0], '/media/davi/526E10CC6E10AAAD/mestrado_davi/train/test_model_0/test_vol__.png' )
                #sess.run(write_png)  

                losses.append( lv )
                if len( losses ) > TRACK_LOSS:
                    del losses[0]

                saver.save( sess, save_path + 'model', global_step = step )

        except tf.errors.OutOfRangeError:
            print( 'Done.' )
        finally:

            write_png = png.writeSlice( m.test_vol[0], '/home/davi/Documentos/train/test_model_0/test_label_depois_treinamento.png' )
            sess.run(write_png)

            coord.request_stop()

        coord.join( threads )
        sess.close()

    tf.reset_default_graph()

    return losses, step


# LABELS
#-------------------------------------------------------------------------------------------
def create_label( fn_proj, fn_vol, rec, geom ):
    proj = dennerlein.read_noqueue( fn_proj )
    volume = rec.apply( proj, geom, fullscan = True ) #MUDAR DEPOIS PARA TRUE!!!!!!
    return dennerlein.write( fn_vol, volume )

def update_labels():
    geom, angles = projtable.read( DATA_P + 'projMat.txt' )
    ref_reconstructor = ct.Reconstructor( CONF, angles, DISPLACEMENT, name = 'RefReconstructor')

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        sess.run( tf.global_variables_initializer() )
        sess.run(tf.local_variables_initializer())


        for fn_proj in PROJ_FILES:
            fn_vol = fn_proj.replace( 'proj', 'vol' )

            if not os.path.exists( fn_vol ):
                print( 'Creating label for %s' % fn_proj )
                sess.run( create_label( fn_proj, fn_vol, ref_reconstructor, geom ) )
                VOL_FILES.append( fn_vol )
        sess.close()


    tf.reset_default_graph()

    PROJ_FILES.sort()
    VOL_FILES.sort()


# TEST
#-------------------------------------------------------------------------------------------
def write_test_volumes( test_proj, test_label ):

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        m = Model( [test_proj], [test_label], [test_proj], [test_label], sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        # compute volumes before training and export dennerlein
        proj_filename = os.path.splitext( os.path.splitext( os.path.basename( test_proj ) )[0] )[0]
        out_fa = LOG_DIR + ( 'test_%s_%s_fa.bin' % ( proj_filename, WEIGHTS_TYPE ) )
        out_la = LOG_DIR + ( 'test_%s_%s_la.bin' % ( proj_filename, WEIGHTS_TYPE ) )

        vol_before = m.test_vol
        write_dennerlein = dennerlein.write( out_la, vol_before )
        write_dennerlein_label = dennerlein.write( out_fa, m.test_label )
        sess.run( [ write_dennerlein, write_dennerlein_label ]  )

        coord.request_stop()
        coord.join( threads )
        sess.close()

    tf.reset_default_graph()

def test_model( validation_proj, validation_label, test_proj, test_label, save_path, log_dir, i ):
    step = 0
    losses = []
 
    print("********* Iteracao: " + str(i) + "\n\n")

    # find checkpoint files
    checkpoints = []
    with open( save_path + 'checkpoint' ) as f:
        f = f.readlines()
        pattern = re.compile( 'all_model_checkpoint_paths:\s\"(.+)\"' )
        for line in f:
            for match in re.finditer( pattern, line ):
                checkpoints.append( match.groups()[0] )


    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        m = Model( [test_proj], [test_label], [test_proj], [test_label], sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

#        write_png = png.writeSlice( m.test_label[0], log_dir + 'slice_label_'+str(i)+'.png' )
#        sess.run(write_png)


        # compute volume before training and export central slice
        print( 'Computing volume '+str(i)+' without trained parameters' )
        vol_before = m.test_vol
#        write_png = png.writeSlice( vol_before[CONF.proj_shape.N // 2], log_dir + 'slice_before_training.png' )
        write_png = png.writeSlice( vol_before[31], log_dir + 'slice_before_training_'+str(i)+'.png' )
#        write_png_label = png.writeSlice( m.test_label[CONF.proj_shape.N // 2], log_dir + 'slice_label.png' )
#        write_png_label = png.writeSlice( m.test_label[0], log_dir + 'slice_label.png' )

#        vol_before_np, _, _, vol_label_np = sess.run( [vol_before, write_png, write_png_label, m.test_label] )
        vol_before_np, _ = sess.run( [vol_before, write_png] )

        # find best checkpoint
#        print( 'Searching best checkpoint' )
        saver = tf.train.Saver()
        m.setTest( validation_proj, validation_label, sess )

        best_cp_i = 51
        best_cp_loss = math.inf

        #print("Hard Code for the best checkpoint 1102")

        print("\n\n**********************************")
        #print(checkpoints)
        print(checkpoints[best_cp_i])

        

        """
        for i, cp in enumerate( checkpoints ):
#        for i in range(300, 400):
            saver.restore( sess, checkpoints[i] )
            loss = sess.run( m.test_loss )
            print("Modelo: ")
            print(i)
            print("Loss: ")
            print(loss)

            if loss < best_cp_loss:
                best_cp_i = i
                best_cp_loss = loss
            print( '.', end = '', flush = True )
        """

        print( '' )
        print("Terminando de testar os modelos ********\n\n")
        

        
        print("Melhor loss: ")
        print(best_cp_i)
        saver.restore( sess, checkpoints[best_cp_i] )
        #print("Loss do modelo " + str(best_cp_i))
        #loss = sess.run( m.test_loss )
        #print(loss)


        # load best model and set test volume
        print( 'Computing volume '+str(i)+' with trained parameters' )
        m.setTest( [test_proj], [test_label], sess )
        saver.restore( sess, checkpoints[best_cp_i] )

        # compute volume after training and export central slice + dennerlein
        vol_after = m.test_vol
        #write_png = png.writeSlice( vol_after[CONF.proj_shape.N // 2], log_dir + 'slice_after_training.png' )
        write_png = png.writeSlice( vol_after[31], log_dir + 'slice_after_training_'+str(i)+'.png' )
        write_dennerlein = dennerlein.write( log_dir + 'after_training_'+str(i)+'.bin', vol_after )
        vol_after_np, _, _ = sess.run( [vol_after, write_png, write_dennerlein]  )

        coord.request_stop()
        coord.join( threads )
        sess.close()

    tf.reset_default_graph()

    return losses, step


def minha_funcao_pessoal( validation_proj, validation_label, test_proj, test_label, save_path, log_dir, i ):
    # Vamos tentar escrever o png do slice da reconstrução analítica

    # Simplesmente cria o arquivo na pasta.
    update_labels()



    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:

        #print("\n\n\nTest label: ")
        #print(test_label)
        #print("\n\n\n")
        m = Model( [test_proj], [test_label], [test_proj], [test_label], sess )

#        print("\n\n\nDados das projeções: ")
#        print(m.train_data)

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        oi = sess.run(m.test_label)


#        for i in range(1):
        write_png = png.writeSlice( m.test_label[0], log_dir + 'slice_label_' + str(i) + '.png' )
        sess.run(write_png)

        #write_png = png.writeSlice( m.train_data[0], log_dir + 'uma_projecao_do_train.png' )
        #sess.run(write_png)

        return

        # compute volume before training and export central slice
        print( 'Computing volume without trained parameters' )
        vol_before = m.test_vol
        write_png = png.writeSlice( vol_before[CONF.proj_shape.N // 2], log_dir + 'slice_before_training.png' )
        write_png_label = png.writeSlice( m.test_label[CONF.proj_shape.N // 2], log_dir + 'slice_label.png' )
        vol_before_np, _, _, vol_label_np = sess.run( [vol_before, write_png, write_png_label, m.test_label] )



def minha_funcao_pessoal2():

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)

        sets = split_train_validation_set( 0 )
        m = Model( *sets, sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )            

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        

        """
        write_png = png.writeSlice( m.train_proj[0], '/media/davi/526E10CC6E10AAAD/mestrado_davi/train/test_model_0/uma_projecao_do_train.png' )
        sess.run(write_png)

        
        #volume_la = sess.run(volume_la)

        write_png = png.writeSlice( volume_la[0], '/media/davi/526E10CC6E10AAAD/mestrado_davi/train/test_model_0/uma_slice_do_train.png' )
        sess.run(write_png)

        write_png = png.writeSlice( m.test_label[0], '/media/davi/526E10CC6E10AAAD/mestrado_davi/train/test_model_0/test_label_0.png' )
        sess.run(write_png)

        """

        optimizer = tf.train.GradientDescentOptimizer( LEARNING_RATE )
        for i in range(100):            

            merge = tf.summary.merge_all()    

            summary = sess.run(merge)    
                        

            volume_la = m.re.apply( m.train_proj, m.geom)

            t1 = tf.math.reduce_max(m.train_label)
            t2 = tf.math.reduce_max(volume_la)
            train_label_ = m.train_label * (254/t1)
            volume_la_ = volume_la * (254/t2)
            loss = tf.losses.mean_squared_error( train_label_, volume_la_ ) 
            resultado_loss = sess.run(loss)
            print(resultado_loss)
            tf.summary.histogram("resultado_loss", resultado_loss)

            train_step = optimizer.minimize(loss)
            #train_step =  tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=LEARNING_RATE, optimizer='Adam', summaries=["gradients"])
            #train_step = tf.contrib.slim.learning.create_train_op(loss, optimizer, summarize_gradients=True)

            sess.run(train_step)            

            train_writer.add_summary(summary, i)


        log_dir = "/home/davi/Documentos/train/test_model_0/"
        write_png = png.writeSlice( m.test_vol[0], log_dir + 'slice_label_' + str(i) + '.png' )
        sess.run(write_png)
        
    exit()







    save_path = '/home/davi/Documentos/train/model_%d/' % 0

    geom, angles = projtable.read( DATA_P + 'projMat.txt' )
    reconstructor = ct.Reconstructor(
            CONF_LA, angles[0:15], DISPLACEMENT,
            trainable = True,
            name = 'LAReconstructor',
            weights_type = WEIGHTS_TYPE
            )

    volume_la = reconstructor.apply( train_proj, geom )



# GO GO GO.. :)
#-------------------------------------------------------------------------------------------
if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group( required = True )
    group.add_argument( "--train", action="store_true" )
    group.add_argument( "--dataset", action="store_true" )
    group.add_argument( "--MyTrain", action="store_true" )
    group.add_argument( "--mano", action="store_true" )
    group.add_argument( "--mano2", action="store_true" )

    group.add_argument( "--test", action="store_true" )
    rgroup = parser.add_mutually_exclusive_group()
    segroup = rgroup.add_argument_group()
    segroup.add_argument( "--start", type=int, default = 0 )
    segroup.add_argument( "--end", type=int, default = -1 )
    segroup.add_argument( "--dataset_dir", action='store')

    rgroup.add_argument( "--only", type=int )
    parser.add_argument( "--resume", action="store_true" )
    args = parser.parse_args()

    print( 'Writing result to %s' % LOG_DIR )

    print( 'Check if all projections have corresponding labels..' )
    update_labels()

    if args.only is not None:
        start = args.only
        end = args.only + 1
    else:
        start = args.start
        end = args.end if args.end > 0 else len( PROJ_FILES )

    if args.dataset:
        
        print(args.dataset_dir)
        if args.dataset_dir == None:
            print("Arguments incorrect. Usage: python tfcone/pipeline.py --dataset --dataset_dir path_to_dicom_file ")
            exit()

        print("\n***** Getting .dcm projections, doing the Beer's Law Transformation, and puting the result in appropriate format *****\nMake this only one time.\n")

        # Copiar as amostras
        # First, copy the projections  
        dir_samples = os.path.join(args.dataset_dir, 'wNoise')
        dir_labels = os.path.join(args.dataset_dir, 'noNoise')
        destination_dir = os.path.join(current_dir, '../../phantoms/lowdose/')
        print(destination_dir)

        dataset_format.transform(dir_samples, destination_dir)

        # Copiar os labels

    if args.MyTrain:
        # NÃO TEM CROSS VALIDATION PQ É MT CUSTOSO
        # TEMOS APENAS 1 CONJUNTO DE TREINAMENTO, 1 DE TESTE E 1 DE VALIDAÇÃO
        my_train()

    # TRAIN
    if args.train:
        for i in range( start, end ):
            print( 'Start training model %d' % (i) )
            print( 'Leaving %s for test purposes..' % PROJ_FILES[i] )

            save_path = LOG_DIR + ( 'model_%d/' %i )
            #save_path = LOG_DIR + ( 'model_2/')

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            l, s = train_model( i,
                    save_path = save_path,
                    resume = args.resume
                )

            print("\n\n************* Saindo... Não vamos fazer Cross Validation. **********")
            exit()

    # TEST
    if args.test:

        for i in range( start, end ):
            print( 'Testing model %d' % i )
            test_proj = PROJ_FILES[i]
            test_label = VOL_FILES[i]

            print( 'Writing test volumes for %s' % test_proj )
            write_test_volumes( test_proj, test_label )

            _, _, validation_proj, validation_label = split_train_validation_set( i )

            save_path = LOG_DIR + ( 'test_model_%d/' %0 )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            test_model( validation_proj, validation_label, test_proj, test_label, '/home/davi/Documentos/train/model_%d/' % 0, save_path, i )
            #exit()


    if args.mano:
        for i in range( start, end ):

            test_proj = PROJ_FILES[i]
            test_label = VOL_FILES[i]
            save_path = LOG_DIR + ( 'test_model_%d/' % 0 )
            _, _, validation_proj, validation_label = split_train_validation_set( 0 )
            minha_funcao_pessoal(validation_proj, validation_label, test_proj, test_label, '/home/davi/Documentos/train/model_%d/' % 0, save_path, i)

    if args.mano2:
        test_proj = PROJ_FILES[0]
        test_label = VOL_FILES[0]
        save_path = LOG_DIR + ( 'test_model_%d/' % 0 )        
        minha_funcao_pessoal2()
