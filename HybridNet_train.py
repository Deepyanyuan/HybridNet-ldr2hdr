# coding:utf-8

"""
train our Hybrid network
"""
import tensorflow as tf
import tensorlayer as tl
import glob, sys, os, datetime
import numpy as np
import cv2
import network
import datetime
import random
import img_io

eps = 1e-5
def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()

# Settings, using TensorFlow arguments
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("width", "256", "Reconstruction image width")
tf.flags.DEFINE_integer("height", "256", "Reconstruction image height")
tf.flags.DEFINE_integer("learn_type", "1", "Learning type. (0:downexposure, 1:upexposure")
tf.flags.DEFINE_string("im_dir", "./training_samples", "Path to image directory or an individual image")
tf.flags.DEFINE_string("out_dir", "out", "Path to output directory")
tf.flags.DEFINE_string("dm", "./models_dm", "Path to trained CNN dm_weights")
tf.flags.DEFINE_string("um", "./models_um", "Path to trained CNN um_weights")
tf.flags.DEFINE_float("scaling", "1.0", "Pre-scaling, which is followed by clipping, in order to remove compression artifacts close to highlights")
tf.flags.DEFINE_float("gamma", "1.0", "Gamma/exponential curve applied before, and inverted after, prediction. This can be used to control the boost of reconstructed pixels.")

min_size = 256
sx = int(np.maximum(min_size, np.round(FLAGS.width/256.0)*min_size))
sy = int(np.maximum(min_size, np.round(FLAGS.height/256.0)*min_size))
if sx != FLAGS.width or sy != FLAGS.height:
    print_("Warning: ", 'w', True)
    print_("prediction size has been changed from %dx%d pixels to %dx%d\n"%(FLAGS.width, FLAGS.height, sx, sy), 'w')
    print_("         pixels, to comply with autoencoder pooling and up-sampling.\n\n", 'w')

# Info
print_("\n\n\t-------------------------------------------------------------------\n", 'm')
print_("\t  HDR image reconstruction from a single exposure using deep CNNs\n\n", 'm')
print_("\t  Prediction settings\n", 'm')
print_("\t  -------------------\n", 'm')
print_("\t  Input image directory/file:         %s\n" % FLAGS.im_dir, 'm')
print_("\t  Output directory:                   %s\n" % FLAGS.out_dir, 'm')
print_("\t  Learning type:                      %s\n" % FLAGS.learn_type, 'm')
print_("\t  CNN dm_weights:                     %s\n" % FLAGS.dm, 'm')
print_("\t  CNN um_weights:                     %s\n" % FLAGS.um, 'm')
print_("\t  Prediction resolution:              %dx%d pixels\n" % (sx, sy), 'm')
if FLAGS.scaling > 1.0:
    print_("\t  Pre-scaling:                    %0.4f\n" % FLAGS.scaling, 'm')
if FLAGS.gamma > 1.0 + eps or FLAGS.gamma < 1.0 - eps:
    print_("\t  Gamma:                          %0.4f\n" % FLAGS.gamma, 'm')
print_("\t-------------------------------------------------------------------\n\n\n", 'm')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

batch_size = 1               
learning_rate = 0.0001      
maximum_epoch = 200         
predicted_window_len = 8    
norm_1_weiggt = 24
cosine_weight = 1           
step = 1            

is_upexposure_trained = int(FLAGS.learn_type)     
data_name = 'DML-HDR'                             
# data_name = 'Fairchild HDR'                     

dir_path_list = glob.glob(FLAGS.im_dir+'/'+ data_name +'/*')      
dir_path_list = dir_path_list[:]                                  
if is_upexposure_trained ==0:
    out_path_ = FLAGS.dm + '/' + data_name
    out_name = 'downexposure_model.npz'
    out_path = out_path_+ '/' + out_name             
else:
    out_path_ = FLAGS.um + '/' + data_name
    out_name = 'upexposure_model.npz'                                 
    out_path = out_path_ + '/' + out_name    

# create a mask
lossmask_list = list()
img_shape = (sy,sx,3)
for i in range(predicted_window_len):
    lossmask = np.ones(img_shape[0]*img_shape[1]*img_shape[2]).reshape((1,)+img_shape[:])    
    for j in range(predicted_window_len-1,0,-1):
        if i<j:
            append_img = np.ones(img_shape[0]*img_shape[1]*img_shape[2]).reshape((1,)+img_shape[:])      
        else:
            append_img = np.zeros(img_shape[0]*img_shape[1]*img_shape[2]).reshape((1,)+img_shape[:])
        lossmask = np.vstack([lossmask, append_img])            

    lossmask = np.broadcast_to(lossmask, (batch_size,)+lossmask.shape).astype(np.float32)   
    lossmask_list.append(lossmask)
lossmask_list = np.array(lossmask_list)       

# define placeholder
x = tf.placeholder(tf.float32, shape=[batch_size, sy, sx, 3], name='x')
x_local = tf.placeholder(tf.float32, shape=[batch_size, sy, sx, 3], name='x_local')
y_ = tf.placeholder(tf.float32, shape=[batch_size, None, 3], name='y_')
mask = tf.placeholder(tf.float32, shape=[batch_size, None, 3], name='mask')

# load model
model, _ = network.HybridNet(x, x_local, is_train=True, batch_size=batch_size, pad='SAME')
yyy = model.outputs      
y = yyy*mask
norm_y = tf.nn.l2_normalize(y, axis=2)
norm_y_ = tf.nn.l2_normalize(y_, axis=2)

# define cost function
cost = norm_1_weiggt*tl.cost.absolute_difference_error(y,y_,is_mean=True,name='absolute_difference_error_loss') + cosine_weight*tf.losses.cosine_distance(norm_y,norm_y_,axis=2) 
cost1 = norm_1_weiggt*tl.cost.absolute_difference_error(y,y_,is_mean=True,name='absolute_difference_error_loss')
cost2 = cosine_weight*tf.losses.cosine_distance(norm_y,norm_y_,axis=2)

# define optimizer
train_param = model.all_params
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)     

tf.add_to_collection('predict',yyy) 

saver = tf.train.Saver()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  
config.gpu_options.allow_growth = True   
sess = tf.Session(config = config)
init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(out_path_+'/')
if ckpt and ckpt.model_checkpoint_path:
    print('Load pre-model Done')
    saver.restore(sess,ckpt.model_checkpoint_path)

N = len(dir_path_list)
for epoch in range(maximum_epoch):
    print ('epoch',epoch)
    start = datetime.datetime.now()
    loss_gen_sum = 0.
    loss_gen_sum1 = 0.
    loss_gen_sum2 = 0.
    perm = np.random.permutation(N)       
    for i in range(N):
        dir_path = dir_path_list[perm[i]]                   
        if i%100 == 0:
            print('i',i)
            end = datetime.datetime.now()
        # read ground truth
        img_path_list_H = glob.glob(dir_path+'/HDR/1.hdr')           
        img_path_list = glob.glob(dir_path+'/LDR/*.png')             
        img_path_list.sort()
        img_list_H = list()
        img_list = list()

        if is_upexposure_trained:
            img_order = range(len(img_path_list))
        else:
            img_order = range(len(img_path_list)-1, -1, -1)

        img_H_ = img_io.readHDR(img_path_list_H[0], (sy,sx))
        img_list_H.append(np.squeeze(img_H_))
        img_list_H = np.array(img_list_H)                    

        for j in img_order:
            img_path = img_path_list[j]
            img = img_io.readLDR(img_path,(sy,sx), True, FLAGS.scaling)     
            img_list.append(np.squeeze(img))
        img_list = np.array(img_list)           

        for input_frame_id in range(len(img_list)-1):
            start_frame_id = input_frame_id+2
            end_frame_id = min(start_frame_id+predicted_window_len, len(img_list))
            x_batch = np.array([img_list[input_frame_id,:,:,:]])         
            y_batch_0 = img_list_H.reshape((1,)+x_batch.shape[:]).astype(np.float32)       
            y_batch_1 = np.array([img_list[start_frame_id:end_frame_id,:,:,:]])      
            y_batch = np.concatenate([y_batch_0, y_batch_1], axis= 1)                
            dummy_len = predicted_window_len-y_batch.shape[1]          
            zero_dummy = np.zeros(x_batch.size*dummy_len).reshape(y_batch.shape[:1]+(dummy_len,)+y_batch.shape[2:]).astype(np.float32)   
            y_batch = np.concatenate([y_batch, zero_dummy], axis=1)      
            y_batch = np.reshape(y_batch, (batch_size,-1,3))             
            lossmask = np.reshape(lossmask_list[dummy_len], (batch_size,-1,3))
                        
            _, loss_gen, loss_gen1, loss_gen2= sess.run([train_op, cost, cost1, cost2], feed_dict= {x: x_batch, x_local: x_batch, mask: lossmask, y_: y_batch})

            loss_gen_sum += loss_gen
            loss_gen_sum1 += loss_gen1
            loss_gen_sum2 += loss_gen2

    print ('loss:',loss_gen_sum/N/(len(img_list)-1))
    print ('loss1:',loss_gen_sum1/N/(len(img_list)-1))
    print ('loss2:',loss_gen_sum2/N/(len(img_list)-1))
    end = datetime.datetime.now()
    print('each train time is ',end-start)
    
    # save model and paras
    if epoch % step == 0:
        print('save model.npz')
        tl.files.save_npz(model.all_params, name=out_path)
        saver.save(sess, out_path, global_step=step)

sess.close()
