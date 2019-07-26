# coding:utf-8

import tensorflow as tf
import img_io
import sys, glob, os, cv2, time
import numpy as np
from merge_HDR import merge_HDR
import matplotlib.pyplot as plt 

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
tf.flags.DEFINE_string("im_dir", "testing_samples", "Path to image directory or an individual image")
tf.flags.DEFINE_string("out_dir", "results", "Path to output directory")
tf.flags.DEFINE_string("dm", "./models_dm", "Path to trained CNN dm_weights")
tf.flags.DEFINE_string("um", "./models_um", "Path to trained CNN um_weights")
tf.flags.DEFINE_float("scaling", "1.0",
                      "Pre-scaling, which is followed by clipping, in order to remove compression artifacts close to highlights")
tf.flags.DEFINE_float("gamma", "1.0",
                      "Gamma/exponential curve applied before, and inverted after, prediction. This can be used to control the boost of reconstructed pixels.")

sx = int(np.maximum(256, np.round(FLAGS.width / 256.0) * 256))
sy = int(np.maximum(256, np.round(FLAGS.height / 256.0) * 256))
if sx != FLAGS.width or sy != FLAGS.height:
    print_("Warning: ", 'w', True)
    print_("prediction size has been changed from %dx%d pixels to %dx%d\n" % (FLAGS.width, FLAGS.height, sx, sy), 'w')
    print_("         pixels, to comply with autoencoder pooling and up-sampling.\n\n", 'w')

# Info
print_("\n\n\t-------------------------------------------------------------------\n", 'm')
print_("\t  HDR image reconstruction from a single exposure using deep CNNs\n\n", 'm')
print_("\t  Prediction settings\n", 'm')
print_("\t  -------------------\n", 'm')
print_("\t  Input image directory/file:         %s\n" % FLAGS.im_dir, 'm')
print_("\t  Output directory:                   %s\n" % FLAGS.out_dir, 'm')
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
alpha = 0.6    

# data_name = 'DML-HDR'                                       
# data_name = 'Fairchild HDR'                                
# data_name = 'NewHDR'                                       
data_name = 'CanonCamera'                                    

dir_path_list = glob.glob(FLAGS.im_dir + '/' + data_name + '/*')
dir_path_list = dir_path_list[:]                         
dir_outpath = glob.glob(FLAGS.out_dir + '/' + data_name)


if not os.path.exists(FLAGS.out_dir + '/' + data_name):
    os.makedirs(FLAGS.out_dir + '/' + data_name)

# GPU set
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)

print_("\nStarting prediction...\n\n")

dm = tf.Graph()
um = tf.Graph()

sess_dm = tf.Session(graph=dm)
sess_um = tf.Session(graph=um)

with dm.as_default():
    ckpt_dm = tf.train.get_checkpoint_state(FLAGS.dm + '/' + data_name + '/')
    if ckpt_dm and ckpt_dm.model_checkpoint_path:
        saver_dm = tf.train.import_meta_graph(ckpt_dm.model_checkpoint_path + '.meta')    
        saver_dm.restore(sess_dm, ckpt_dm.model_checkpoint_path)                          
        pred_placehoder_dm = tf.get_collection('predict')[0]
        gragh_dm = tf.get_default_graph()
        x_dm = gragh_dm.get_tensor_by_name('x:0')
        x_local_dm = gragh_dm.get_tensor_by_name('x_local:0')

with um.as_default():
    ckpt_um = tf.train.get_checkpoint_state(FLAGS.um + '/' + data_name + '/')
    if ckpt_um and ckpt_um.model_checkpoint_path:
        saver_um = tf.train.import_meta_graph(ckpt_um.model_checkpoint_path + '.meta')
        saver_um.restore(sess_um, ckpt_um.model_checkpoint_path)
        pred_placehoder_um = tf.get_collection('predict')[0]
        gragh_um = tf.get_default_graph()
        x_um = gragh_um.get_tensor_by_name('x:0')
        x_local_um = gragh_um.get_tensor_by_name('x_local:0')

N = len(dir_path_list)
print('N.len', N)
for i in range(N):
    start_time = time.clock()
    dir_path = dir_path_list[i]
    # frams = [glob.glob(dir_path + '/LDR/1.png')[0], glob.glob(dir_path + '/LDR/4.png')[0], glob.glob(dir_path + '/LDR/7.png')[0]]
    frams = [glob.glob(dir_path + '/LDR/002.png')[0], glob.glob(dir_path + '/LDR/005.png')[0], glob.glob(dir_path + '/LDR/007.png')[0]]
    filename_root = os.path.basename(dir_path)
    print('filename', filename_root)
    save_path = dir_outpath[0] + '/' + filename_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        x_input_1 = img_io.readLDR(frams[0], (sy, sx), True, FLAGS.scaling)
        x_input_4 = img_io.readLDR(frams[1], (sy, sx), True, FLAGS.scaling)
        x_input_7 = img_io.readLDR(frams[2], (sy, sx), True, FLAGS.scaling)

        y_predict_dm_1 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_1 * 1, x_local_dm: x_input_1 * 1})
        y_predict_dm_4 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_4 * 1, x_local_dm: x_input_4 * 1})
        y_predict_dm_7 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_7 * 1, x_local_dm: x_input_7 * 1})

        y_Unet_predict_dm_1 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_1 * 1, x_local_dm: x_input_1 * 0})
        y_Unet_predict_dm_4 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_4 * 1, x_local_dm: x_input_4 * 0})
        y_Unet_predict_dm_7 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_7 * 1, x_local_dm: x_input_7 * 0})

        y_local_predict_dm_1 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_1 * 0, x_local_dm: x_input_1 * 1})
        y_local_predict_dm_4 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_4 * 0, x_local_dm: x_input_4 * 1})
        y_local_predict_dm_7 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_7 * 0, x_local_dm: x_input_7 * 1})

        y_predict_um_1 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_1 * 1, x_local_um: x_input_1 * 1})
        y_predict_um_4 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_4 * 1, x_local_um: x_input_4 * 1})
        y_predict_um_7 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_7 * 1, x_local_um: x_input_7 * 1})

        y_Unet_predict_um_1 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_1 * 1, x_local_um: x_input_1 * 0})
        y_Unet_predict_um_4 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_4 * 1, x_local_um: x_input_4 * 0})
        y_Unet_predict_um_7 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_7 * 1, x_local_um: x_input_7 * 0})

        y_local_predict_um_1 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_1 * 0, x_local_um: x_input_1 * 1})
        y_local_predict_um_4 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_4 * 0, x_local_um: x_input_4 * 1})
        y_local_predict_um_7 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_7 * 0, x_local_um: x_input_7 * 1})

        y_1, y_1_gamma, y_1_log, y_1_debevec = merge_HDR(y_predict_dm_1, y_predict_um_1, x_input_1, alpha=alpha)
        y_4, y_4_gamma, y_4_log, y_4_debevec = merge_HDR(y_predict_dm_4, y_predict_um_4, x_input_4, alpha=alpha)
        y_7, y_7_gamma, y_7_log, y_7_debevec = merge_HDR(y_predict_dm_7, y_predict_um_7, x_input_7, alpha=alpha)

        y_Unet_1, _, _, _ = merge_HDR(y_Unet_predict_dm_1, y_Unet_predict_um_1, x_input_1, alpha=alpha)
        y_Unet_4, _, _, _ = merge_HDR(y_Unet_predict_dm_4, y_Unet_predict_um_4, x_input_4, alpha=alpha)
        y_Unet_7, _, _, _ = merge_HDR(y_Unet_predict_dm_7, y_Unet_predict_um_7, x_input_7, alpha=alpha)

        y_local_1, _, _, _ = merge_HDR(y_local_predict_dm_1, y_local_predict_um_1, x_input_1, alpha=alpha)
        y_local_4, _, _, _ = merge_HDR(y_local_predict_dm_4, y_local_predict_um_4, x_input_4, alpha=alpha)
        y_local_7, _, _, _ = merge_HDR(y_local_predict_dm_7, y_local_predict_um_7, x_input_7, alpha=alpha)

        y_predict_1 = np.power(np.maximum(y_1, 0.0), FLAGS.gamma)
        y_predict_4 = np.power(np.maximum(y_4, 0.0), FLAGS.gamma)
        y_predict_7 = np.power(np.maximum(y_7, 0.0), FLAGS.gamma)

        img_io.writeEXR(y_predict_1, '%s/HDR_Hybrid_1.exr' % save_path)
        img_io.writeEXR(y_predict_4, '%s/HDR_Hybrid_4.exr' % save_path)
        img_io.writeEXR(y_predict_7, '%s/HDR_Hybrid_7.exr' % save_path)

        y_predict_1_gamma = np.power(np.maximum(y_1_gamma, 0.0), FLAGS.gamma)
        y_predict_4_gamma = np.power(np.maximum(y_4_gamma, 0.0), FLAGS.gamma)
        y_predict_7_gamma = np.power(np.maximum(y_7_gamma, 0.0), FLAGS.gamma)

        img_io.writeEXR(y_predict_1_gamma, '%s/HDR_Hybrid_gamma_1.exr' % save_path)
        img_io.writeEXR(y_predict_4_gamma, '%s/HDR_Hybrid_gamma_4.exr' % save_path)
        img_io.writeEXR(y_predict_7_gamma, '%s/HDR_Hybrid_gamma_7.exr' % save_path)

        y_predict_1_log = np.power(np.maximum(y_1_log, 0.0), FLAGS.gamma)
        y_predict_4_log = np.power(np.maximum(y_4_log, 0.0), FLAGS.gamma)
        y_predict_7_log = np.power(np.maximum(y_7_log, 0.0), FLAGS.gamma)

        img_io.writeEXR(y_predict_1_log, '%s/HDR_Hybrid_log_1.exr' % save_path)
        img_io.writeEXR(y_predict_4_log, '%s/HDR_Hybrid_log_4.exr' % save_path)
        img_io.writeEXR(y_predict_7_log, '%s/HDR_Hybrid_log_7.exr' % save_path)

        y_predict_1_debevec = np.power(np.maximum(y_1_debevec, 0.0), FLAGS.gamma)
        y_predict_4_debevec = np.power(np.maximum(y_4_debevec, 0.0), FLAGS.gamma)
        y_predict_7_debevec = np.power(np.maximum(y_7_debevec, 0.0), FLAGS.gamma)

        img_io.writeEXR(y_predict_1_debevec, '%s/HDR_Hybrid_debevec_1.exr' % save_path)
        img_io.writeEXR(y_predict_4_debevec, '%s/HDR_Hybrid_debevec_4.exr' % save_path)
        img_io.writeEXR(y_predict_7_debevec, '%s/HDR_Hybrid_debevec_7.exr' % save_path)

        y_Unet_predict_1 = np.power(np.maximum(y_Unet_1, 0.0), FLAGS.gamma)
        y_Unet_predict_4 = np.power(np.maximum(y_Unet_4, 0.0), FLAGS.gamma)
        y_Unet_predict_7 = np.power(np.maximum(y_Unet_7, 0.0), FLAGS.gamma)

        img_io.writeEXR(y_Unet_predict_1, '%s/HDR_Hybrid_Unet_1.exr' % save_path)
        img_io.writeEXR(y_Unet_predict_4, '%s/HDR_Hybrid_Unet_4.exr' % save_path)
        img_io.writeEXR(y_Unet_predict_7, '%s/HDR_Hybrid_Unet_7.exr' % save_path)

        y_local_predict_1 = np.power(np.maximum(y_local_1, 0.0), FLAGS.gamma)
        y_local_predict_4 = np.power(np.maximum(y_local_4, 0.0), FLAGS.gamma)
        y_local_predict_7 = np.power(np.maximum(y_local_7, 0.0), FLAGS.gamma)

        img_io.writeEXR(y_local_predict_1, '%s/HDR_Hybrid_Local_1.exr' % save_path)
        img_io.writeEXR(y_local_predict_4, '%s/HDR_Hybrid_Local_4.exr' % save_path)
        img_io.writeEXR(y_local_predict_7, '%s/HDR_Hybrid_Local_7.exr' % save_path)

        print_("\tdone\n")
        elapsed_time = (time.clock() - start_time)

    except img_io.IOException as e:
        print_("\n\t\tWarning! ", 'w', True)
        print_("%s\n" % e, 'w')
    except Exception as e:
        print_("\n\t\tError: ", 'e', True)
        print_("%s\n" % e, 'e')

print_("Done!\n")

