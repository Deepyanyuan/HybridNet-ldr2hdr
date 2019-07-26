# coding:utf-8
'''
双分支网络，保持batch normal layer(或者pool层)，分辨率为256*256*3
'''

import tensorflow as tf
import tensorlayer as tl
import numpy as np

def HybridNet(x_in, x_local, is_train=False, batch_size=1, pad='SAME', reuse=None):
  
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("HybridNet"):        
        inputs = tl.layers.InputLayer(x_in, name='inputs')
        inputs_local = tl.layers.InputLayer(x_local, name='inputs_local')
        inputs_local.outputs = tf.expand_dims(inputs_local.outputs, 1)
    
        # encoder
        conv1 = tl.layers.Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        
        conv2 = tl.layers.Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')        
        conv2 = tl.layers.BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn2')
        
        conv3 = tl.layers.Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = tl.layers.BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn3')
        
        conv4 = tl.layers.Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = tl.layers.BatchNormLayer(conv4, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn4')
        
        conv5 = tl.layers.Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = tl.layers.BatchNormLayer(conv5, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn5')
        
        conv6 = tl.layers.Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = tl.layers.BatchNormLayer(conv6, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn6')
        
        conv7 = tl.layers.Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = tl.layers.BatchNormLayer(conv7, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name='bn7')
        
        conv8 = tl.layers.Conv2d(conv7, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        
        conv8_0 = conv8
        conv8_0.outputs = tf.expand_dims(conv8_0.outputs, 1)
        conv8_0 = tl.layers.TileLayer(conv8_0, [1,1,1,1,1], name='concat8_0')
        up8 = tl.layers.ConcatLayer([conv8, conv8_0], concat_dim=4, name='concat8')
        up8 = tl.layers.DeConv3d(up8, 512, filter_size=[4,4,4], strides=[2,2,2], padding=pad, W_init=w_init, b_init=b_init, name='deconv8')
        up8 = tl.layers.BatchNormLayer(up8, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn8')
        
        conv7_0 = conv7
        conv7_0.outputs = tf.expand_dims(conv7_0.outputs, 1)
        conv7_0 = tl.layers.TileLayer(conv7_0, [1,2,1,1,1], name='concat7_0')
        up7 = tl.layers.ConcatLayer([up8, conv7_0], concat_dim=4, name='concat7')
        up7 = tl.layers.DeConv3d(up7, 512, filter_size=[4,4,4], strides=[2,2,2], padding=pad, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = tl.layers.BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')
        
        conv6_0 = conv6
        conv6_0.outputs = tf.expand_dims(conv6_0.outputs, 1)
        conv6_0 = tl.layers.TileLayer(conv6_0, [1,4,1,1,1], name='concat6_0')
        up6 = tl.layers.ConcatLayer([up7, conv6_0], concat_dim=4, name='concat6')
        up6 = tl.layers.DeConv3d(up6, 512, filter_size=[4,4,4], strides=[2,2,2], padding=pad, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = tl.layers.BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        
        conv5_0 = conv5
        conv5_0.outputs = tf.expand_dims(conv5_0.outputs, 1)
        conv5_0 = tl.layers.TileLayer(conv5_0, [1,8,1,1,1], name='concat5_0')
        up5 = tl.layers.ConcatLayer([up6, conv5_0], concat_dim=4, name='concat5')
        up5 = tl.layers.DeConv3d(up5, 512, filter_size=[4,4,4], strides=[1,2,2], padding=pad, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = tl.layers.BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        
        conv4_0 = conv4
        conv4_0.outputs = tf.expand_dims(conv4_0.outputs, 1)
        conv4_0 = tl.layers.TileLayer(conv4_0, [1,8,1,1,1], name='concat4_0')
        up4 = tl.layers.ConcatLayer([up5, conv4_0], concat_dim=4, name='concat4')
        up4 = tl.layers.DeConv3d(up4, 256, filter_size=[4,4,4], strides=[1,2,2], padding=pad, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = tl.layers.BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        conv3_0 = conv3
        conv3_0.outputs = tf.expand_dims(conv3_0.outputs, 1)
        conv3_0 = tl.layers.TileLayer(conv3_0, [1,8,1,1,1], name='concat3_0')
        up3 = tl.layers.ConcatLayer([up4, conv3_0], concat_dim=4, name='concat3')
        up3 = tl.layers.DeConv3d(up3, 128, filter_size=[4,4,4], strides=[1,2,2], padding=pad, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = tl.layers.BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        conv2_0 = conv2
        conv2_0.outputs = tf.expand_dims(conv2_0.outputs, 1)
        conv2_0 = tl.layers.TileLayer(conv2_0, [1,8,1,1,1], name='concat2_0')
        up2 = tl.layers.ConcatLayer([up3, conv2_0], concat_dim=4, name='concat2')
        up2 = tl.layers.DeConv3d(up2, 64, filter_size=[4,4,4], strides=[1,2,2], padding=pad, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = tl.layers.BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        conv1_0 = conv1
        conv1_0.outputs = tf.expand_dims(conv1_0.outputs, 1)
        conv1_0 = tl.layers.TileLayer(conv1_0, [1,8,1,1,1],name='concat1_0')
        up1 = tl.layers.ConcatLayer([up2, conv1_0], concat_dim=4, name='concat1')
        up1 = tl.layers.DeConv3d(up1, 3, filter_size=[4,4,4], strides=[1,2,2], padding=pad, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = tl.layers.BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')
        
        conv0_0 = inputs
        conv0_0.outputs = tf.expand_dims(conv0_0.outputs, 1)
        conv0_0 = tl.layers.TileLayer(conv0_0, [1,8,1,1,1],name='concat0_0')
        up0 = tl.layers.ConcatLayer([up1, conv0_0], concat_dim=4, name='concat1')
        
        # local detail layers
        local_inputs = inputs_local
       
        local_conv1 = tl.layers.DeConv3d(local_inputs,64,filter_size=[4,4,4],strides=[2,1,1],padding=pad,W_init=w_init, b_init=b_init, name='local_conv1')

        local_conv2 = tl.layers.DeConv3d(local_conv1,64,filter_size=[4,4,4],strides=[2,1,1],padding=pad,W_init=w_init, b_init=b_init,name='local_conv2')
        local_conv2 = tl.layers.BatchNormLayer(local_conv2, act=tf.nn.leaky_relu, is_train=is_train,gamma_init=gamma_init,name='local_bn2')

        local_conv3 = tl.layers.DeConv3d(local_conv2,64,filter_size=[4,4,4],strides=[2,1,1],padding=pad,W_init=w_init, b_init=b_init,name='local_conv3')
        local_conv3 = tl.layers.BatchNormLayer(local_conv3, act=tf.nn.leaky_relu, is_train=is_train,gamma_init=gamma_init,name='local_bn3')

        local_conv4 = tl.layers.DeConv3d(local_conv3,64,filter_size=[4,4,4],strides=[1,1,1],padding=pad,W_init=w_init, b_init=b_init,name='local_conv4')
        local_conv4 = tl.layers.BatchNormLayer(local_conv4, act=tf.nn.leaky_relu, is_train=is_train,gamma_init=gamma_init,name='local_bn4')

        fusion1 = tl.layers.ConcatLayer([up0, local_conv4], concat_dim=4, name='fusion1')
        fusion1 = tl.layers.Conv3dLayer(fusion1, act=None,shape=[3,4,4,70,32], strides=[1, 1, 1, 1, 1], padding=pad, W_init=w_init, b_init=b_init, name='local_fusion1')
        fusion1 = tl.layers.BatchNormLayer(fusion1,act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='local_fusion_bn1')

        out = tl.layers.Conv3dLayer(fusion1, act=tf.nn.sigmoid,shape=[3,4,4,32,3], strides=[1, 1, 1, 1, 1], padding=pad, W_init=w_init, b_init=b_init, name='out')   # all networks outputs   
        
        out_ = out.outputs
        out_ = tf.reshape(out_, (batch_size,-1,3))
        out.outputs = out_
        out_data = tf.add(out.outputs,tf.constant(0,tf.float32),name='predict')
    return out, out_data

