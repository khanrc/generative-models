# coding: utf-8

import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../')
from utils import *

slim = tf.contrib.slim

def lrelu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class BGAN(object):
    # LSGAN stability test architecture
    def generator(self, z, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose], 
                                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params,
                                activation_fn=tf.nn.relu):
                net = z
                net = slim.fully_connected(net, 1024)
                net = slim.fully_connected(net, 7*7*128)
                net = tf.reshape(net, [-1, 7, 7, 128])
                net = slim.conv2d_transpose(net, 128, [5,5], stride=2)
                net = slim.conv2d_transpose(net, 1, [5,5], stride=2, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                expected_shape(net, [None, 28, 28, 1])
                
                return net

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params,
                                activation_fn=lrelu):
                net = tf.reshape(x, [-1, 28, 28, 1])
                net = slim.conv2d(net, 64, [5,5], stride=2, normalizer_fn=None)
                net = slim.conv2d(net, 128, [5,5], stride=2)
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1024)
                d_logits = slim.fully_connected(net, 1, activation_fn=None, normalizer_fn=None)
                d_prob = tf.nn.sigmoid(d_logits)
                expected_shape(d_logits, [None, 1])
                
                return d_logits, d_prob


    def __init__(self, name='BGAN', z_dim=64):
        self.name = name

        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 784])
            self.z = tf.placeholder(tf.float32, [None, z_dim])
            self.training = tf.placeholder(tf.bool)

            self.bn_params = {
                'is_training': self.training,
                'scale': True, 
                'decay': 0.99
            }

            self.fake = self.generator(self.z)
            D_real_logits, D_real_prob = self.discriminator(self.X)
            D_fake_logits, D_fake_prob = self.discriminator(self.fake, reuse=True)

            self.D_loss = -tf.reduce_mean(tf.log(D_real_prob + 1e-8) + tf.log(1 - D_fake_prob + 1e-8)) # - for minimize
            self.G_loss = 0.5 * tf.reduce_mean(tf.square(tf.log(D_fake_prob + 1e-8) - tf.log(1 - D_fake_prob + 1e-8)))

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/discriminator")
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/generator")

            self.D_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.D_loss, var_list=D_vars)
            self.G_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(self.G_loss, var_list=G_vars)
