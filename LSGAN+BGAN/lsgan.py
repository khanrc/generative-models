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

class LSGAN(object):
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
                d_prob = self.sigmoid_scaler(d_prob) # 사실 이 함수를 통과하면 prob 이 아니게 됨
                expected_shape(d_logits, [None, 1])
                
                return d_logits, d_prob

    def sigmoid_scaler(self, value):
        """a <= c <= b 라고 가정

        Caution:
        README 에도 써놨지만 이렇게 할게 아니라 sigmoid 를 애초에 쓰면 안 됨;
        """
        size = self.b - self.a
        return value * size + self.a


    def __init__(self, name='LSGAN', a=0., b=1., c=1., z_dim=64):
        """a: fake label
        b: real label
        c: real label for G (G가 D를 속이고자 하는 값 - 보통은 real label)

        Pearson chi-square divergence: a=-1, b=1, c=0.
        Intuitive (real label 1, fake label 0): a=0, b=c=1.
        check - 현재 D 의 output 이 sigmoid 이므로, a/b/c 값에 맞게 scale 되어야 함. 지금은 1/0 에 맞게 되어 있는것.
        """
        self.name = name
        self.a = a
        self.b = b
        self.c = c

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

            self.D_loss = 0.5 * tf.reduce_mean(tf.square(D_real_prob - b)) + 0.5 * tf.reduce_mean(tf.square(D_fake_prob - a))
            self.G_loss = 0.5 * tf.reduce_mean(tf.square(D_fake_prob - c))

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/discriminator")
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/generator")

            self.D_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.D_loss, var_list=D_vars)
            self.G_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(self.G_loss, var_list=G_vars)
