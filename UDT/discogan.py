# coding: utf-8

import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../')
from utils import *
slim = tf.contrib.slim

def slog(x):
    """safe log for tf"""
    return tf.log(x+1e-8)

def lrelu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


class DiscoGAN(object):
    # can be little different with original discogan
    def G(self, x, name, reuse=False):
        bn_params = {
            'is_training': self.training,
            # 'scale': True, 
            'decay': 0.99
        }
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
                                normalizer_fn=slim.batch_norm, normalizer_params=bn_params):
                # encoding
                with slim.arg_scope([slim.conv2d], activation_fn=lrelu):
                    # net = tf.reshape(x, [-1, 28, 28, 1])
                    net = slim.conv2d(x, 64, [4,4], stride=2, normalizer_fn=None)
                    net = slim.conv2d(net, 64, [4,4], stride=1)
                    net = slim.conv2d(net, 128, [4,4], stride=2)
                    net = slim.conv2d(net, 128, [4,4], stride=1)
                    expected_shape(net, [None, 7, 7, 128])

                # decoding
                with slim.arg_scope([slim.conv2d_transpose], activation_fn=tf.nn.relu):
                    net = slim.conv2d_transpose(net, 64, [4,4], stride=2) # 14x14x64 encoding
                    net = slim.conv2d_transpose(net, 64, [4,4], stride=1)
                    net = slim.conv2d_transpose(net, 1, [4,4], stride=2, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                    expected_shape(net, [None, 28, 28, 1])
                    # net = tf.reshape(net, [-1, 28*28])
                
                return net
        
    def D(self, x, name, reuse=False):
        bn_params = {
            'is_training': self.training,
            # 'scale': True, 
            'decay': 0.99
        }
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.fully_connected, slim.conv2d], 
                                normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                                activation_fn=lrelu):
                # net = tf.reshape(x, [-1, 28, 28, 1])
                net = slim.conv2d(x, 64, [4,4], stride=2, normalizer_fn=None)
                net = slim.conv2d(net, 128, [4,4], stride=2)
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1024)
                d_prob = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                expected_shape(d_prob, [None, 1])
                
                return d_prob

    # build nets
    def __init__(self, oneway=False, name='DiscoGAN'):
        if oneway:
            name = name + '-oneway'
        self.oneway = oneway
        self.name = name

        with tf.variable_scope(name):
            X_A = tf.placeholder(tf.float32, [None, 28, 28, 1])
            X_B = tf.placeholder(tf.float32, [None, 28, 28, 1])
            training = tf.placeholder(tf.bool)

            self.X_A = X_A
            self.X_B = X_B
            self.training = training

            # X_A = tf.reshape(X_A, [-1, 28, 28, 1])
            # X_B = tf.reshape(X_B, [-1, 28, 28, 1])

            # A -> B.
            ## GAN loss
            G_AB = self.G(X_A, 'G_AB')
            D_B_real = self.D(X_B, 'D_B')
            D_B_fake = self.D(G_AB, 'D_B', reuse=True)

            D_B_loss = -tf.reduce_mean(slog(D_B_real) + slog(1-D_B_fake)) # L_D_B
            G_AB_loss = -tf.reduce_mean(slog(D_B_fake)) # L_GAN_B

            ## recon loss
            G_ABA = self.G(G_AB, 'G_BA')
            # G_ABA_loss = tf.losses.mean_squared_error(G_ABA, X_A) # L_CONST_A
            G_ABA_loss = tf.reduce_mean(tf.reduce_sum((G_ABA-X_A)**2, axis=1))

            tf.summary.image('X_A', X_A, collections=['image'])
            tf.summary.image('X_B', X_B, collections=['image'])
            tf.summary.image('G_AB', G_AB, collections=['image'])
            tf.summary.image('G_ABA', G_ABA, collections=['image'])

            G_loss = G_AB_loss + G_ABA_loss
            D_loss = D_B_loss

            self.G_AB = G_AB

            # B -> A.
            if oneway == False: # True 면 B -> A 그래프 자체를 만들 필요가 없음
                G_BA = self.G(X_B, 'G_BA', reuse=True)
                D_A_real = self.D(X_A, 'D_A')
                D_A_fake = self.D(G_BA, 'D_A', reuse=True)

                D_A_loss = -tf.reduce_mean(slog(D_A_real) + slog(1-D_A_fake)) # L_D_A
                G_BA_loss = -tf.reduce_mean(slog(D_A_fake)) # L_GAN_A

                # ## recon loss
                G_BAB = self.G(G_BA, 'G_AB', reuse=True)
                # G_BAB_loss = tf.losses.mean_squared_error(G_BAB, X_B) # L_CONST_B
                G_BAB_loss = tf.reduce_mean(tf.reduce_sum((G_BAB-X_B)**2, axis=1))

                tf.summary.image('G_BA', G_BA, collections=['image'])
                tf.summary.image('G_BAB', G_BAB, collections=['image'])

                G_loss += G_BA_loss + G_BAB_loss
                D_loss += D_A_loss

                self.G_BA = G_BA


            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/G_')
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+'/D_')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name+'/G_')
            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name+'/D_')

            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(G_loss, var_list=G_vars)
            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss, var_list=D_vars)
            
            tf.summary.scalar('D_loss', D_loss, collections=['scalar'])
            tf.summary.scalar('G_loss', G_loss, collections=['scalar'])

            self.G_loss = G_loss
            self.D_loss = D_loss
            self.G_train_op = G_train_op
            self.D_train_op = D_train_op
            self.summary_scalar_op = tf.summary.merge_all('scalar')
            self.summary_image_op = tf.summary.merge_all('image')