# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, '../')
from utils import *

slim = tf.contrib.slim
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# hyperparams
z_dim = 100
m = 5.
pt_weight = 5.
pt_reg = False


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


class EBGAN():
    def __init__(self, name, use_pt_regularizer=False):
        with tf.variable_scope(name):
            X = tf.placeholder(tf.float32, [None, 784])
            z = tf.placeholder(tf.float32, [None, z_dim])
            training = tf.placeholder(tf.bool)

            self.bn_params = {
                'is_training': training, 
                'scale': True, 
                'decay': 0.99
            }

            fake = self.generator(z)
            D_real_lf, D_real_mse = self.discriminator(X)
            D_fake_lf, D_fake_mse = self.discriminator(fake, reuse=True)
            expected_shape(D_real_mse, []) # scalar 
            expected_shape(D_fake_mse, []) # scalar

            D_fake_hinge = tf.maximum(0., m - D_fake_mse) # hinge_loss
            D_loss = D_real_mse + D_fake_hinge
            G_loss = D_fake_mse
            pt_loss = pt_weight * self.pt_regularizer(D_fake_lf) # pt_loss
            if use_pt_regularizer:
                G_loss += pt_loss

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/discriminator/")
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/generator/")

            D_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name+'/discriminator/')
            G_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name+'/generator/')

            with tf.control_dependencies(D_update):
                D_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update):
                G_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(G_loss, var_list=G_vars)

            # access points
            self.X = X
            self.z = z
            self.training = training
            self.D_loss = D_loss
            self.G_loss = G_loss
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake = fake
            self.pt_loss = pt_loss

    # energy function!
    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse): # auto-encoder
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d],
                                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params,
                                activation_fn=lrelu):
                # encoder
                x_img = tf.reshape(x, [-1, 28, 28, 1])
                net = x_img
                net = slim.conv2d(net, 64, [4,4], stride=2, normalizer_fn=None) # 14x14
                latent = slim.conv2d(net, 128, [4,4], stride=2) # 7x7
                # decoder
                net = slim.conv2d_transpose(latent, 64, [4, 4], stride=2) # 14x14
                x_recon = slim.conv2d_transpose(net, 1, [4, 4], stride=2, activation_fn=None, normalizer_fn=None) # 28x28
                mse = tf.losses.mean_squared_error(x_img, x_recon) # loss 를 mse 로 계산하므로 sigmoid 를 걸면 안되는 것 같음...
                
                return latent, mse

    def generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose], 
                                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params,
                                activation_fn=tf.nn.relu):
                net = z
                net = slim.fully_connected(net, 1024)
                net = slim.fully_connected(net, 7*7*128)
                net = tf.reshape(net, [-1, 7, 7, 128])
                net = slim.conv2d_transpose(net, 64, [4,4], stride=2)
                net = slim.conv2d_transpose(net, 1, [4,4], stride=2, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                expected_shape(net, [None, 28, 28, 1])
                
                return net

    # lf: latent features
    def pt_regularizer(self, lf):
        eps = 1e-8 # epsilon for numerical stability
        lf = slim.flatten(lf)
        # l2_norm = tf.sqrt(tf.reduce_sum(tf.square(lf), axis=1, keep_dims=True))
        l2_norm = tf.norm(lf, axis=1, keep_dims=True)
        expected_shape(l2_norm, [None, 1])
        unit_lf = lf / (l2_norm + eps) # this is unit vector?
        cos_sim = tf.square(tf.matmul(unit_lf, unit_lf, transpose_b=True)) # [N, h_dim] x [h_dim, N] = [N, N]
        N = tf.cast(tf.shape(lf)[0], tf.float32) # batch_size
        pt_loss = (tf.reduce_sum(cos_sim)-N) / (N*(N-1))
        return pt_loss


if __name__ == "__main__":
    # build nets
    tf.reset_default_graph()
    name = 'ebgan'
    if pt_reg:
        name += '-pt'
    ebgan = EBGAN(name, use_pt_regularizer=pt_reg)

    if tf.gfile.Exists('./out'):
        tf.gfile.DeleteRecursively('./out')
    tf.gfile.MkDir('./out')

    # train
    batch_size = 128
    n_iter = 1000000
    print_step = 1000

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = '2' # Works same as CUDA_VISIBLE_DEVICES!
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for i in range(n_iter):
        X_batch, _ = mnist.train.next_batch(batch_size)
        z_batch = sample_z(batch_size, z_dim)
        _, _, D_loss_cur1, G_loss_cur1, pt_loss_cur1 = sess.run([ebgan.D_train_op, ebgan.G_train_op, ebgan.D_loss, ebgan.G_loss, ebgan.pt_loss],
                                                                {ebgan.X: X_batch, ebgan.z: z_batch, ebgan.training: True})
        
        if i % print_step == 0 or i == n_iter-1:
            print('[{}/{}] D_loss: {:.4f}, G_loss: {:.4f}, pt_loss {:.4f}'.format(i, n_iter, D_loss_cur1, G_loss_cur1, pt_loss_cur1))
            z_ = sample_z(16, z_dim)
            samples1 = sess.run(ebgan.fake, {ebgan.z: z_, ebgan.training: False})
            fig1 = plot(samples1)
            c = int((i+1) / print_step)
            fig1.savefig('out/{}_{:0>4d}.png'.format(name,c), bbox_inches='tight')
            plt.close(fig1)

