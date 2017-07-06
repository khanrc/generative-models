
# coding: utf-8

# In[1]:

# %matplotlib inline
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

slim = tf.contrib.slim


# In[2]:

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


# In[3]:

import sys
sys.path.insert(0, '../')
from utils import *


# In[4]:

# hyperparams
z_dim = 64
m = 5.


# In[5]:

def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        net = slim.fully_connected(z, 128)
        logits = slim.fully_connected(net, 784, activation_fn=None)
        prob = tf.nn.sigmoid(logits)
        
        return logits, prob


# In[6]:

# lf: latent features
def pt_regularizer(lf):
    l2_norm = tf.sqrt(tf.reduce_sum(tf.square(lf), axis=1, keep_dims=True))
    expected_shape(l2_norm, [None, 1])
    unit_lf = lf / l2_norm # this is unit vector?
    cos_sim = tf.square(tf.matmul(unit_lf, unit_lf, transpose_b=True)) # [N, h_dim] x [h_dim, N] = [N, N]
    N = tf.cast(tf.shape(lf)[0], tf.float32) # batch_size
    pt_loss = (tf.reduce_sum(cos_sim)-N) / (N*(N-1))
    return pt_loss


# In[7]:

def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse): # auto-encoder
        # latent features lf
        lf = slim.fully_connected(x, 128) # encoder
        x_recon = slim.fully_connected(lf, 784, activation_fn=None) # decoder
        mse = tf.losses.mean_squared_error(x, x_recon) # 데이터포인트마다 하는줄 알았는데 그냥 다합해서 하는건가봄
    
        return lf, mse


# In[8]:

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# In[9]:

class EBGAN():
    def __init__(self, name, use_pt_regularizer=False):
        with tf.variable_scope(name):
            X = tf.placeholder(tf.float32, [None, 784])
            z = tf.placeholder(tf.float32, [None, z_dim])

            fake_logits, fake = generator(z)
            D_real_lf, D_real_mse = discriminator(X)
            D_fake_lf, D_fake_mse = discriminator(fake, reuse=True)
            expected_shape(D_real_mse, []) # scalar
            expected_shape(D_fake_mse, []) # scalar

            D_fake_hinge = tf.reduce_mean(tf.maximum(0., m - D_fake_mse)) # hinge_loss
            D_loss = D_real_mse + D_fake_hinge
            G_loss = D_fake_mse
            if use_pt_regularizer:
                G_loss += pt_regularizer(D_fake_lf) # pt_loss

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/discriminator")
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/generator")

            D_train_op = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_vars)
            G_train_op = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_vars)

            self.X = X
            self.z = z
            self.D_loss = D_loss
            self.G_loss = G_loss
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake = fake


# In[10]:

# build nets
tf.reset_default_graph()

ebgan = EBGAN('ebgan')
reg_ebgan = EBGAN('ebgan-pt', use_pt_regularizer=True)


# In[12]:

batch_size = 128
n_iter = 1000000
print_step = 10000

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(n_iter):
    X_batch, _ = mnist.train.next_batch(batch_size)
    z_batch = sample_z(batch_size, z_dim)
    _, _, D_loss_cur1, G_loss_cur1 = sess.run([ebgan.D_train_op, ebgan.G_train_op, ebgan.D_loss, ebgan.G_loss],
                                              {ebgan.X: X_batch, ebgan.z: z_batch})
    _, _, D_loss_cur2, G_loss_cur2 = sess.run([reg_ebgan.D_train_op, reg_ebgan.G_train_op, reg_ebgan.D_loss, reg_ebgan.G_loss],
                                              {reg_ebgan.X: X_batch, reg_ebgan.z: z_batch})
    
    if i % print_step == 0 or i == n_iter-1:
        print('[{}/{}] (non-reg) D_loss: {:.4f}, G_loss: {:.4f} | (reg) D_loss: {:.4f}, G_loss: {:.4f}'.
              format(i, n_iter, D_loss_cur1, G_loss_cur1, D_loss_cur2, G_loss_cur2))
        z_ = sample_z(16, z_dim)
        samples1 = sess.run(ebgan.fake, {ebgan.z: z_})
        samples2 = sess.run(reg_ebgan.fake, {reg_ebgan.z: z_})
        fig1 = plot(samples1)
        fig2 = plot(samples2)
        c = int((i+1) / print_step)
        fig1.savefig('out/ebgan_{:0>4d}.png'.format(c), bbox_inches='tight')
        fig2.savefig('out/ebgan-pt_{:0>4d}.png'.format(c), bbox_inches='tight')
        plt.close(fig1)
        plt.close(fig2)

