# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys
sys.path.insert(0, '../')
from utils import *
from lsgan import LSGAN
from bgan import BGAN

slim = tf.contrib.slim
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


# hyperparams
z_dim = 128
# a, b, c = -1., 1., 0. # pearson chi-square divergence
# a, b, c = 0., 1., 1. #intuitive way


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def train(model):
    batch_size = 128
    n_iter = 100000
    print_step = 1000
    op_dir = 'out_' + model.name
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iter):
        X_batch, _ = mnist.train.next_batch(batch_size)
        z_batch = sample_z(batch_size, z_dim)
        _, _, D_loss_cur, G_loss_cur = sess.run([model.D_train_op, model.G_train_op, model.D_loss, model.G_loss], 
                                                {model.X: X_batch, model.z: z_batch, model.training: True})
        
        if i % print_step == 0 or i == n_iter-1:
            print('[{}/{}] D_loss: {:.4f}, G_loss: {:.4f}'.format(i, n_iter, D_loss_cur, G_loss_cur))
            
            # plotting
            z_ = sample_z(16, z_dim)
            samples = sess.run(model.fake, {model.z: z_, model.training: False})
            fig = plot(samples)
            c = int((i+1) / print_step)
            fig.savefig('{}/{}_{:0>4d}.png'.format(op_dir, model.name, c), bbox_inches='tight')
            plt.close(fig)

if __name__ == "__main__":
    # lsgan = LSGAN('LSGAN2', a=0., b=1., c=1., z_dim=z_dim)
    # train(lsgan)
    
    bgan = BGAN('BGAN', z_dim=z_dim)
    train(bgan)