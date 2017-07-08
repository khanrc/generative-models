# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
import sys, os
sys.path.insert(0, '../')
from utils import *
from discogan import DiscoGAN
from argparse import ArgumentParser

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# prepare dataset
def rotate(x):
    x = x.reshape(-1, 28, 28)
    x = scipy.ndimage.interpolation.rotate(x, 90, axes=(1, 2))
    return x.reshape(-1, 28*28)

N = mnist.train.num_examples/2
train_A = mnist.train.images[:N].reshape(-1, 28, 28, 1)
train_B = rotate(mnist.train.images[N:]).reshape(-1, 28, 28, 1)
# 100 test images for each domain
test_A = mnist.test.images[:100].reshape(-1, 28, 28, 1)
test_B = rotate(mnist.test.images[100:200]).reshape(-1, 28, 28, 1)

# A_dim = B_dim = 784


def train(model, n_epoch=500, batch_size=64, print_epoch=5):
    # train model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    op_dir = 'out_' + model.name
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)
    tb_dir = 'tmp/' + model.name
    writer = tf.summary.FileWriter(tb_dir, sess.graph, flush_secs=10)

    for epoch in range(n_epoch):
        np.random.shuffle(train_A)
        np.random.shuffle(train_B)
        for i in range(0, N, batch_size):
            batch_A = train_A[i:i+batch_size]
            batch_B = train_B[i:i+batch_size]

            _, G_loss_cur = sess.run([model.G_train_op, model.G_loss], {model.X_A: batch_A, model.X_B: batch_B, model.training: True})
            _, D_loss_cur, summary_scalar_cur = sess.run([model.D_train_op, model.D_loss, model.summary_scalar_op], 
                                                         {model.X_A: batch_A, model.X_B: batch_B, model.training: True})
            # _, _, G_loss_cur, D_loss_cur, summary_cur = sess.run([model.G_train_op, model.D_train_op, model.G_loss, model.D_loss, model.summary_op],
            #                                                      {model.X_A: batch_A, model.X_B: batch_B, model.training: True})
            writer.add_summary(summary_scalar_cur, epoch)


        if epoch % print_epoch == 0 or epoch == n_epoch-1:
            # logging
            print('[{}/{}] D_loss: {:.4f}, G_loss: {:.4f}'.format(epoch+1, n_epoch, D_loss_cur, G_loss_cur))
                
            # plotting
            w = 10 # the number of test images
            if model.oneway == False:
                samples_AB, samples_BA, summary_image_cur = sess.run([model.G_AB, model.G_BA, model.summary_image_op], 
                                                                     {model.X_A: test_A[:w], model.X_B: test_B[:w], model.training: False})

                samples = np.concatenate([test_A[:w], samples_AB, test_B[:w], samples_BA], axis=0)
                fig = plot(samples, shape=(4,w))
            else:
                samples_AB, summary_image_cur = sess.run([model.G_AB, model.summary_image_op], {model.X_A: test_A[:w], model.X_B: test_B[:w], model.training: False})
                samples = np.concatenate([test_A[:w], samples_AB, test_B[:w]], axis=0)
                fig = plot(samples, shape=(3,w))
            
            writer.add_summary(summary_image_cur, epoch)
            
            fig.savefig('{}/{:0>3d}.png'.format(op_dir, epoch), bbox_inches='tight', pad_inches=0.)
            plt.close(fig)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', default=500, help="Number of training epochs (default: 500)", type=int)
    parser.add_argument('--batch_size', default=64, help="Size of mini-batch (default: 64)", type=int)
    parser.add_argument('--print_step', default=5, help="Print and plot intermediate results for each steps (default: 5)", type=int)
    parser.add_argument('--oneway', action='store_true', default=False, help="Use only A -> B -> A (No cycle-consistency)")

    FLAGS = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    discogan = DiscoGAN(oneway=FLAGS.oneway, name='DiscoGAN')
    train(model=discogan, n_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, print_epoch=FLAGS.print_step)

