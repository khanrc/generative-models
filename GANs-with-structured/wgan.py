# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
based on DCGAN.

WGAN:
WD = max_f [ Ex[f(x)] - Ez[f(g(z))] ] where f has K-Lipschitz constraint
J = min WD (G_loss)
'''

class WGAN(BaseModel):
    def _build_train_graph(self):
        '''build computational graph for training
        '''
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            C_real = self._critic(X)
            C_fake = self._critic(G, reuse=True)

            # reduce_mean!
            W_dist = tf.reduce_mean(C_real - C_fake) # maximize
            C_loss = -W_dist # minimize
            G_loss = tf.reduce_mean(-C_fake)

            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/generator/')

            # 사실 C 는 n_critic 번 학습시켜줘야 하는데 귀찮아서 그냥 러닝레이트로 때려박음 
            # 학습횟수를 건드리려면 train.py 를 수정해야해서...
            n_critic = 5
            lr = 0.00005
            with tf.control_dependencies(C_update_ops):
                C_train_op = tf.train.RMSPropOptimizer(learning_rate=lr*n_critic).minimize(C_loss, var_list=C_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(G_loss, var_list=G_vars, global_step=global_step)

            # weight clipping
            '''
            이 때 웨이트 클리핑은 자동으로 실행이 안 되니 control dependency 를 설정해주거나
            group_op 로 묶어주거나 둘중 하나를 해야 할 듯
            Q. batch_norm parameter 도 clip 해줘야 하나?
            베타는 해 주는게 맞는 것 같은데, 감마는 좀 이상한데...? 
            => 대부분의 구현체들이 감마도 하고 있는 것 같음. 일단 해준다.
            '''
            # print 'C_vars: {}'.format(C_vars)
            # ver 1. 대부분의 구현체
            C_clips = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in C_vars] # with gamma
            # ver 2. 이건 안 됨
            # C_clips = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in C_vars if 'gamma' not in var.op.name] # without gamma

            # ver 3. 이건 되긴 하는데 흠... 잘모르겠음 
            # C_clips = []
            # for var in C_vars:
            #     if 'gamma' not in var.op.name:
            #         C_clips.append(tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)))
            #     else:
            #         C_clips.append(tf.assign(var, tf.clip_by_value(var, -1.00, 1.00)))

            print 'Weight clipping: {}'.format(C_clips)
            with tf.control_dependencies([C_train_op]): # should be iterable
            	C_train_op = tf.tuple(C_clips) # tf.group can be better ...

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('C_loss', C_loss),
                tf.summary.scalar('W_dist', W_dist)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=6)
            # tf.summary.histogram('real_probs', D_real_prob)
            # tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = C_train_op # train.py 와의 accesibility 를 위해... 흠... 구린데...
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _critic(self, X, reuse=False):
    	'''
    	K-Lipschitz function.
    	확인해봐야겠지만 K-Lipschitz function 을 근사하는 함수고, 
    	Lipschitz constraint 는 weight clipping 으로 걸어주니 
    	짐작컨대 그냥 linear 값을 추정하면 될 것 같음.
    	'''
        with tf.variable_scope('critic', reuse=reuse):
            net = X
            
            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=ops.lrelu, 
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d(net, 64, normalizer_fn=None)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])

            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, activation_fn=None)

            return net

    def _generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=tf.nn.relu, 
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])

                return net
