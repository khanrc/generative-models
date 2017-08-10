#coding: utf-8
import tensorflow as tf
import numpy as np
import utils
import config
import os, glob
import scipy.misc
from argparse import ArgumentParser
slim = tf.contrib.slim


def build_parser():
    parser = ArgumentParser()
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True) 
    parser.add_argument('--name', help='default: name=model')

    return parser


def sample_z(shape):
    return np.random.normal(size=shape)


def get_all_checkpoints(ckpt_dir, force=False):
    '''
    학습이 끊어졌다 재개되면 get_checkpoint_state 로는 모든 체크포인트를 가져올 수 없다 (재개된 시점부터 다시 기록됨).
    이걸 강제로 다 가져오는 함수 (when force=True)
    '''

    if force:
        ckpts = os.listdir(ckpt_dir) # get all fns
        ckpts = map(lambda p: os.path.splitext(p)[0], ckpts) # del ext => 중복 fn 생성됨
        ckpts = set(ckpts) # unique
        ckpts = filter(lambda x: x.split('-')[-1].isdigit(), ckpts) # ckpt 가 아닌것들 필터링
        ckpts = sorted(ckpts, key=lambda x: int(x.split('-')[-1])) # 정렬
        ckpts = map(lambda x: os.path.join(ckpt_dir, x), ckpts) # fn => path
    else:
        ckpts = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths
    
    return ckpts


def eval(model, name, sample_shape=[4,4], load_all_ckpt=True):
    if name == None:
        name = model.name
    dir_name = 'eval/' + name
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)

    # training=False => generator 만 생성
    restorer = tf.train.Saver(slim.get_model_variables())
    with tf.Session() as sess:
        ckpts = get_all_checkpoints('./checkpoints/' + name, force=load_all_ckpt)
        size = sample_shape[0] * sample_shape[1]

        z_ = sample_z([size, model.z_dim])

        for v in ckpts:
            print("Evaluating {} ...".format(v))
            restorer.restore(sess, v)
            global_step = int(v.split('/')[-1].split('-')[-1])
            
            fake_samples = sess.run(model.fake_sample, {model.z: z_})            

            # inverse transform: [-1, 1] => [0, 1]
            fake_samples = (fake_samples + 1.) / 2.
            merged_samples = utils.merge(fake_samples, size=sample_shape)
            fn = "{:0>5d}.png".format(global_step)
            scipy.misc.imsave(os.path.join(dir_name, fn), merged_samples)


'''
하지만 이렇게 말고도 그냥 imagemagick 을 통해 할 수 있다:
$ convert -delay 20 eval/* movie.gif

아래처럼 할꺼면 shading 효과를 넣어주면 좋을 듯 (convert 로는 하기 힘듦)
'''
# def to_gif(dir_name='eval'):
#     images = []
#     for path in glob.glob(os.path.join(dir_name, '*.png')):
#         im = scipy.misc.imread(path)
#         images.append(im)

#     # make_gif(images, dir_name + '/movie.gif', duration=10, true_image=True)
#     imageio.mimsave('movie.gif', images, duration=0.2)


if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model = FLAGS.model.upper()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)

    model = config.get_model(FLAGS.model, FLAGS.name, input_pipe=None)
    eval(model, name=FLAGS.name, sample_shape=[4,4], load_all_ckpt=True)
