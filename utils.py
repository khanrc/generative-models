# check python `logging` module
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


warnings.simplefilter('error')


def expected_shape(tensor, expected):
    """batch size N is set to None. you can use shape instead of tensor.
    None indicates shape `?`.
    
    Usage:
    expected_shape(tensor, [None, 28, 28, 1])
    expected_shape(tensor.shape, [None, 28, 28, 1])
    """
    if isinstance(tensor, tf.Tensor):
        shape = tensor.shape
    else:
        shape = tensor
    shape = map(lambda x: x.value, shape)
    if not shape == expected:
        warnings.warn('wrong shape {} (expected shape is {})'.format(shape, expected))


def plot(samples, wh=4, figratio=0.75):
    """only for square-size samples
    wh = sqrt(samples.size)
    figratio: small-size = 0.75 (default) / big-size = 1.0
    """
    if len(samples) != wh*wh:
        print("Error: # of samples = {} but wh is {}".format(len(samples), wh))
        return
    
    figsize = wh * figratio
    fig = plt.figure(figsize=(figsize, figsize))
    gs = gridspec.GridSpec(wh, wh)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

