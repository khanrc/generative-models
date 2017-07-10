# DCGAN

## ToDo

* Input pipeline
    1. Read from files using QueueRunner and Coordinator
    2. Use TFRecords
* Apply DCGAN to multiple dataset
    * MNIST
    * CelebA
    * Flower?
    * LSUN?
* References
    * https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
        * just data_downloader for LSUN, CelebA, MNIST
    * https://github.com/nmhkahn/DCGAN-tensorflow-slim/blob/master/dataset/download_and_convert.py
        * data download and convert to TFRecords for CelebA, Flower


* B-e-a-utiful tensorboard
    * Try everything possible
* Advanced:
    * theoretical
        * WGAN
        * LSGAN?
        * WGAN-GP
        * BEGAN
        * CramerGAN
    * UDT
        * DiscoGAN / CycleGAN
        * DistanceGAN
        * DTN
* Misc
    * make gif?
        * how? ilguyi code is not working (just generate images in each step)
    * add progress bar
        * tqdm