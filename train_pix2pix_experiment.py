import os
import sys
import argparse

sys.path.insert(0, '../../')

import tensorflow as tf
import numpy as np
from model import Pix2Pix
import scipy.misc
from data_factory.dataset_factory import ImageCollector

parser = argparse.ArgumentParser(description='Easy Implementation of Pix2Pix + Test with graspgan dataset')

# parameters
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--batch_size', type=int, default=1)
HEIGHT = 256
WIDTH = 256

# Function for save the generated result
def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    nh, nw = nh_nw
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    scipy.misc.imsave(save_path, img)


def main():
    args = parser.parse_args()
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    batch_size = args.batch_size

    sess = tf.Session()
    model = Pix2Pix(sess, batch_size)

    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    input_img = tf.placeholder(tf.float32, [batch_size] + [472, 472, 3])
    target_img = tf.placeholder(tf.float32, [batch_size] + [WIDTH, HEIGHT, 3])
    real_dataset = ImageCollector("../../../new_env_dataset", 1, 64, batch_size)  # Real data
    simul_dataset = ImageCollector("../../../simul_dataset__", 1, 64, batch_size)

    #########################
    # tensorboard summary   #
    #########################
    dstep_loss = tf.placeholder(tf.float32)
    gstep_loss = tf.placeholder(tf.float32)
    image_shaped_input = tf.reshape(tf.image.resize_images(input_img, [WIDTH, HEIGHT]), [-1, WIDTH, HEIGHT, 3])
    image_shaped_output = tf.reshape(target_img, [-1, WIDTH, HEIGHT, 3])

    tf.summary.scalar('d_step_loss', dstep_loss)
    tf.summary.scalar('g_step_loss', gstep_loss)

    tf.summary.image('input / output', tf.concat([image_shaped_input, image_shaped_output], axis=2), 3)
    summary_merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(args.out_dir, sess.graph)

    real_dataset.StartLoadData()
    simul_dataset.StartLoadData()

    iter = 100000
    for i in range(iter):
        real_data = real_dataset.getLoadedData()
        simul_data = simul_dataset.getLoadedData()

        rgb_img = real_data[1]
        simul_img = simul_data[1]

        loss_D = model.train_discrim(simul_img, rgb_img)  # Train Discriminator and get the loss value
        loss_GAN = model.train_gen(simul_img, rgb_img)  # Train Generator and get the loss value

        if i % 100 == 0:
            print('Step: [', i, '/', iter, '], D_loss: ', loss_D, ', G_loss_GAN: ', loss_GAN)

        if i % 500 == 0:
            generated_samples = model.sample_generator(simul_img, batch_size=batch_size)

            summary = sess.run(summary_merged, feed_dict={dstep_loss: loss_D,
                                                          gstep_loss: loss_GAN,
                                                          input_img: simul_img,
                                                          target_img: generated_samples})
            writer.add_summary(summary, i)

        if i % 5000 == 0:
            saver.save(sess, ckpt_dir + '/model_iter' + str(i))


if __name__ == "__main__":
    main()
