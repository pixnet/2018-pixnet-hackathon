import os
import random
import warnings

import numpy as np
import skimage.io as ski_io
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator

from .envs import (PJ, ckpt_dir, data_dir, discriminator_loss, evaluate_dir,
                   generator_loss, glcic_alpha)
from .helpers import FixMaskGenerator, RandomMaskGenerator
from .models import GLCICBuilder

os.makedirs(evaluate_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)


def evaluate(x_test, completion_net, w=32, h=32):
    maskgan = FixMaskGenerator()
    steps = [0, 32, 64, 96, 128]
    union_results = x_test[0, ...].copy()
    for y in steps:
        for x in steps:
            p = x + ((128 - w) >> 1)
            q = y + ((128 - h) >> 1)
            mask, _ = maskgan._draw(x=x, y=y, w=w, h=h)
            eval_images = completion_net.predict([x_test, np.asarray([mask])])
            union_results[q:q+h, p:p+w, :] = eval_images[0, q:q+h, p:p+w, :]

    return union_results







def training(x_train, x_test=None, init_iters=1,
             eval_iters=-1, ckpt_iters=-1, max_iters=-1,
             tc_prior=0.3, td_prior=0.6,
             pretrained_completion=PJ(ckpt_dir, 'completion.h5'),
             pretrained_local=PJ(ckpt_dir, 'local_discriminator.h5'),
             pretrained_global=PJ(ckpt_dir, 'global_discriminator.h5')):
    input_tensor = Input(shape=(256, 256, 3), name='raw_image')
    mask_tensor = Input(shape=(256, 256, 1), name='mask')
    bbox_tensor = Input(shape=(4,), dtype='int32', name='bounding_box')
    color_prior = np.asarray([128, 128, 128], dtype=np.float) / 255.0
    alpha = glcic_alpha

    d = float(tc_prior + td_prior)
    tc_prior, td_prior = (tc_prior/d, td_prior/d) if d > 0 else (0.0, 0.0)

    glcic = GLCICBuilder(activation='relu', optimizer='adam',
                         loss=[generator_loss, discriminator_loss])
    glcic_net = glcic.create(input_tensor, mask_tensor,
                             bbox_tensor, color_prior=color_prior,
                             pretrained_completion=pretrained_completion,
                             pretrained_local=pretrained_local,
                             pretrained_global=pretrained_global)

    glcic_net = glcic.compile(glcic_net, loss_weights=[1.0, alpha])
    completion_net = glcic.glcic_completion
    discriminator_net = glcic.glcic_discriminator
    local_discriminator = glcic.discriminator_builder.local_net
    global_discriminator = glcic.discriminator_builder.global_net

    # if x_test is not None:
    #     eval_images = K.get_value(glcic.completion_builder.preprocessing(x_test))
    #     ski_io.imsave(PJ(evaluate_dir, f'eval_00000.jpg'), eval_images[0, ...], quality=100)

    batch_size = x_train.shape[0]
    datagan = ImageDataGenerator(rotation_range=20, horizontal_flip=True,
                                 width_shift_range=0.2, height_shift_range=0.2)
    maskgan = RandomMaskGenerator(mask_size=(256, 256), box_size=(128, 128),
                                  max_size=(96, 96), min_size=(16, 16))

    g_loss, d_loss, acc = -1, -1, -1
    path = -1
    # Suppress trainable weights and collected trainable inconsistency warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        for i, (real_images, (masks, bboxes)) in enumerate(zip(datagan.flow(x_train, batch_size=batch_size),
                                                               maskgan.flow(batch_size=batch_size)), init_iters):
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            fake_images = real_images * (1.0 - masks) + masks * color_prior
            images = np.concatenate((fake_images, real_images), axis=0)
            labels = np.concatenate((fake, real), axis=0)
            concat_bboxes = np.concatenate((bboxes, bboxes), axis=0)

            dice = random.random()

            if dice <= tc_prior:
                # ['loss', 'mean_absolute_error']
                g_loss = completion_net.train_on_batch([real_images, masks], real_images)[0]
                path = 'completion'

            elif dice <= td_prior:
                # ['loss', 'acc']
                d_loss, acc = discriminator_net.train_on_batch([images, concat_bboxes], labels)
                path = 'discriminator'

            else:
                # ['loss', 'glcic_completion_loss', 'glcic_discriminator_loss']
                glcic_loss = glcic_net.train_on_batch([real_images, masks, bboxes], [real_images, real])
                g_loss = glcic_loss[1]
                d_loss = glcic_loss[2]
                path = 'glcic'

            print(f'Iter: {i:05} Path: {path}\
                    \tLosses: completion: {g_loss:.3E}, discriminator: {d_loss:.3E}\
                    \tAccuracy: {acc:.2f}', flush=True)
            if (eval_iters > 0) and (i % eval_iters == 0) and (x_test is not None):
                eval_image = evaluate(x_test, completion_net)

                ski_io.imsave(PJ(evaluate_dir, f'eval_{i:06}.jpg'), eval_image, quality=100)
            if ckpt_iters > 0 and i % ckpt_iters == 0:
                print('Saving completion_net...')
                completion_net.save(PJ(ckpt_dir, 'completion.h5'))
                print('Saving local_net...')
                local_discriminator.save(PJ(ckpt_dir, 'local_discriminator.h5'))
                print('Saving global_net...')
                global_discriminator.save(PJ(ckpt_dir, 'global_discriminator.h5'))

            if max_iters > 0 and i > max_iters:
                break


if __name__ == '__main__':
    eval_mask = list(RandomMaskGenerator().flow(total_size=1))[0][0]
    input_image = ski_io.imread(PJ(data_dir, 'food.jpg')).astype(np.float) / 255.0
    x_train = np.asarray([input_image])
    # x_test = [x_train, eval_mask]
    training(x_train, x_test=x_train, tc_prior=1.0, td_prior=0.0, max_iters=100)
