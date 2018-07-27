import os
import warnings
import numpy as np
import skimage.io as ski_io
from keras.layers import Input, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from .models import _DisBuilder
from .helpers import RandomMaskGenerator
from .envs import (
    PJ, data_dir, ckpt_dir, evaluate_dir,
    discriminator_loss,
    pretrained_global
)

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(evaluate_dir, exist_ok=True)


def training(x_train, x_test=None, init_iters=1,
             eval_iters=-1, ckpt_iters=-1, max_iters=-1,
             pretrained_file=pretrained_global):

    input_tensor = Input(shape=(256, 256, 3), name='raw_image')
    color_prior = np.asarray([128, 128, 128], dtype=np.float) / 255.0

    discriminator_builder = _DisBuilder(activation='relu', metrics=['acc'],
                                        loss=discriminator_loss, debug=True,
                                        pretrained_file=pretrained_file)

    # Default config
    global_net = discriminator_builder(input_tensor,
                                       conv_n_filters=[64, 128, 256, 512, 512, 512],
                                       conv_kernel_sizes=[5, 5, 5, 5, 5, 5],
                                       conv_strides=[2, 2, 2, 2, 2, 2],
                                       out_n_filter=1024,
                                       tag='glcic_global_discriminator')

    # Test for shallow net
    # global_net = discriminator_builder(input_tensor, tag='glcic_global_discriminator')

    h = global_net(input_tensor)
    output_tensor = Dense(1, activation='sigmoid')(h)

    discriminator_net = discriminator_builder.compile(Model(input_tensor, output_tensor))

    batch_size = min(x_train.shape[0], 32)
    datagan = ImageDataGenerator(rotation_range=20, horizontal_flip=True,
                                 fill_mode='reflect',
                                 width_shift_range=0.2, height_shift_range=0.2)
    maskgan = RandomMaskGenerator(mask_size=(256, 256), box_size=(128, 128),
                                  max_size=(96, 96), min_size=(16, 16))

    # Suppress trainable weights and collected trainable inconsistency warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        for i, (real_images, (masks, _)) in enumerate(zip(datagan.flow(x_train, batch_size=batch_size),
                                                          maskgan.flow(batch_size=batch_size)), init_iters):
            fake_images = real_images * (1.0 - masks) + masks * color_prior
            images = np.concatenate((fake_images, real_images), axis=0)
            labels = np.asarray([[lb] for lb in ([0] * fake_images.shape[0] + [1] * real_images.shape[0])])

            # ('loss', 'acc')
            d_loss, acc = discriminator_net.train_on_batch(images, labels)

            print(f'Iter: {i:05},\t Loss: {d_loss:.3E}, Accuracy: {acc:2f}', flush=True)

            if eval_iters > 0 and i % eval_iters == 0:
                eval_image = np.concatenate(
                    (fake_images[0, ...],
                     np.repeat(masks[0, ...], 3, axis=2),
                     real_images[0, ...]),
                    axis=1
                )
                ski_io.imsave(PJ(evaluate_dir, f'eval_{i:05}.jpg'), eval_image, quality=100)

            if ckpt_iters > 0 and i % ckpt_iters == 0:
                print('Saving global_net.')
                global_net.save(pretrained_global)

            if max_iters > 0 and i >= max_iters:
                break


if __name__ == '__main__':
    input_image = ski_io.imread(PJ(data_dir, 'food.jpg')).astype(np.float) / 255.0
    x_train = np.asarray([input_image])
    training(x_train, x_test=x_train, max_iters=100)
    print('Accuracy should converge to 1.')
