import os
import warnings
import numpy as np
import skimage.io as ski_io
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from .models import GLCICBuilder
from .helpers import MaskGenerator
from .envs import PJ, data_dir, evaluate_dir

os.makedirs(evaluate_dir, exist_ok=True)


def training(x_train, x_test=None, max_iters=1000, eval_iters=20, tc=100, td=50):
    input_tensor = Input(shape=(256, 256, 3), name='raw_image')
    mask_tensor = Input(shape=(256, 256, 1), name='mask')
    bbox_tensor = Input(shape=(4,), dtype='int32', name='bounding_box')
    color_prior = np.asarray([128, 128, 128], dtype=np.float) / 255.0
    alpha = 0.0004

    glcic = GLCICBuilder(activation='relu', loss=['mse', 'binary_crossentropy'])
    glcic_net = glcic.create(input_tensor, mask_tensor,
                             bbox_tensor, color_prior=color_prior)
    glcic_net = glcic.compile(glcic_net, loss_weights=[1.0, alpha])
    completion_net = glcic.glcic_completion
    discriminator_net = glcic.glcic_discriminator

    if x_test is not None:
        eval_images = K.get_value(glcic.completion_builder.preprocessing(x_test))
        ski_io.imsave(PJ(evaluate_dir, f'eval_00000.jpg'), eval_images[0, ...], quality=100)

    batch_size = x_train.shape[0]
    datagan = ImageDataGenerator(rotation_range=20, horizontal_flip=True,
                                 width_shift_range=0.2, height_shift_range=0.2)
    maskgan = MaskGenerator(mask_size=(256, 256), box_size=(128, 128), max_size=(96, 96), min_size=(16, 16))
    g_loss, d_loss = -1, -1
    # Suppress trainable weights and collected trainable inconsistency warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        for i, (images, (masks, bboxes)) in enumerate(zip(datagan.flow(x_train, batch_size=batch_size),
                                                          maskgan.flow(batch_size=batch_size)), 1):
            completed_images = completion_net.predict([images, masks])
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            if i <= tc:
                # ['loss', 'mean_absolute_error']
                g_loss = completion_net.train_on_batch([images, masks], images)[0]

            else:
                # ['loss', 'acc']
                d_loss_real = discriminator_net.train_on_batch([images, bboxes], real)[0]
                d_loss_fake = discriminator_net.train_on_batch([completed_images, bboxes], fake)[0]
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                if i > tc + td:
                    # ['loss', 'glcic_completion_loss', 'glcic_discriminator_loss']
                    glcic_loss = glcic_net.train_on_batch([images, masks, bboxes], [images, real])
                    g_loss = glcic_loss[1]
                    d_loss = glcic_loss[2]
            print(f'Iter: {i:05}\tLosses: generator: {g_loss:03.3f}, discriminator: {d_loss:03.3f}')

            if i % eval_iters == 0 and (x_test is not None):
                eval_images = completion_net.predict(x_test)
                ski_io.imsave(PJ(evaluate_dir, f'eval_{i:05}.jpg'), eval_images[0, ...], quality=100)
            if i > max_iters:
                break


if __name__ == '__main__':
    eval_mask = list(MaskGenerator().flow(total_size=1))[0][0]
    input_image = ski_io.imread(PJ(data_dir, 'food.jpg')).astype(np.float) / 255.0
    x_train = np.asarray([input_image])
    x_test = [x_train, eval_mask]
    training(x_train, x_test=x_test)
