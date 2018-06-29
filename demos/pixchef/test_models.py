import os
import warnings
import numpy as np
import skimage.io as ski_io
from keras.layers import Input
import keras.backend as K
from .envs import PJ, this_dir
from .models import GLCICBuilder
from .helpers import MaskGenerator


if __name__ == '__main__':
    input_image = ski_io.imread(PJ(this_dir, 'data/food.jpg')).astype(np.float) / 255.0
    input_masks, bboxes = list(MaskGenerator().flow(total_size=1))[0]
    color_prior = np.asarray([10, 255, 10], dtype=np.float) / 255.0

    temp_dir = 'temp'
    os.makedirs(PJ(this_dir, temp_dir), exist_ok=True)

    input_tensor = Input(shape=(256, 256, 3), name='raw_image')
    mask_tensor = Input(shape=(256, 256, 1), name='mask')
    bbox_tensor = Input(shape=(4,), dtype='int32', name='bounding_box')

    glcic = GLCICBuilder(activation='relu', loss=['mse', 'binary_crossentropy'])
    glcic_net = glcic(input_tensor, mask_tensor, bbox_tensor,
                      color_prior=color_prior, do_compile=True)

    # Suppress trainable weights and collected trainable inconsistency warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        glcic.nested_summary(glcic_net)

    completion_net = glcic.glcic_completion
    filled_image = K.get_value(
        glcic.completion_builder.preprocessing(
            [input_image, input_masks[0, ...]]
        )
    )

    completed_images = completion_net.predict([np.asarray([input_image]), input_masks])

    cropped_image = K.get_value(
        glcic.discriminator_builder.cropping(filled_image, bboxes[0, ...])
    )

    ski_io.imsave(PJ(this_dir, temp_dir, '0_reference.jpg'), input_image, quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '1_mask.jpg'), input_masks[0, :, :, 0], quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '2_preprocessed.jpg'), filled_image, quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '3_completed.jpg'), completed_images[0, ...], quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '4_cropped.jpg'), cropped_image, quality=100)
