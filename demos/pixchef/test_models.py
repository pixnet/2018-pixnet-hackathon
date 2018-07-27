import os
import warnings
import numpy as np
from skimage.feature import match_template
from skimage.measure import regionprops
import skimage.io as ski_io
from keras.layers import Input
import keras.backend as K
from .envs import PJ, this_dir
from .models import GLCICBuilder
from .helpers import FixMaskGenerator


temp_dir = 'temp'
os.makedirs(PJ(this_dir, temp_dir), exist_ok=True)


if __name__ == '__main__':
    input_image = ski_io.imread(PJ(this_dir, 'data/food.jpg')).astype(np.float) / 255.0
    input_mask, bbox = FixMaskGenerator()._draw(x=120, y=128, w=30, h=50)
    color_prior = np.asarray([10, 255, 10], dtype=np.float) / 255.0

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
            [input_image, input_mask]
        )
    )

    completed_images = completion_net.predict([np.asarray([input_image]), np.asarray([input_mask])])

    minr, minc, maxr, maxc = regionprops(input_mask.astype(int))[0].bbox
    template = input_image[minr:maxr, minc:maxc]
    attention_map = match_template(input_image, template, pad_input=True).sum(axis=2) * 0.33
    attention_map[attention_map < 0] = 0
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    attention_map[minr:maxr, minc:maxc] = 0
    attention_map[:bbox[1], :] = 0
    attention_map[:, :bbox[0]] = 0

    attention_map[bbox[1]+bbox[3]:, :] = 0
    attention_map[:, bbox[0]+bbox[2]:] = 0

    cropped_image = K.get_value(
        glcic.discriminator_builder.cropping(filled_image, bbox)
    )

    ski_io.imsave(PJ(this_dir, temp_dir, '0_reference.jpg'), input_image, quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '1_mask.jpg'), input_mask[..., 0], quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '2_preprocessed.jpg'), filled_image, quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '3_completed.jpg'), completed_images[0, ...], quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '4_attention.jpg'), attention_map, quality=100)
    ski_io.imsave(PJ(this_dir, temp_dir, '5_cropped.jpg'), cropped_image, quality=100)
