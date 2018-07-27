import os

PJ = os.path.join
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = PJ(this_dir, 'data')
train_dir = PJ(data_dir, 'train')
evaluate_dir = PJ(this_dir, 'evaluate')
ckpt_dir = PJ(this_dir, 'ckpt')

generator_loss = 'mse'
discriminator_loss = 'binary_crossentropy'
glcic_alpha = 0.0004

pretrained_local = PJ(ckpt_dir, 'local_discriminator.h5')
pretrained_global = PJ(ckpt_dir, 'global_discriminator.h5')
pretrained_completion = PJ(ckpt_dir, 'completion.h5')
