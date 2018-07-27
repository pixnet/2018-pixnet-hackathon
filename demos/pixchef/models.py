import os
from itertools import chain
import warnings
import numpy as np

from keras.layers import (
    Lambda, Flatten, Activation,
    Conv2D, Conv2DTranspose,
    Dense, Input
)
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model
import keras.backend as K

import tensorflow as tf


PJ = os.path.join
this_dir = os.path.dirname(os.path.abspath(__file__))

Graph = Model


class GraphBuilder:
    __tag__ = 'abstract_graph'
    line_length = 98
    intent = 4

    def __init__(self, activation='sigmoid', dropout=0.1,
                 optimizer='sgd', loss=None, metrics=None,
                 pretrained_file=None, debug=False):
        self.activation = activation
        self.dropout = dropout
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.pretrained_file = pretrained_file
        self.debug = debug

    def __call__(self, *create_args, do_compile=False, **create_kwargs):
        model = self.create(*create_args, **create_kwargs)
        if self.pretrained_file is not None and os.path.isfile(self.pretrained_file):
            print(f'Loading pretrained model for `{self.__tag__}`')
            model.load_weights(self.pretrained_file)
        if do_compile:
            return self.compile(model)
        else:
            return model

    def _build_cell(self, cell_layer, n_filter, ksize, activation='sigmoid', **cell_kwargs):

        return [cell_layer(n_filter, kernel_size=ksize, **cell_kwargs),
                BatchNormalization(),
                Activation(activation)]

    def create(self):
        raise NotImplementedError

    def _print_row(self, fields, positions=[.33, .55, .67, 1.], recursive=0):
        line = ' ' * self.intent * recursive
        if positions[-1] <= 1:
            positions = [int(self.line_length * p) for p in positions]
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)

    def _print_layer_summary_with_connections(self, layer, relevant_nodes=[], recursive=0):
        """Prints a summary for a single layer.

        # Arguments
            layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        connections = []
        inbound_nodes = getattr(layer, 'inbound_nodes', None) or getattr(layer, '_inbound_nodes', None)
        for node in inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i].name
                inbound_node_index = node.node_indices[i]
                inbound_tensor_index = node.tensor_indices[i]
                connections.append(
                    inbound_layer + '[' + str(inbound_node_index) + '][' + str(inbound_tensor_index) + ']')

        name = layer.name
        cls_name = layer.__class__.__name__
        if not connections:
            first_connection = ''
        else:
            first_connection = connections[0]
        fields = [name + ' (' + cls_name + ')', output_shape,
                  layer.count_params(), first_connection]
        self._print_row(fields, recursive=recursive)
        if len(connections) > 1:
            for i in range(1, len(connections)):
                fields = ['', '', '', connections[i]]
                self._print_row(fields, recursive=recursive)

    def nested_summary(self, model, recursive=0):
        relevant_nodes = []
        layer_num = len(model.layers)
        header = ' ' * self.intent * recursive
        nodes_by_depth = getattr(model, 'nodes_by_depth', None) or getattr(model, '_nodes_by_depth', None)
        for v in nodes_by_depth.values():
            relevant_nodes += v
        print(header + ' ' * len(model.name))
        print(header + '* ' + model.name)
        print(header + '=' * self.line_length)
        for i, layer_or_model in enumerate(model.layers):
            if isinstance(layer_or_model, Model):
                self.nested_summary(layer_or_model, recursive=recursive+1)
            else:
                self._print_layer_summary_with_connections(
                    layer_or_model, relevant_nodes, recursive=recursive)
            if i == layer_num - 1:
                print(header + '=' * self.line_length)
            else:
                print(header + '_' * self.line_length)

        model._check_trainable_weights_consistency()
        if hasattr(model, '_collected_trainable_weights'):
            trainable_count = int(np.sum([K.count_params(p) for p in set(model._collected_trainable_weights)]))
        else:
            trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))

        non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

        print(header + 'Total params: {:,}'.format(trainable_count + non_trainable_count))
        print(header + 'Trainable params: {:,}'.format(trainable_count))
        print(header + 'Non-trainable params: {:,}'.format(non_trainable_count))
        print()  # print(header + '_' * self.line_length)

    def getnet(self, tag):
        try:
            return self.__getattribute__(tag)
        except AttributeError:
            raise AttributeError(f'Network "{self.__tag__}" has no subnet named "{tag}".')

    def compile(self, model, **kwargs):
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
            **kwargs
        )

        return model


class CompletionBuilder(GraphBuilder):
    """ Completion Network.
        -------------------------------------------------------------------
        Type                Kernel    Dilation (η)    Stride        Outputs
        -------------------------------------------------------------------
        conv.               5 x 5         1           1 x 1           64
        -------------------------------------------------------------------
        conv.               3 x 3         1           2 x 2           128
        conv.               3 x 3         1           1 x 1           128
        -------------------------------------------------------------------
        conv.               3 x 3         1           2 x 2           256
        conv.               3 x 3         1           1 x 1           256
        conv.               3 x 3         1           1 x 1           256
        dilated conv.       3 x 3         2           1 x 1           256
        dilated conv.       3 x 3         4           1 x 1           256
        dilated conv.       3 x 3         8           1 x 1           256
        dilated conv.       3 x 3         16          1 x 1           256
        conv.               3 x 3         1           1 x 1           256
        conv.               3 x 3         1           1 x 1           256
        -------------------------------------------------------------------
        deconv.             4 x 4         1           1/2 x 1/2       128
        conv.               3 x 3         1           1 x 1           128
        -------------------------------------------------------------------
        deconv.             4 x 4         1           1/2 x 1/2       64
        conv.               3 x 3         1           1 x 1           32
        output              3 x 3         1           1 x 1           3
        -------------------------------------------------------------------
    """

    __tag__ = 'glcic_completion'

    def __init__(self, color_prior, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_tensor = K.variable(color_prior)

    def preprocessing(self, x):
        image, mask = x
        return image * (1.0 - mask) + mask * self.prior_tensor

    def create(self, input_tensor, mask_tensor, padding='same'):

        # preprocessing stage
        filled_tensor = Lambda(
            self.preprocessing,
            output_shape=K.int_shape(input_tensor)[1:],
            name='preprocessing'
        )([input_tensor, mask_tensor])

        preprocessed_tensor = concatenate([filled_tensor, mask_tensor])

        # convolution stage
        conv_n_filters = [64, 128, 128, 256, 256, 256]
        conv_kernel_sizes = [5, 3, 3, 3, 3, 3]
        conv_strides = [1, 2, 1, 2, 1, 1]

        conv_layers = (
            self._build_cell(Conv2D,
                             n_filter,
                             ksize,
                             activation=self.activation,
                             strides=strides,
                             padding=padding)
            for n_filter, ksize, strides
            in zip(conv_n_filters, conv_kernel_sizes, conv_strides)
        )

        # dilation stage
        dilate_n_filters = [256, 256, 256, 256]
        dilate_kernel_sizes = [3, 3, 3, 3]
        dilate_strides = [1, 1, 1, 1]
        dilate_dilation_rate = [2, 4, 8, 16]
        dilate_layers = (
            self._build_cell(Conv2D,
                             n_filter,
                             ksize,
                             activation=self.activation,
                             strides=strides,
                             dilation_rate=dilation_rate,
                             padding=padding)
            for n_filter, ksize, strides, dilation_rate
            in zip(dilate_n_filters, dilate_kernel_sizes, dilate_strides, dilate_dilation_rate)
        )

        # additional convolution stage
        aconv_n_filters = [256, 256]
        aconv_kernel_sizes = [3, 3]
        aconv_strides = [1, 1]
        aconv_layers = (
            self._build_cell(Conv2D,
                             n_filter,
                             ksize,
                             activation=self.activation,
                             strides=strides,
                             padding=padding)
            for n_filter, ksize, strides
            in zip(aconv_n_filters, aconv_kernel_sizes, aconv_strides)
        )

        # deconvolution stage
        dconv_n_filters = [128, 128, 64, 32]
        dconv_kernel_sizes = [4, 3, 4, 3]
        dconv_strides = [2, 1, 2, 1]
        dconv_layers = (
            self._build_cell(Conv2DTranspose if i % 2 == 0 else Conv2D,
                             n_filter,
                             ksize,
                             activation=self.activation,
                             strides=strides,
                             padding=padding)
            for i, (n_filter, ksize, strides)
            in enumerate(zip(dconv_n_filters, dconv_kernel_sizes, dconv_strides))
        )

        # output stage
        out_n_filters = [3]
        out_kernel_sizes = [3]
        out_strides = [1]
        out_layers = (
            self._build_cell(Conv2D,
                             n_filter,
                             ksize,
                             activation='sigmoid',
                             strides=strides,
                             padding=padding)
            for n_filter, ksize, strides
            in zip(out_n_filters, out_kernel_sizes, out_strides)
        )

        def _double_chain(*ls):
            return chain.from_iterable(chain.from_iterable(ls))

        # pipe layers together
        all_layers = _double_chain(conv_layers, dilate_layers, aconv_layers, dconv_layers, out_layers)
        output_tensor = preprocessed_tensor

        debug_graphs = []
        for layer in all_layers:
            output_tensor = layer(output_tensor)
            if self.debug:
                debug_graphs.append(Graph(input_tensor, output_tensor, name=f'debug_{layer.name}'))

        setattr(self, f'debug_{self.__tag__}', debug_graphs)

        # completion stage
        completion_tensor = Lambda(
            lambda x: x[0] * x[2] + x[1] * (1 - x[2]),
            output_shape=K.int_shape(input_tensor)[1:],
            name='completion'
        )([output_tensor, input_tensor, mask_tensor])

        return Graph([input_tensor, mask_tensor], completion_tensor, name=self.__tag__)


class _DisBuilder(GraphBuilder):
    """ Local Discriminators Network.
        --------------------------------------------
        Type         Kernel    Stride        Outputs
        --------------------------------------------
        conv.        5 x 5     2 x 2          64
        conv.        5 x 5     2 x 2          128
        conv.        5 x 5     2 x 2          256
        conv.        5 x 5     2 x 2          512
        conv.        5 x 5     2 x 2          512
        --------------------------------------------
        FC             -         -            1024
        --------------------------------------------

        Global Discriminators Network.
        --------------------------------------------
        Type         Kernel    Stride        Outputs
        --------------------------------------------
        conv.        5 x 5     2 x 2          64
        conv.        5 x 5     2 x 2          128
        conv.        5 x 5     2 x 2          256
        conv.        5 x 5     2 x 2          512
        conv.        5 x 5     2 x 2          512
        conv.        5 x 5     2 x 2          512
        --------------------------------------------
        FC             -         -            1024
        --------------------------------------------
    """

    __tag__ = 'glcic_discriminator'

    def create(self, input_tensor,
               conv_n_filters=[8, 16, 32],
               conv_kernel_sizes=[5, 5, 5],
               conv_strides=[2, 2, 2],
               padding='same', tag='glcic_discriminator',
               out_n_filter=32, out_activation='relu'):

        self.__tag__ = tag if tag else self.__tag__

        conv_layers = (
            self._build_cell(Conv2D,
                             n_filter,
                             ksize,
                             activation=self.activation,
                             strides=strides,
                             padding=padding)
            for n_filter, ksize, strides
            in zip(conv_n_filters, conv_kernel_sizes, conv_strides)
        )

        out_layers = [Flatten(), Dense(out_n_filter, activation=out_activation)]

        debug_graphs = []
        output_tensor = input_tensor
        for layer in chain.from_iterable(conv_layers):
            output_tensor = layer(output_tensor)
            if self.debug:
                debug_graphs.append(Graph(input_tensor, output_tensor, name=f'debug_{layer.name}'))

        for layer in out_layers:
            output_tensor = layer(output_tensor)
            if self.debug:
                debug_graphs.append(Graph(input_tensor, output_tensor, name=f'debug_{layer.name}'))

        setattr(self, f'debug_{self.__tag__}', debug_graphs)

        return Graph(input_tensor, output_tensor, name=self.__tag__)


class DiscriminatorBuilder(_DisBuilder):

    def cropping(self, image, bbox):
        return tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3], bbox[2])

    def create(self, global_input_tensor, bbox_tensor,
               local_shape=(128, 128, 3),
               local_net=None, global_net=None,
               pretrained_local=None, pretrained_global=None):

        if global_net is None:
            global_builder = _DisBuilder(activation=self.activation,
                                         loss='binary_crossentropy',
                                         optimizer=self.optimizer,
                                         pretrained_file=pretrained_global,
                                         metrics=['acc'], debug=self.debug)
            global_net = global_builder(
                global_input_tensor,
                conv_n_filters=[64, 128, 256, 512, 512, 512],
                conv_kernel_sizes=[5, 5, 5, 5, 5, 5],
                conv_strides=[2, 2, 2, 2, 2, 2],
                out_n_filter=1024,
                tag='glcic_global_discriminator',
            )

        cropping_layer = Lambda(
            lambda x: K.map_fn(
                lambda z: self.cropping(z[0], z[1]),
                elems=x,
                dtype=tf.float32
            ),
            output_shape=local_shape,
            name='cropping'
        )

        cropped_tensor = cropping_layer([global_input_tensor, bbox_tensor])
        local_input_tensor = Input(shape=K.int_shape(cropped_tensor)[1:], name='local_roi')

        if local_net is None:
            local_builder = _DisBuilder(activation=self.activation,
                                        loss='binary_crossentropy',
                                        optimizer=self.optimizer,
                                        pretrained_file=pretrained_local,
                                        metrics=['acc'], debug=self.debug)
            local_net = local_builder(
                local_input_tensor,
                conv_n_filters=[64, 128, 256, 512, 512],
                conv_kernel_sizes=[5, 5, 5, 5, 5],
                conv_strides=[2, 2, 2, 2, 2],
                out_n_filter=1024,
                tag='glcic_local_discriminator',
            )

        global_output_tensor = global_net(global_input_tensor)
        local_output_tensor = local_net(cropped_tensor)

        y = concatenate([global_output_tensor, local_output_tensor])
        y = Dense(1, activation='sigmoid')(y)

        self.local_net = local_net
        self.global_net = global_net

        return Graph([global_input_tensor, bbox_tensor], y, name=self.__tag__)


class GLCICBuilder(GraphBuilder):
    '''
        Ref: http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/
    '''

    __tag__ = 'glcic'

    def create(self, input_tensor, mask_tensor, bbox_tensor,
               completion_net=None,
               pretrained_completion=None,
               global_net=None, local_net=None,
               pretrained_local=None, pretrained_global=None,
               color_prior=None):

        if completion_net is None:
            completion_builder = CompletionBuilder(color_prior,
                                                   activation=self.activation,
                                                   optimizer=self.optimizer,
                                                   loss='mse', metrics=['mae'],
                                                   pretrained_file=pretrained_completion,
                                                   debug=self.debug)

            completion_net = completion_builder(input_tensor, mask_tensor, do_compile=True)

        completion_output_tensor = completion_net([input_tensor, mask_tensor])

        global_input_tensor = Input(shape=K.int_shape(completion_output_tensor)[1:], name='completed_image')

        discriminator_builder = DiscriminatorBuilder(activation=self.activation,
                                                     optimizer='sgd',
                                                     loss='binary_crossentropy', metrics=['acc'],
                                                     debug=self.debug)
        discriminator_net = discriminator_builder(global_input_tensor, bbox_tensor,
                                                  global_net=global_net, local_net=local_net,
                                                  pretrained_local=pretrained_local,
                                                  pretrained_global=pretrained_global,
                                                  do_compile=True)
        discriminator_net.trainable = False

        discriminator_output_tensor = discriminator_net(
            [completion_output_tensor, bbox_tensor]
        )

        self.glcic_completion = completion_net
        self.glcic_discriminator = discriminator_net
        self.glcic_local_discriminator = discriminator_builder.local_net
        self.glcic_global_discriminator = discriminator_builder.global_net
        self.discriminator_builder = discriminator_builder
        self.completion_builder = completion_builder

        return Graph([input_tensor, mask_tensor, bbox_tensor],
                     [completion_output_tensor, discriminator_output_tensor],
                     name=self.__tag__)


if __name__ == '__main__':

    input_tensor = Input(shape=(256, 256, 3), name='raw_image')
    mask_tensor = Input(shape=(256, 256, 1), name='mask')
    bbox_tensor = Input(shape=(4,), dtype='int32', name='bounding_box')
    color_prior = np.asarray([128, 128, 20])
    alpha = 0.0004

    glcic = GLCICBuilder(activation='relu', loss=['mse', 'binary_crossentropy'])
    glcic_net = glcic.create(input_tensor, mask_tensor, bbox_tensor, color_prior=color_prior)
    glcic_net = glcic.compile(glcic_net, loss_weights=[1.0, alpha])

    # Suppress trainable weights and collected trainable inconsistency warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        glcic.nested_summary(glcic_net)

    completion_net = glcic.glcic_completion
    discriminator_net = glcic.glcic_discriminator
    glcic_local_discriminator_net = glcic.glcic_local_discriminator
    glcic_global_discriminator_net = glcic.glcic_global_discriminator

    graph_dir = 'graphs'
    os.makedirs(PJ(this_dir, graph_dir), exist_ok=True)

    plot_model(completion_net,
               to_file=PJ(this_dir, graph_dir, 'completion.pdf'),
               show_shapes=True)
    plot_model(discriminator_net,
               to_file=PJ(this_dir, graph_dir, 'discriminator.pdf'),
               show_shapes=True)
    plot_model(glcic_global_discriminator_net,
               to_file=PJ(this_dir, graph_dir, 'global_discriminator.pdf'),
               show_shapes=True)
    plot_model(glcic_local_discriminator_net,
               to_file=PJ(this_dir, graph_dir, 'local_discriminator.pdf'),
               show_shapes=True)
    plot_model(glcic_net,
               to_file=PJ(this_dir, graph_dir, 'glcic.pdf'),
               show_shapes=True)
