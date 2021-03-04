from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.layers import Concatenate, Lambda, add
from keras.models import Model
from keras import backend as K
from functools import partial
from utils.img_utils import pre_process, l2_norm
import numpy as np
import time

def scaling(x, scale):
    return x * scale

def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))

    return '_'.join((prefix, 'Branch', str(branch_idx), name))

def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
    if not use_bias:
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name=_generate_layer_name('BatchNorm', prefix=name))(x)
    if activation is not None:
        x = Activation(activation, name=_generate_layer_name('Activation', prefix=name))(x)
    return x

def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    channel_axis = 3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))

    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == "Block35":
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == "Block17":
        branch_0 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 128, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 128, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == "Block8":
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 192, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 192, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)

    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name_fmt('Conv2d_1x1'))
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': scale})(up)

    x = add([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)

    return x

def _InceptionResNetV1(input_shape=(160,160,3), classes=128, keep_prob=0.8):
    """InceptionResNetV1 Network to convert (160,160,30) -> 128.
        # Arguments
            input_shape: tuple of 3 integers,
                (batch_size, height, width, channels)
            classes: Integer, output dimension.
            keep_prob: dropout keep_prob.
    """
    channel_axis = 3
    inputs = Input(shape=(input_shape))
    # 160,160,3 -> 77,77,64
    x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
    # 77,77,64 -> 38,38,64
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)

    # 38,38,64 -> 17,17,256
    x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
    x = conv2d_bn(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')

    # 5 x Block35 (17, 17, 256)
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x, scale=0.17, block_type='Block35', block_idx=block_idx)

    # 17,17,256 -> 8,8,896
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 192, 3, name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = conv2d_bn(branch_1, 256, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 1))
    branch_pool = MaxPooling2D(3, strides=2, padding='valid', name=name_fmt('MaxPool_1a_3x3', 2))(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    # 10 x Block17 (8, 8, 896)
    for block_idx in range(1, 11):
        x = _inception_resnet_block(x, scale=0.1, block_type='Block17', block_idx=block_idx)

    # 8,8,896 -> 3,3,1792
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 256, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 1))
    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 256, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2, 256, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 2))
    branch_pool = MaxPooling2D(3, strides=2, padding='valid', name=name_fmt('MaxPool_1a_3x3', 3))(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    # 5 x Block8 (3,3,1792)
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x, scale=0.2, block_type='Block8', block_idx=block_idx)
    x = _inception_resnet_block(x, scale=1, activation=None, block_type='Block8', block_idx=6)

    # GlobalAveragePooling2D (None, 1792)
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1 - keep_prob, name='Dropout')(x)

    # Dense 128
    x = Dense(classes, use_bias=False, name='Bottleneck')(x)
    bn_name = _generate_layer_name('BatchNorm', prefix='Bottleneck')
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name=bn_name)(x)

    model = Model(inputs, x, name='inception_resnet_v1')
    # Total params: 22,808,144
    # Trainable params: 22,779,312
    # Non trainable params: 28,832
    return model

class InceptionResNetV1(object):
    _defaults = {
        "model_path": 'model_data/facenet/facenet_keras.h5'
    }

    def __init__(self, input_shape=(160, 160, 3), classes=128, keep_prob=0.8):
        self.__dict__.update(self._defaults)
        self.channel_axis = 3
        self.input_shape = input_shape
        self.classes = classes
        self.keep_prob = keep_prob
        self.generate()

    def generate(self):
        start = time.time()
        print("Model Load Start>>>>>>", start)
        self.model = _InceptionResNetV1(self.input_shape, self.classes, self.keep_prob)

        # Load model Weight
        self.model.load_weights(self.model_path)
        print(self.model_path)
        endTime = time.time() - start
        print("Model Load Finished>>>>>>", endTime)

    def encode(self, img):
        faceImg = pre_process(img)
        result = self.model.predict(faceImg)
        result = l2_norm(np.concatenate(result))
        result = np.reshape(result, [128])
        return result
























