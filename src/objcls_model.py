# 20/02/2017, Rodrigo Santa Cruz
# Script for define classifier models based on CNNs

from keras.applications import xception, vgg16, vgg19, inception_v3, resnet50
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from enum import Enum
from keras.optimizers import SGD
import re


def write_model_str(ebase_cnn, classes):
    return ".BM_{}.CLS_{}".format(ebase_cnn.name, '-'.join(classes))


def read_model_str(model_str):
    # read of base model enum
    ebase_cnn = re.search(".*\.BM_(.+?)\..*", model_str).group(1)
    ebase_cnn = EBaseCNN[ebase_cnn]

    # read of classes
    classes = re.search(".*\.CLS_(.+?)\..*", model_str).group(1)
    classes = classes.split('-')

    return ebase_cnn, classes


class EBaseCNN(Enum):
    XCEPTION = 1
    VGG16 = 2
    VGG19 = 3
    INCEPTION_V3 = 4
    RESNET_50 = 5


def _base_model_factory(ecnn, input_shape, pre_init=True):
    model = None
    weights = 'imagenet' if pre_init else None
    if ecnn == EBaseCNN.XCEPTION:
        model = xception.Xception(include_top=False, weights=weights, input_shape=input_shape)
    elif ecnn == EBaseCNN.VGG16:
        model = vgg16.VGG16(include_top=False, weights=weights, input_shape=input_shape)
    elif ecnn == EBaseCNN.VGG19:
        model = vgg19.VGG19(include_top=False, weights=weights, input_shape=input_shape)
    elif ecnn == EBaseCNN.INCEPTION_V3:
        model = inception_v3.InceptionV3(include_top=False, weights=weights, input_shape=input_shape)
    elif ecnn == EBaseCNN.RESNET_50:
        model = resnet50.ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    else:
        raise ValueError("Base model is not available")
    return model


class CNNModelBuilder:
    def __init__(self, ebase_cnn, num_cls, input_shape=(150, 150, 3), weights=None):

        # Create Base Model
        self._base_model = _base_model_factory(ebase_cnn, input_shape, pre_init=(weights is None))

        # Add new classifier
        last = self._base_model.output
        x = Flatten()(last)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_cls, activation='softmax')(x)
        self._model = Model(input=self._base_model.input, output=predictions)

        if weights:
            self._model.load_weights(weights)

    def inference_model(self):
        return self._model

    def learning_model(self):

        for layer in self._model.layers:
            layer.trainable = True

        self._model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',
                            metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

        return self._model
