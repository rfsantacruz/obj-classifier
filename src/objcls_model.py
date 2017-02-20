# 20/02/2017, Rodrigo Santa Cruz
# Script for define classifier models based on CNNs

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout


def vgg_based_model(num_cls, input_shape=(150, 150, 3), weights=None):
    """
    Create a classifier based on VGG
    :param num_cls: num of outputs
    :param input_shape: shape of the input images
    :param weights: keras h5 weight filepaht
    :return: model and vgg base model
    """
    # Create a base model
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    # Add new classifier
    last = base_model.output
    x = Flatten()(last)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_cls, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)

    if weights:
        model.load_weights(weights)

    return model, base_model
