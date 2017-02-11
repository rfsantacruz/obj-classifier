from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import argparse


def prediction(weights_path, imgs_path):
    # Load trained model
    base_model = VGG16(include_top=False, weights=None, input_shape=(150, 150, 3))
    last = base_model.output
    x = Flatten()(last)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(input=base_model.input, output=predictions)
    model.load_weights(weights_path)

    # read image, pre-process and predict
    print("Model prediction")
    for img_path in imgs_path:
        img = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)
        print("{}: {}".format(img_path, pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Prediction demo")
    parser.add_argument("weigths_path", type=str, help="Path to model weight file")
    parser.add_argument("imgs_path", type=str, nargs='*', help="Path to images samples")
    args = parser.parse_args()
    print("Classification demo: {}".format(args))
    prediction(args.weigths_path, args.imgs_path)
