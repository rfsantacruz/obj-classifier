# 20/02/2017, Rodrigo Santa Cruz
# Script to perform prediction of a model in a set of images

from keras.preprocessing import image
from keras import backend as K
from matplotlib import pyplot as plt
import objcls_model
import numpy as np
import argparse, re


def prediction(weights_path, imgs_path, show):
    """
    Perform prediction and visualization
    :param weights_path:  keras weights file path
    :param imgs_path: list of images to perform prediction
    :param show: Plot visualization
    :return: list of predictions
    """
    classes = re.search(".*\.CLS_(.+?)\..*", weights_path).group(1)
    classes = classes.split('-')

    # Load trained model
    model, _ = objcls_model.vgg_based_model(len(classes), input_shape=(150, 150, 3))
    model.load_weights(weights_path)

    # read image, pre-process and predict
    print("Model prediction")
    preds = []
    for img_path in imgs_path:
        # Load image and predict
        img = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x *= (1. / 255)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        preds.append(pred)

        # Show results
        pred_str = ", ".join(["{}:{:.2f}".format(cls, prob) for cls, prob in zip(classes, np.squeeze(pred))])
        print("{}: {}".format(img_path, pred_str))
        if show:
            plt.imshow(img)
            plt.title(pred_str)
            plt.show()

    # Clear keras session
    K.clear_session()

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Prediction Demo")
    parser.add_argument("weigths_path", type=str, help="Path to model weight file")
    parser.add_argument("imgs_path", type=str, nargs='*', help="Path to images samples")
    parser.add_argument("-show", default=False, action="store_true", help="Visualize predictions")
    args = parser.parse_args()
    print("Classification demo: {}".format(args))
    prediction(args.weigths_path, args.imgs_path, args.show)
