# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import re
import objcls_model
from keras import backend as K

# Color segmentation boundaries
lower = np.array([100, 130, 0], dtype="uint8")
upper = np.array([140, 210, 255], dtype="uint8")


def object_detection(weights_path, images_path):
    # load CNN model
    classes = re.search(".*\.CLS_(.+?)\..*", weights_path).group(1)
    classes = classes.split('-')
    model, _ = objcls_model.vgg_based_model(len(classes), input_shape=(150, 150, 3))
    model.load_weights(weights_path)

    for image_path in images_path:

        # load the image
        image = cv2.imread(image_path)

        # color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        mask = cv2.Canny(mask, 100, 100 * 2, 3)

        # find bounding boxes
        bbs = []
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                cnt = cv2.approxPolyDP(cnt, 3, True)
                x, y, w, h = cv2.boundingRect(cnt)
                bbs.append((x, y, w, h))

        # classify bbs
        dets = []
        for x, y, w, h in bbs:
            roi = cv2.cvtColor(image[y:y + h, x: x + w], cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, (150, 150)).astype(np.float32)
            roi *= (1. / 255)
            roi = np.expand_dims(roi, axis=0)
            pred = model.predict(roi)
            dets.append((x, y, w, h, pred))

        # draw detections and show image
        for x, y, w, h, pred in dets:
            pred_str = ", ".join(["{}:{:.2f}".format(cls, prob) for cls, prob in zip(classes, np.squeeze(pred))])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, pred_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    # Clear keras session
    K.clear_session()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("model_weights", type=str, help="path to the model weights")
    ap.add_argument("images_path", type=str, nargs='*', help="path to the image")
    args = ap.parse_args()

    object_detection(args.model_weights, args.images_path)
