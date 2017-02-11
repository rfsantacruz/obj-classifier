from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.vgg16 import preprocess_input
import os


def train_eval(data_dir, output_dir, samples_per_epoch, nb_epoch, nb_val_samples):
    # Set up directories
    if os.path.exists(output_dir):
        raise ValueError("Output directory exists; {}".format(output_dir))
    os.makedirs(output_dir)
    models_dir = os.path.join(output_dir, 'models')
    os.mkdir(models_dir)
    logs_cls_dir = os.path.join(output_dir, 'cls_train_val')
    os.mkdir(logs_cls_dir)
    logs_ft_dir = os.path.join(output_dir, 'ft_train_val')
    os.mkdir(logs_ft_dir)

    # Data Generators
    # Train with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    # Test generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    # Create a base model
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

    # Add new classifier
    last = base_model.output
    x = Flatten()(last)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    print("Training Classifier...")
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    callbacks = [TensorBoard(log_dir=logs_cls_dir, write_graph=False),
                 ModelCheckpoint(os.path.join(models_dir, "clstrain_weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
                                 save_weights_only=False, period=1)]
    model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=nb_val_samples,
                        callbacks=callbacks)

    # Second: train the whole net
    print("Finetune all layers...")
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    callbacks = [TensorBoard(log_dir=logs_ft_dir, write_graph=False),
                 ModelCheckpoint(os.path.join(models_dir, "finetune_weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
                                 save_weights_only=False, period=1)]
    model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=nb_val_samples,
                        callbacks=callbacks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="VGG classifier training and evaluation")
    parser.add_argument("data_dir", type=str, help="Data directory for train and validation")
    parser.add_argument("output_dir", type=str, help="Output directory for logs and snapshots")
    parser.add_argument("-train_epoch", type=int, default=2000, help="Number of samples per epoch")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("-val_samples", type=int, default=800, help="Number of samples for validation")
    args = parser.parse_args()
    print(args)
    train_eval(args.data_dir, args.output_dir, args.train_epoch, args.epochs, args.val_samples)
