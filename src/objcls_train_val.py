# 20/02/2017, Rodrigo Santa Cruz
# Training and validation script

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import objcls_model
import os


def train_eval(data_dir, classes, output_dir, samples_per_epoch, nb_val_samples, nb_epoch, bottleneck_epochs,
               snapshot_interval):
    """
    CNN classifier train and validation script
    :param data_dir: Directory with train/classes and validation/classes folders
    :param classes: list of classes. If [] use all subdirectories of data_dir/train in the alphabetical order
    :param output_dir: directory to save logs and models
    :param samples_per_epoch: num of samples per epoch
    :param nb_val_samples: number of samples for validation
    :param nb_epoch: number of epochs
    :param bottleneck_epochs: number bottleneck epochs
    :param snapshot_interval: interval to snapshot models
    :return: Return trained model
    """
    # Setup Data Generators. Train with data augmentation!
    if not classes:
        for file in sorted(os.listdir(os.path.join(data_dir, 'train'))):
            if os.path.isdir(os.path.join(data_dir, 'train', file)):
                classes.append(file)
    cls_string = '-'.join(classes)
    print("Classifier for classes: {}".format(classes))

    train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=(150, 150),
            batch_size=32,
            classes=classes,
            class_mode='categorical')
    # Test generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
            os.path.join(data_dir, 'validation'),
            target_size=(150, 150),
            batch_size=32,
            classes=classes,
            class_mode='categorical')

    # Set up directories
    if os.path.exists(output_dir):
        raise ValueError("Output directory already exists; {}".format(output_dir))
    os.makedirs(output_dir)

    # Create Model
    model, base_model = objcls_model.vgg_based_model(len(classes), input_shape=(150, 150, 3))

    # train only the top layers (which were randomly initialized)
    if bottleneck_epochs > 0:
        print("Training Classifier...")
        logs_cls_dir = os.path.join(output_dir, 'cls_train_val')
        os.mkdir(logs_cls_dir)
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
        callbacks = [TensorBoard(log_dir=logs_cls_dir, write_graph=False),
                     ModelCheckpoint(os.path.join(logs_cls_dir,
                                                  "clstrain_weights.CLS_" + cls_string + ".E_{epoch:02d}-VACC_{val_acc:.2f}.hdf5"),
                                     save_weights_only=True, period=snapshot_interval)]
        model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=bottleneck_epochs,
                            validation_data=validation_generator,
                            nb_val_samples=nb_val_samples,
                            callbacks=callbacks)

    # train the whole net
    print("Finetune all layers...")
    logs_ft_dir = os.path.join(output_dir, 'ft_train_val')
    os.mkdir(logs_ft_dir)
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    callbacks = [TensorBoard(log_dir=logs_ft_dir, write_graph=False),
                 ModelCheckpoint(
                         os.path.join(logs_ft_dir,
                                      "finetune_weights.CLS_" + cls_string + ".E_{epoch:02d}-VACC_{val_acc:.2f}.hdf5"),
                         save_weights_only=True, period=snapshot_interval)]
    model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=nb_val_samples,
                        callbacks=callbacks)

    return model


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog="CNN Classifier training and Validation")
    parser.add_argument("data_dir", type=str, help="Data directory for train and validation")
    parser.add_argument("output_dir", type=str, help="Output directory for logs and snapshots")
    parser.add_argument("-train_samples", type=int, default=2000, help="Number of samples per epoch")
    parser.add_argument("-val_samples", type=int, default=800, help="Number of samples for validation")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("-bottleneck", type=int, default=0,
                        help="Number of epochs for bottleneck training. "
                             "If not passed, not bottleneck training is performed ")
    parser.add_argument("-snap_intv", type=int, default=10, help="Model snapshot interval")
    parser.add_argument("-classes", type=str, nargs='*', default=[],
                        help="Classes to learn a classifier. They should be subdirectories of train and validation "
                             "folder on data_dir. If not passed all subdirectories will be considered")
    args = parser.parse_args()
    print(args)

    # Run train and validation script
    train_eval(args.data_dir, args.classes, args.output_dir, args.train_samples, args.val_samples, args.epochs,
               args.bottleneck,
               args.snap_intv)
