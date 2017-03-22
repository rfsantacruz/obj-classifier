# 20/02/2017, Rodrigo Santa Cruz
# Training and validation script

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import objcls_model
import os


def train_eval(base_model_str, data_dir, classes, output_dir, samples_per_epoch, nb_val_samples, nb_epoch,
               snapshot_interval):
    """
    CNN classifier train and validation script
    :param base_model_str: String name of architecture to use as base model
    :param data_dir: Directory with train/classes and validation/classes folders
    :param classes: list of classes. If [] use all subdirectories of data_dir/train in the alphabetical order
    :param output_dir: directory to save logs and models
    :param samples_per_epoch: num of samples per epoch
    :param nb_val_samples: number of samples for validation
    :param nb_epoch: number of epochs
    :param snapshot_interval: interval to snapshot models
    :return: Return trained model
    """
    # Setup Data Generators. Train with data augmentation!
    if not classes:
        for file in sorted(os.listdir(os.path.join(data_dir, 'train'))):
            if os.path.isdir(os.path.join(data_dir, 'train', file)):
                classes.append(file)
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
    model_builder = objcls_model.CNNModelBuilder(objcls_model.EBaseCNN[base_model_str], len(classes), input_shape=(150, 150, 3))
    model = model_builder.learning_model()

    # Train and validate the network
    print("Finetune model")
    callbacks = [TensorBoard(log_dir=output_dir, write_graph=False),
                 ModelCheckpoint(
                         os.path.join(output_dir,
                                      "finetune_weights" + objcls_model.write_model_str(
                                              objcls_model.EBaseCNN[base_model_str],
                                              classes) + ".E_{epoch:02d}-VACC_{val_acc:.2f}.hdf5"),
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
    parser.add_argument("base_model", type=str, choices=[ecnn.name for ecnn in list(objcls_model.EBaseCNN)],
                        help="Base architecture to use")
    parser.add_argument("data_dir", type=str, help="Data directory for train and validation")
    parser.add_argument("output_dir", type=str, help="Output directory for logs and snapshots")
    parser.add_argument("-train_samples", type=int, default=2000, help="Number of samples per epoch")
    parser.add_argument("-val_samples", type=int, default=800, help="Number of samples for validation")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("-snap_intv", type=int, default=10, help="Model snapshot interval")
    parser.add_argument("-classes", type=str, nargs='*', default=[],
                        help="Classes to learn a classifier. They should be subdirectories of train and validation "
                             "folder on data_dir. If not passed all subdirectories will be considered")
    args = parser.parse_args()
    print(args)

    # Run train and validation script
    train_eval(args.base_model, args.data_dir, args.classes, args.output_dir, args.train_samples, args.val_samples,
               args.epochs,
               args.snap_intv)
