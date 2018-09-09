import os
import keras.utils.np_utils as np_utils

class Dataset:
    def __init__(self, dataset_name, image_size, num_channels, num_classes, normalize=True, pre_trained=True):
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.normalize = normalize
        self.pre_trained = pre_trained

    def pre_process(self, X_train, y_train, X_test, y_test):
        X_train = X_train.reshape(X_train.shape[0], self.image_size, self.image_size, self.num_channels)
        X_train = X_train.astype('float32')
        X_train /= 255
        y_train = np_utils.to_categorical(y_train, self.num_classes)

        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        y_test = np_utils.to_categorical(y_test, self.num_classes)

        return (X_train, y_train), (X_test, y_test)

    def load_dataset(self):
        if self.dataset_name.lower() is 'mnist':
            from keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            (X_train, y_train), (X_test, y_test) = self.pre_process(X_train, y_train, X_test, y_test, self.normalize)
            if self.pre_trained:
                del X_train, y_train
                return X_test, y_test
            else:
                return (X_train, y_train), (X_test, y_test)

        elif self.dataset_name.lower() is 'cifar10':
            from keras.datasets import cifar10
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            (X_train, y_train), (X_test, y_test) = self.pre_process(X_train, y_train, X_test, y_test, self.normalize)
            if self.pre_trained:
                del X_train, y_train
                return X_test, y_test
            else:
                return (X_train, y_train), (X_test, y_test)

        elif self.dataset_name.lower() is 'imagenet':

            print('Download it by yourself due to the imagenet dataset is too large to download')

        else:
            print('The dataset you inputed is not supported by this project')














