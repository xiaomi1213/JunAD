import os
import urllib.request
import pickle

def load_data(dataset_name, path, pre_trained=True):
    if os.path.isdir(path):
        dataset = pickle.load(open(path, 'r'))

    else:
        if dataset_name.lower() is 'mnist':
            from keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            (X_train, y_train), (X_test, y_test) = prepare_data(self.image_size, self.num_channals, self.num_classes)

            if pre_trained is True:
                del X_train, y_train
            return X_test, y_test


        elif dataset_name.lower() is 'cifar10':
            return True, False

        elif dataset_name.lower() is 'imagenet':
            return True, False

        else:
            return True, False
        download_dataset = urllib.request.
        x,y = load(dataset_name,path)

    x, y = prepare(dataset)
    return x, y