
class Load_dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if self.dataset_name.lower() not in ['mnist','cifar10','imagenet']:
            raise Exception("Please input dataset name mnist, cifar10, imagenet")

    def get_dataset(self,pre_trained=True):
        if self.dataset_name.lower() is 'mnist':
            from keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            (X_train, y_train), (X_test, y_test) = prepare_data(self.image_size, self.num_channals, self.num_classes)

            if pre_trained is True:
                del X_train, y_train
            return X_test, y_test


        elif self.dataset_name.lower() is 'cifar10':
            return True, False

        elif self.dataset_name.lower() is 'imagenet':
            return True, False

        else:
            return True, False


    def get_batch_dataset(self,batch_size):
        a,b  = self.get_dataset(pre_trained=False)
        a,b = make_batch(a,b)
