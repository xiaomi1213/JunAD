from datasets.load_dataset import Load_dataset

mnist = Load_dataset('mnist')
a ,y_test = mnist.get_dataset()