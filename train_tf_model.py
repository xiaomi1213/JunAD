import time
import tensorflow as tf
import utils
import numpy as np
rs = np.random.RandomState()

from cleverhans.utils_mnist import data_mnist

def train_tf_model(sess, loss, x, y, X_train, Y_train, batch_size=100, num_epoch=10, learning_rate=1e-4):
    num_batches = int(float(len(X_train)) / batch_size)
    dataset_size = len(X_train)
    dataset_idx = list(range(len(X_train)))
    rs.shuffle(dataset_idx)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        train_start = time.time()
        print('------------------start training-----------------')
        for epoch in range(num_epoch):
            epoch_start = time.time()
            training_loss = 0
            for batch in range(num_batches):
                batch_prev = time.time()
                batch_start, batch_end = utils.iter_indeices(batch, batch_size, dataset_size)
                batch_dataset = dataset_idx[batch_start: batch_end]
                X_train_batch = X_train[batch_dataset]
                Y_train_batch = Y_train[batch_dataset]
                _, training_loss = sess.run([train_step,loss], feed_dict={x:X_train_batch, y:Y_train_batch})
                batch_duration = time.time() - batch_prev
                #print('epoch %d -- step %d -- batch duration: %.3fs -- loss: %.4f\n' % (epoch, batch, batch_duration, training_loss))
            epoch_loss = training_loss
            epoch_duration = time.time() - epoch_start
            print('epoch %d -- epoch duration: %.3fs -- epoch_loss: %.4f\n' % (epoch, epoch_duration, epoch_loss))
        train_duration = time.time() - train_start
        print('----------end training, Total time consuming: %.3fs-----------\n'% train_duration)





if __name__ == "__main__":
    from basic_cnn_tf_model import Basic_cnn_tf_model
    sess = tf.Session()
    model = Basic_cnn_tf_model(keep_prob=0.5, num_classes=10)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    _,_,loss = model.fprop(x,y)
    x_train, y_train, x_test, y_test = data_mnist(train_start=0,
                                                  train_end=5000,
                                                  test_start=0,
                                                  test_end=1000)

    train_tf_model(sess, loss, x, y, x_train, y_train)

