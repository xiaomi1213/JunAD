import time
import os
import tensorflow as tf
import utils
import numpy as np
rs = np.random.RandomState()

"""
def train_tf_model_test_without_holder(sess, loss, x, y, X_train, Y_train, batch_size=100, num_epoch=3, learning_rate=1e-4, save=True):
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

                # get the iterable indices for training
                batch_start, batch_end = utils.iter_indeices(batch, batch_size, dataset_size)
                batch_dataset = dataset_idx[batch_start: batch_end]
                X_train_batch = X_train[batch_dataset]
                Y_train_batch = Y_train[batch_dataset]

                # training run
                _, training_loss = sess.run([train_step,loss], feed_dict={x:X_train_batch, y:Y_train_batch})
                batch_duration = time.time() - batch_prev
                #print('epoch %d -- step %d -- batch duration: %.3fs -- loss: %.4f\n' % (epoch, batch, batch_duration, training_loss))
            epoch_loss = training_loss
            epoch_duration = time.time() - epoch_start
            print('epoch %d -- epoch duration: %.3fs -- epoch_loss: %.4f\n' % (epoch, epoch_duration, epoch_loss))
        train_duration = time.time() - train_start
        print('----------end training, Total time consuming: %.3fs-----------\n'% train_duration)
        if save:
            file_name = 'trained_model.ckpt'
            train_path = r'E:\Bluedon\3Code\expriments\train\2'
            save_path = os.path.join(train_path, file_name)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Model saved in path: %s" % save_path)
"""

def train_tf_model_test(sess, loss, x, y, X_train, Y_train, batch_size=100, num_epoch=3, learning_rate=1e-4, save=True):
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

                # get the iterable indices for training
                batch_start, batch_end = utils.iter_indeices(batch, batch_size, dataset_size)
                batch_dataset = dataset_idx[batch_start: batch_end]
                X_train_batch = X_train[batch_dataset]
                Y_train_batch = Y_train[batch_dataset]

                # training run
                _, training_loss = sess.run([train_step,loss], feed_dict={x:X_train_batch, y:Y_train_batch})
                batch_duration = time.time() - batch_prev
                #print('epoch %d -- step %d -- batch duration: %.3fs -- loss: %.4f\n' % (epoch, batch, batch_duration, training_loss))
            epoch_loss = training_loss
            epoch_duration = time.time() - epoch_start
            print('epoch %d -- epoch duration: %.3fs -- epoch_loss: %.4f\n' % (epoch, epoch_duration, epoch_loss))
        train_duration = time.time() - train_start
        print('----------end training, Total time consuming: %.3fs-----------\n'% train_duration)
        if save:
            file_name = 'trained_model.ckpt'
            train_path = r'E:\Bluedon\3Code\expriments\train\2'
            save_path = os.path.join(train_path, file_name)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Model saved in path: %s" % save_path)



if __name__ == "__main__":
    from cleverhans.utils_mnist import data_mnist
    from model_test import Basic_cnn_tf_model
    from model_test import Cleverhans_model_wrapper

    x_train, y_train, x_test, y_test = data_mnist(train_start=0,
                                                  train_end=5000,
                                                  test_start=0,
                                                  test_end=1000)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    sess = tf.Session()
    model = Basic_cnn_tf_model(keep_prob=0.5, num_classes=10)
    loss = model.jun_get_loss(x,y)
    train_tf_model_test(sess, loss, x, y, x_train, y_train, num_epoch=1)



