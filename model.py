import time
import utils
import os
from abc import ABCMeta
import tensorflow as tf
import numpy as np


"""
class Model(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def fprop(self, x):

        raise NotImplementedError

    def get_logits(self, x):
        logits, _ = self.fprop(x)
        return logits

    def get_probs(self, x):
        _, probs = self.fprop(x)
        return probs

    def get_loss(self, x, y):
        logits,_ = self.fprop(x)
        loss = tf.losses.softmax_cross_entropy(y, logits)
        return loss

    def predict(self, x):
        logits, _ = self.fprop(x)
        preds = tf.argmax(logits, axis=1)
        return preds
"""

"""
class Basic_cnn_tf_model(object):

    __metaclass__ = ABCMeta
    def __init__(self, keep_prob, num_classes, ):
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        
        self.sess = tf.Session()


    def jun_fprop(self,x):
        with tf.variable_scope(self.__class__.__name__, reuse=tf.AUTO_REUSE):
            # shape(28, 28, 1)
            conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=5,
                                     strides=1, padding='same', activation=tf.nn.relu)  # -> (28, 28, 16 )

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)  # -> (14, 14, 16)

            conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=5,
                                     strides=1, padding='same', activation=tf.nn.relu)  # -> (14, 14, 32)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)  # -> (7, 7, 32)

            flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
            dense1 = tf.layers.dense(flat, 256)
            dropout = tf.layers.dropout(dense1, rate=self.keep_prob)
            logits = tf.layers.dense(dropout, self.num_classes)
            probs = tf.nn.softmax(logits)
            return logits, probs

    def jun_get_logits(self, x):
        tf_logits, _ = self.jun_fprop(self.x)
        with self.sess.as_default():
            #self.sess.run(tf.global_variables_initializer())
            logits = self.sess.run(tf_logits, feed_dict={self.x: x})

        return logits

    def jun_get_probs(self, x):
        _, tf_probs = self.jun_fprop(self.x)
        with self.sess.as_default():
            #self.sess.run(tf.global_variables_initializer())
            probs = self.sess.run(tf_probs, feed_dict={self.x: x})

        return probs

    def jun_predict(self, x):
        tf_logits, _ = self.jun_fprop(x)
        preds = tf.argmax(tf_logits, axis=1)
        with self.sess.as_default():
            #self.sess.run(tf.global_variables_initializer())
            preds = self.sess.run(preds, feed_dict={self.x: x})
        return preds

    def train(self, X_train, Y_train, batch_size=100, num_epoch=2, learning_rate=1e-4, save=True):
        num_batches = int(float(len(X_train)) / batch_size)
        dataset_size = len(X_train)
        dataset_idx = list(range(len(X_train)))
        rs.shuffle(dataset_idx)

        logits, _ = self.jun_fprop(self.x)
        loss = tf.losses.softmax_cross_entropy(self.y, logits)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
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
                    _, training_loss = self.sess.run([train_step, loss], feed_dict={self.x: X_train_batch, self.y: Y_train_batch})

                    #batch_duration = time.time() - batch_prev
                    # print('epoch %d -- step %d -- batch duration: %.3fs -- loss: %.4f\n' % (epoch, batch, batch_duration, training_loss))
                epoch_loss = training_loss
                epoch_duration = time.time() - epoch_start
                print('epoch %d -- epoch duration: %.3fs -- epoch_loss: %.4f\n' % (epoch, epoch_duration, epoch_loss))
            train_duration = time.time() - train_start
            print('----------end training, Total time consuming: %.3fs-----------\n' % train_duration)
            if save:
                file_name = 'trained_model.ckpt'
                train_path = r'E:\Bluedon\3Code\expriments\train\1'
                save_path = os.path.join(train_path, file_name)
                saver = tf.train.Saver()
                saver.save(self.sess, save_path)
                print("Model saved in path: %s" % save_path)
"""

class Basic_cnn_tf_model(object):
    __metaclass__ = ABCMeta

    def __init__(self, keep_prob, num_classes, ):
        self.keep_prob = keep_prob
        self.num_classes = num_classes

    def fprop(self, x):
        with tf.variable_scope(self.__class__.__name__, reuse=tf.AUTO_REUSE):
            # shape(28, 28, 1)
            conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=5,
                                     strides=1, padding='same', activation=tf.nn.relu)  # -> (28, 28, 16 )

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)  # -> (14, 14, 16)

            conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=5,
                                     strides=1, padding='same', activation=tf.nn.relu)  # -> (14, 14, 32)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)  # -> (7, 7, 32)

            flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
            dense1 = tf.layers.dense(flat, 256)
            dropout = tf.layers.dropout(dense1, rate=self.keep_prob)
            logits = tf.layers.dense(dropout, self.num_classes)
            probs = tf.nn.softmax(logits)
            return logits, probs

    def jun_get_logits(self, x):
        logits, _ = self.fprop(x)
        return logits

    def jun_get_probs(self, x):
        _, probs = self.fprop(x)
        return probs

    """
        def jun_predict(self, x, y):#y is useless
        tf_logits, _ = self.fprop(x)
        preds = tf.argmax(tf_logits, axis=1)
        return preds
    """

    def jun_get_loss(self, x, y):
        logits, _ = self.fprop(x)
        loss = tf.losses.softmax_cross_entropy(y, logits)
        return loss

from cleverhans.model import Model
class Cleverhans_model_wrapper(Model,Basic_cnn_tf_model):

    def __init__(self, scope, num_classes, **kwargs):
        del kwargs
        Model.__init__(self, scope, num_classes, locals())
        self.scope = scope or self.__class__.__name__
        self.num_classes = num_classes
        Basic_cnn_tf_model.__init__(self,keep_prob=0.5, num_classes=num_classes)

    def fprop(self, x, **kwargs):
        del kwargs
        logits, probs =  Basic_cnn_tf_model.fprop(self, x)
        #return logits, probs

        return {'logits': logits,
                'probs': probs}

        """
                with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # shape(28, 28, 1)
            conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=5,
                                     strides=1, padding='same', activation=tf.nn.relu)  # -> (28, 28, 16 )

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)  # -> (14, 14, 16)

            conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=5,
                                     strides=1, padding='same', activation=tf.nn.relu)  # -> (14, 14, 32)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)  # -> (7, 7, 32)

            flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
            dense1 = tf.layers.dense(flat, 256)
            dropout = tf.layers.dropout(dense1, rate=0.5)
            logits = tf.layers.dense(dropout, self.num_classes)
            probs = tf.nn.softmax(logits)
            #return logits, probs
            return logits, probs
        """
    def jun_get_loss(self, x, y):
        logits = self.fprop(x)['logits']
        loss = tf.losses.softmax_cross_entropy(y, logits)
        return loss

    def jun_get_logits(self, x):
        logits = self.fprop(x)['logits']
        return logits

    """
        def jun_predict(self, x, x_test):
        tf_logits = self.fprop(x)['logits']
        preds = tf.argmax(tf_logits, axis=1)
        with tf.Session() as sess:
            predictions = sess.run(preds, feed_dict={x:x_test})
        return predictions

    """



