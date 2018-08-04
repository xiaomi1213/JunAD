from abc import ABCMeta
import tensorflow as tf


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


sess = tf.Session()
class Basic_cnn_tf_model(object):
    """

    """
    def __init__(self,keep_prob, num_classes):
        super(Basic_cnn_tf_model, self).__init__()
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])

    def fprop(self, x):
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

    def get_logits(self, x):
        tf_logits, _ = self.fprop(self.x)
        with sess.as_default():
            logits = sess.run(tf_logits, feed_dict={self.x: x})
        return logits

    def get_probs(self, x):
        _, tf_probs = self.fprop(self.x)
        with sess.as_default():
            probs = sess.run(tf_probs, feed_dict={self.x: x})
        return probs

    def predict(self, x):
        tf_logits, _ = self.fprop(x)
        preds = tf.argmax(tf_logits, axis=1)
        with sess.as_default():
            preds = sess.run(preds, feed_dict={self.x: x})
        return preds

    def get_loss(self, x, y):
        tf_logits,_ = self.fprop(x)
        tf_loss = tf.losses.softmax_cross_entropy(y, tf_logits)
        with sess.as_default():
            loss = sess.run(tf_loss, feed_dict={self.x: x, self.y: y})
        return loss



# the wrapper for cleverhans model
from cleverhans.model import Model
"""
class Cleverhans_model_wrapper(Model):
    O_LOGITS, O_PROBS, O_FEATURES = 'logits probs features'.split()
    def __init__(self,scope, num_classes, **kwargs):
        del kwargs
        Model.__init__(self, scope, num_classes, locals())
        self.scope = scope or self.__class__.__name__
        self.num_classes = num_classes

    def fprop(self, x, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(inputs=x,  # shape(28, 28, 1)
                                          filters=16,
                                          kernel_size=5,
                                          strides=1,
                                          padding='same',
                                          activation=tf.nn.relu)  # -> (28, 28, 16 )

            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                 pool_size=2,
                                                 strides=2)  # -> (14, 14, 16)

            conv2 = tf.layers.conv2d(inputs=pool1,
                                          filters=32,
                                          kernel_size=5,
                                          strides=1,
                                          padding='same',
                                          activation=tf.nn.relu)  # -> (14, 14, 32)

            pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                                 pool_size=2,
                                                 strides=2
                                                 )  # -> (7, 7, 32)

            flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
            dense1 = tf.layers.dense(flat, 256)
            dropout = tf.layers.dropout(dense1, rate=0.5)
            logits = tf.layers.dense(dropout, self.num_classes)
            #probs = tf.nn.softmax(logits)

            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}
            #raise NotImplementedError('`fprop` not implemented.')

    def get_layer_names(self):
     
        pass
"""

class Cleverhans_model_wrapper(Model):


    def __init__(self, scope, num_classes, **kwargs):
        del kwargs
        Model.__init__(self, scope, num_classes, locals())
        self.scope = scope or self.__class__.__name__
        self.num_classes = num_classes
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])

    def fprop(self, x, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(inputs=x,  # shape(28, 28, 1)
                                     filters=16,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu)  # -> (28, 28, 16 )

            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                            pool_size=2,
                                            strides=2)  # -> (14, 14, 16)

            conv2 = tf.layers.conv2d(inputs=pool1,
                                     filters=32,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu)  # -> (14, 14, 32)

            pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                            pool_size=2,
                                            strides=2
                                            )  # -> (7, 7, 32)

            flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
            dense1 = tf.layers.dense(flat, 256)
            dropout = tf.layers.dropout(dense1, rate=0.5)
            logits = tf.layers.dense(dropout, self.num_classes)
            probs = tf.nn.softmax(logits)
            return logits, probs

    def obtain_logits(self, x):
        tf_logits, _ = self.fprop(self.x)
        with sess.as_default():
            logits = sess.run(tf_logits, feed_dict={self.x: x})
        return logits

    def obtain_probs(self, x):
        _, tf_probs = self.fprop(self.x)
        with sess.as_default():
            probs = sess.run(tf_probs, feed_dict={self.x: x})
        return probs

    def predict(self, x):
        tf_logits, _ = self.fprop(x)
        preds = tf.argmax(tf_logits, axis=1)
        with sess.as_default():
            preds = sess.run(preds, feed_dict={self.x: x})
        return preds

    def get_loss(self):
        tf_logits, _ = self.fprop(self.x)
        tf_loss = tf.losses.softmax_cross_entropy(tf_logits, self.y)
        return tf_loss