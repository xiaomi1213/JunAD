import tensorflow as tf

class Basic_cnn_tf_model(object):
    def __init__(self,keep_prob, num_classes):
        self.keep_prob = keep_prob
        self.num_classes = num_classes

    def fprop(self, x, y):
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
        dropout = tf.layers.dropout(dense1, rate=self.keep_prob)
        logits = tf.layers.dense(dropout, self.num_classes)
        outputs = tf.nn.softmax(logits)
        loss = tf.losses.softmax_cross_entropy(y,logits)

        return logits, outputs, loss



