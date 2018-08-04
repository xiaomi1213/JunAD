import tensorflow as tf


def predict(sess, logits, x, x_test):
    preds = tf.argmax(logits, axis=1)
    with sess.as_default():
        predictions = sess.run(preds, feed_dict={x: x_test})
    return predictions

