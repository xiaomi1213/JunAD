import tensorflow as tf
import numpy as np

"""
def evaluate_adv(sess, x, y, X_test, Y_test, Y_adv):
    correct_prediction = tf.equal(tf.argmax(Y_adv, axis=-1), tf.argmax(Y_test, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        accu = sess.run(accuracy, feed_dict={x: X_test, y: Y_test})
        print('\nThe trained model accuracy: %.4f' % accu)
"""

"""
def evaluate(Y_preds, Y_test):

    correct_prediction = np.equal(np.argmax(Y_preds, axis=1), np.argmax(Y_test, axis=1))
    accuracy = np.mean(correct_prediction)
    acc = (float(accuracy) * float(100))
    print('\nThe accuracy: %.2f%%' % acc)
"""

def evaluate(sess, logits, x, y, X_test, Y_test):
    correct_prediction = tf.equal(tf.argmax(logits, axis=-1), tf.argmax(y, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with sess.as_default():
        #sess.run(tf.global_variables_initializer())
        accu = sess.run(accuracy, feed_dict={x:X_test, y:Y_test})
        accu = float(accu*100)
        print('\nThe trained model accuracy: %.2f%%' % accu)
    #return acc

if __name__ == "__main__":
    from cleverhans.utils_mnist import data_mnist
    from model_test import Basic_cnn_tf_model
    from model_test import Cleverhans_model_wrapper
    import numpy as np
    from train_tf_model_test import train_tf_model_test
    x_train, y_train, x_test, y_test = data_mnist(train_start=0,
                                                  train_end=5000,
                                                  test_start=0,
                                                  test_end=1000)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    sess = tf.Session()
    model = Basic_cnn_tf_model(keep_prob=0.5, num_classes=10)
    # model.train(x_train, y_train)
    loss = model.jun_get_loss(x, y)
    train_tf_model_test(sess, loss, x, y, x_train, y_train, num_epoch=2)
    logits = model.jun_get_logits(x)
    evaluate(sess, logits, x, y, x_test, y_test)

