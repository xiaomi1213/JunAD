import tensorflow as tf


def evaluate(sess, logits, x, y, X_test, Y_test):
    """
    evaluate trained model's accuracy, false positive rate, true positive rate
    :param sess:
    :param logits:
    :param x:
    :param y:
    :param X_test:
    :param Y_test:
    :return: accuracy, fpr, tpr
    """

    correct_prediction = tf.equal(tf.argmax(logits, axis=-1), tf.argmax(y, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        accu = sess.run(accuracy, feed_dict={x:X_test, y:Y_test})
        print('\nThe trained model accuracy: %.4f' % accu)
    #return acc


def evaluate_adv(sess, x, y, X_test, Y_test, Y_adv):
    correct_prediction = tf.equal(tf.argmax(Y_adv, axis=-1), tf.argmax(Y_test, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        accu = sess.run(accuracy, feed_dict={x: X_test, y: Y_test})
        print('\nThe trained model accuracy: %.4f' % accu)


def evaluate_adv_without_tf(Y_adv, Y_test):
    """
    The calculate the accuracy of the adversarial model
    :param Y_adv: adversarial examples
    :param Y_test: original test examples
    :return: The accuracy(numpy scaler)
    """
    import numpy as np
    correct_prediction = np.equal(tf.argmax(Y_adv, axis=1), tf.argmax(Y_test, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('\nThe adversarial accuracy: %.4f' % accuracy)



if __name__ == '__main__':
    from cleverhans.utils_mnist import data_mnist
    from models import Basic_cnn_tf_model
    from train_tf_model import train_tf_model

    x_train, y_train, x_test, y_test = data_mnist(train_start=0,
                                                  train_end=5000,
                                                  test_start=0,
                                                  test_end=1000)

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    model = Basic_cnn_tf_model(keep_prob=0.5, num_classes=10)
    loss = model.get_loss(x, y)
    logits = model.get_logits(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        pre_trained = False
        checkpoint_dir = r'E:\Bluedon\3Code\expriments\train\1'

        if pre_trained:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                pass

            evaluate(sess, logits, x, y, x_test, y_test)
        else:
            train_tf_model(sess, loss, x, y, x_train, y_train, num_epoch=3)
            evaluate(sess, logits, x, y, x_train, y_train)
