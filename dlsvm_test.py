import tensorflow as tf
import numpy as np

def svm_l1loss(a, y, weight, C=1.0):
    relu = tf.nn.relu(1 - a * y)
    loss = 0.5 * tf.matmul(weight, weight) + C * relu.sum()
    return loss

def svm_l2loss(a, y, weight, C=1.0):
    relu = tf.nn.relu(1 - a * y)
    loss = 0.5 * tf.matmul(weight, weight) + C * relu.sum()
    return loss



if __name__ == '__main__':

    from cleverhans.utils_mnist import data_mnist
    import tensorflow as tf
    from evaluate_test import evaluate
    from model_test import Cleverhans_model_wrapper
    from show_image import show_a_image
    from train_tf_model_test import train_tf_model_test
    from predict_test import predict
    from attack_test import fgsm_attacks
    x_train, y_train, x_test, y_test = data_mnist(train_start=0,
                                                  train_end=5000,
                                                  test_start=0,
                                                  test_end=1000)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    sess = tf.Session()
    kwargs = {}
    model = Cleverhans_model_wrapper(scope=None, num_classes=10, **kwargs)
    #model = Basic_cnn_tf_model(keep_prob=0.5, num_classes=10)
    #model.train(x_train, y_train)
    loss = model.jun_get_loss(x, y)
    train_tf_model_test(sess, loss, x, y, x_train, y_train, num_epoch=100)
    logits = model.jun_get_logits(x)
    evaluate(sess, logits, x, y, x_test, y_test)
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.}
    adv_x = fgsm_attacks(model, sess=sess, x=x_test, **fgsm_params)
    show_a_image(np.squeeze(adv_x[0]))
    show_a_image(np.squeeze(x_test[0]))
    y_adv = predict(sess, logits, x, adv_x)
    print('y_adv: ', y_adv[0])
    evaluate(sess, logits, x, y, adv_x, y_test)
