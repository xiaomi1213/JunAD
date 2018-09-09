import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging


from cleverhans.loss import LossCrossEntropy
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.tutorial_models import ModelBasicCNN

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
flags.DEFINE_bool('backprop_through_attack', False,
                  ('If True, backprop through adversarial example '
                   'construction process during adversarial training'))


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None,
                   label_smoothing=True
                   ):
    report = AccuracyReport()

    tf.set_random_seed(1234)

    set_log_level(logging.DEBUG)

    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    x_train, y_train, x_test, y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    if label_smoothing:
        label_smooth = .1
        y_train = y_train.clip(label_smooth/(nb_classes-1), 1. - label_smooth)

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    eval_params = {'batch_size': batch_size}
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    rng = np.random.RandomState([2017, 8, 30])
    sess = tf.Session()

    def do_eval(preds, x_set, y_set, report_key, is_adv=None):
        acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
        setattr(report, report_key, acc)
        if is_adv is None:
            report_text = None
        elif is_adv:
            report_text = 'adversarial'
        else:
            report_text = 'legitimate'
        if report_text:
            print('Test accuracy on %s examples: %0.4f' % (report_text, acc))


    model = ModelBasicCNN('model1', nb_classes, nb_filters)
    print('done 1')
    preds = model.get_logits(x)
    print('done 2')
    loss = LossCrossEntropy(model, smoothing=0.1)
    print('done 3')

    def evaluate():
        do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)

    train(sess, loss, x, y, x_train, y_train, evaluate=evaluate,
        args=train_params, rng=rng, var_list=model.get_params())
    print('done 4')

    if testing:
        do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

    fgsm = FastGradientMethod(model, sess=sess)
    print('done 5')
    adv_x = fgsm.generate_np(x_test, **fgsm_params)


    print('adv_x shape', adv_x.shape)
    print('adv_x[0]', adv_x)
    from PIL import Image
    import show_image
    show_image.show_a_image(np.squeeze(adv_x[0]))


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    main()


