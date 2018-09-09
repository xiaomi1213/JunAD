
from cleverhans import attacks

def fgsm_attacks(sess, model, x, **kwargs):
    fgsm = attacks.FastGradientMethod(model, back='tf', sess=sess)
    adv_x = fgsm.generate(x, **kwargs)
    return adv_x





if __name__ == '__main__':

    from cleverhans.utils_mnist import data_mnist
    import tensorflow as tf
    from model_evaluate import evaluate
    from models import Cleverhans_model_wrapper
    from train_tf_model import train_tf_model
    from show_image import show_a_image
    import numpy as np
    x_train, y_train, x_test, y_test = data_mnist(train_start=0,train_end=5000,test_start=0,test_end=1000)

    x = tf.placeholder(tf.float32, [None,28,28,1])
    y = tf.placeholder(tf.float32, [None,10])
    kwargs = {}
    model = Cleverhans_model_wrapper(scope=None, num_classes=10, **kwargs)
    logits = model.fprop(x,**kwargs)['logits']
    loss = tf.losses.softmax_cross_entropy(y,logits)
    saver = tf.train.Saver()
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pre_trained = False
        checkpoint_dir = r'E:\Bluedon\3Code\expriments\train\1'
        adv_x = fgsm_attacks(sess, model, x, **kwargs)
        adv = sess.run(adv_x,feed_dict={x:x_test, y:y_test})
        print(adv.shape)
        show_a_image(np.squeeze(x_test[0]))
        show_a_image(np.squeeze(adv[0]))
        if pre_trained:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                pass

            evaluate(sess, logits, x, y, x_test, y_test)
        else:
            train_tf_model(sess, loss, x, y, x_train, y_train, num_epoch=100)
            evaluate(sess, logits, x, y, x_test, y_test)






