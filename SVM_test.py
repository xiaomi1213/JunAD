
from cleverhans.utils_mnist import data_mnist
from cleverhans import attacks
import tensorflow as tf
from evaluate_test import evaluate
from model_test import Cleverhans_model_wrapper
from model_test import Basic_cnn_tf_model
from show_image import show_a_image
import numpy as np
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
train_tf_model_test(sess, loss, x, y, x_train, y_train, num_epoch=1)
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

from sklearn import svm
svm_x = np.reshape(x_train,[5000,28*28])
svm_y = np.argmax(y_train,axis=1)
svm_x_adv = np.reshape(adv_x, [1000,28*28])
clf = svm.SVC()
clf.fit(svm_x, svm_y)
svm_preds =[]
for i in range(1000):
    svm_pred = clf.predict([svm_x_adv[i]])
    svm_preds.append(svm_pred)
svm_preds = np.array(svm_preds)
print(svm_preds[0].shape)
print(svm_preds[0])
print(svm_preds.shape)
accu = np.equal(np.squeeze(svm_preds),np.argmax(y_test,1))
#print(accu)
print(accu.shape)
accuracy = np.mean(accu)
print(accuracy)
