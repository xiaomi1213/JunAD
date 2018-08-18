import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os
from show_image import show_a_image
from svm_loss_test import svm_l1loss, svm_l2loss
from sklearn import svm
import numpy as np
import foolbox
"""
x = [[1,2,3,4],[5,6,2,7]]
#y = [[1,2,3,4],[5,6,2,7]]
y = [[3,4,5,6],[4,3,2,2]]
correct_prediction = tf.equal(tf.argmax(x,axis=1),tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:

    acc = sess.run(accuracy)
    print(acc)


sess = tf.Session()

class Test(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [1])
        self.y = tf.placeholder(tf.float32, [1])

    def get(self,x_input):

        y = self.x + [1]
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            preds = sess.run(y,feed_dict={self.x:x_input})
            return preds

test = Test()
inputs = [1]
predict = test.get(x_input=inputs)
print(predict)


class A(object):
    def __init__(self):
        pass
    def get_A(self,a):
        b = a+1
        return b

class B(object):
    def __init__(self):
        pass
    def get_B(self,b):
        c = b-1
        return c

class C(A, B):
    def __init__(self):
        pass

c = C()
d = c.get_A(1)
print('class A: %d' % d)

e = c.get_B(1)
print('class B: %d' % e)

preds = [False, False, False, True, True, False, False,]


ui = [2,3,4,8,6]
uj = [1,3,6,8,9]
u = np.equal(ui, uj)
usum = np.sum(u)
print(u, usum)

x1 = tf.placeholder([None])
x2 = tf.placeholder([None])

def a1(a, b):
    a = tf.constant(a)
    b = tf.constant(b)
    c = a+b
    return c

def a2(d):
    c = a1(a, b)
    e = c+d

t = tf.constant([1])
with tf.Session() as sess1:
    sess1.run(tf.variables_initializer([tf.Variable(t)]))
    a = sess.run(t)
print(a)



from tqdm import tnrange
from time import sleep

for i in tnrange(4, desc='1st loop'):
    for j in tnrange(10, desc='2nd loop'):
        sleep(0.01)

"""


a = np.array([1,2,3])
print(len(a))