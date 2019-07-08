# This file is largely copied from pgd_attack.py found at
# githu.com/MadryLab/mnist_challenge. Relevant Paper:
# A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. Towards Deep Learning
# Models Resistant to Adversarial Attacks. ICLR 2018.
# I've changed it to work with more general input spaces and convolutional
# networks.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class IntervalPGDAttack:
    def __init__(self, model, k, a, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.k = k
        self.a = a
        self.rand = random_start

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]

        with tf.Session() as sess:
            self.graph = sess.graph

    def perturb(self, x_nat, y, lower, upper, sess):
        if self.rand:
            # Picks a uniformly distributed random point inside the region
            #x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.random.uniform(lower, upper, x_nat.shape);
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            x += self.a * np.sign(grad)

            x = np.clip(x, lower, upper)
            if x_nat.shape[1] == 5:
                x = np.clip(x, -1, 1)
            else:
                x = np.clip(x, 0, 1) # ensure valid pixel range

        return x

class Model:
    def __init__(self, layers):
        # A layer is a tuple
        # (type, (in_height, in_width, in_depth) (out_height, out_width, out_depth) ...)
        # where type is one of 'conv', 'fc', or 'maxpool'.
        # If type is 'fc' then the tuple is ('fc', in_size, out_size, weight, bias)
        # If type is 'conv' then the tuple is ('conv', in_size, out_size, filter, biases)
        # If type is 'maxpool' then the tuple is ('maxpool', in_size, out_size, height_stride, width_stride)
        in_size = layers[0][1][0] * layers[0][1][1] * layers[0][1][2]
        out_size = layers[-1][2][0] * layers[-1][2][1] * layers[-1][2][2]
        self.x_input = tf.placeholder(tf.float32, shape = [None, in_size])
        self.y_input = tf.placeholder(tf.int64, shape = [None])

        self.hs = [self.x_input]

        for i in range(len(layers)):
            l = layers[i]
            if l[0] == 'conv':
                tmp = tf.reshape(self.hs[-1], [-1, l[1][0], l[1][1], l[1][2]])
                cl = tf.nn.conv2d(
                        input=tmp,
                        filter=tf.convert_to_tensor(np.array(l[3]), dtype=tf.float32),
                        strides=[1, 1, 1, 1],
                        padding='VALID')
                bias = tf.nn.bias_add(
                        value=cl,
                        bias=tf.convert_to_tensor(np.array(l[4]), dtype=tf.float32))
                if i < len(layers) - 1:
                    self.hs.append(tf.nn.relu(bias))
                else:
                    self.hs.append(bias)
            elif l[0] == 'maxpool':
                self.hs.append(tf.nn.max_pool(
                    value=self.hs[-1],
                    ksize=[1, l[3], l[4], 1],
                    strides=[1, l[4], l[4], 1],
                    padding='VALID'))
            else:
                tmp = tf.reshape(self.hs[-1], [-1, l[1][0] * l[1][1] * l[1][2]])
                w = tf.convert_to_tensor(np.transpose(np.array(l[3])), dtype=tf.float32)
                b = tf.convert_to_tensor(np.array(l[4]), dtype=tf.float32)
                if i < len(layers) - 1:
                    self.hs.append(tf.nn.relu(tf.nn.xw_plus_b(tmp, w, b)))
                else:
                    self.hs.append(tf.nn.xw_plus_b(tmp, w, b))

        self.pre_softmax = tf.reshape(self.hs[-1], [-1, out_size])
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = self.y_input, logits = self.pre_softmax)
        self.xent = tf.reduce_sum(y_xent)
        self.y_pred = tf.argmax(self.pre_softmax, 1)
        correct_prediction = tf.equal(self.y_pred, self.y_input)
        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def initialize_pgd_class(layers):
    model = Model(layers)
    attack = IntervalPGDAttack(model, 500, 0.01, False, 'xent')
    return attack

def run_attack(attack, x, y, lower, upper):
    with tf.Session(graph=attack.graph) as sess:
        x_adv = attack.perturb(
                np.array(x)[np.newaxis],
                np.array([y]),
                np.array(lower),
                np.array(upper),
                sess)
    return x_adv.flatten().tolist()

def run_model(attack, x, y):
    with tf.Session(graph=attack.graph) as sess:
        score = sess.run(attack.model.pre_softmax,
                feed_dict = {attack.model.x_input: np.array(x)[np.newaxis],
                             attack.model.y_input: np.array([y])})
    return score.flatten().tolist()

