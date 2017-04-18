import tf_utils
import numpy as np
import os
import subprocess
import collections
import copy
import tensorflow as tf
# from path import project_dir, tf_data_dir
import cv2

# from path import project_dir
tf_data_dir = "/home/ashvin/tf-poke/tf-data/"

slim = tf.contrib.slim
# from nets import alexnet_conv, inception_v3_conv, vgg_conv, vgg_16, alexnet_geurzhoy

from data import mnist_data

import lenet
import rcnn

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops

def get_default_conf(**kwargs):
    conf = collections.OrderedDict()
    conf['network'] = 'lenet'
    conf['data'] = 'mnist'
    conf['batch']= 100
    conf['initLr'] = 0.0001
    conf['run'] = 0
    conf['length'] = None # layers per module
    conf['module'] = None
    conf['share'] = None
    conf['channels'] = None
    conf['resid'] = None
    conf['dropout'] = None

    for arg in kwargs:
        assert arg in conf
        conf[arg] = kwargs[arg]

    return conf

class DigitsModel(object):
    def __init__(self, train_conf, test_conf = {}, erase_model=False):
        print "setting up network"
        self.name = tf_utils.dict_to_string(train_conf)
        self.network = tf_utils.TFNet(self.name,
            logDir= tf_data_dir + 'tf_training/openai/',
            modelDir= tf_data_dir + 'tf_models/',
            outputDir= tf_data_dir + 'tf_outputs/',
            eraseModels=erase_model)

        self.conf = train_conf
        self.batch_size = self.conf['batch']

        # image_batch  = tf.placeholder("float", [None, 15, 64, 64, 3])
        # action_batch = tf.placeholder("float", [None, 15, 2])
        # self.inputs = list(read_tf_record.build_tfrecord_input(self.conf, training=True))

        X = tf.placeholder("float", [None, 28*28])
        Y = tf.placeholder("float", [None, 10])
        image = tf.reshape(X, [self.batch_size, 28, 28, 1])
        # tf.image_summary("image", image)

        self.inputs = [X, Y]

        with slim.arg_scope(
              [slim.conv2d, slim.fully_connected],
              weights_regularizer=slim.l2_regularizer(0.00001),
              weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
              activation_fn=tf.nn.relu) as sc:
            if self.conf['network'] == 'lenet':
                preds = lenet.lenet(image)
            if self.conf['network'] == 'lenet2':
                preds = lenet.lenet2(image, self.conf['share'])
            if self.conf['network'] == 'lenet3':
                preds = lenet.lenet3(image, self.conf['share'])
            if self.conf['network'] == 'rcnn':
                preds =rcnn.rcnn(image, self.conf['module'], self.conf['length'], self.conf['share'], self.conf['channels'], self.conf['resid'])

        correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(preds, Y))

        self.network.add_to_losses(self.loss)

        # make a training network
        self.train_network = tf_utils.TFTrain(self.inputs, self.network, batchSz=self.batch_size, initLr=self.conf['initLr'])
        self.train_network.add_loss_summaries([self.loss, self.accuracy], ['loss', 'acc'])

        self.outputs = [self.loss, self.accuracy, preds]
        self.output_names = ['loss', 'acc', 'pred']
        print "done with network setup"

    def init_sess(self):
        self.sess = tf.Session() # tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # tf.train.start_queue_runners(self.sess)
        self.sess.run(tf.initialize_all_variables())

    def train_batch(self, inputs, batch_size, isTrain):
        # image_batch, action_batch, state_batch, object_pos_batch = inputs
        # image_data, action_data, state_data, object_pos = self.sess.run([image_batch, action_batch, state_batch, object_pos_batch])
        # feed_dict = {image_batch: image_data, action_batch: action_data}
        # return feed_dict
        X, Y = inputs
        name = "train" if isTrain else "test"
        x_batch, y_batch = mnist_data.get_batch(name, self.conf['batch'])
        return {X: x_batch, Y: y_batch}

    def train(self, max_iters = 100000, use_existing = True):
        self.init_sess()
        self.train_network.maxIter_ = max_iters
        self.train_network.dispIter_ = 100
        self.train_network.train(self.train_batch, self.train_batch, use_existing=use_existing, sess=self.sess)

    def run(self, dataset, batches=1, i = None, sess=None):
        """Return batches*batch_size examples from the dataset ("train" or "val")
        i: specific model to restore, or restores the latest
        """
        f = self.get_f(i, sess)
        if not f:
            return None

        ret = []

        for j in range(batches):
            tb = self.train_batch(self.inputs, self.batch_size, dataset=="train")
            inps, result = f(tb)
            ret.append([inps, result])

        return ret

    def get_f(self, i = None, sess=None):
        """Return the network forward function"""
        ret = []
        if not sess:
            sess = tf.Session()
            sess.run(tf.initialize_all_variables())
            restore = self.network.restore_model(sess, i)
            if i and not restore: # model requested but not found
                return None

        def f(training_batch):
            feed_dict = training_batch
            result = sess.run(self.outputs, feed_dict)
            inps = [feed_dict[x] for x in self.inputs]
            out = {}
            for i, name in enumerate(self.output_names):
                out[name] = result[i]
            return inps, out

        return f

    def get_log(self):
        # name = params_to_name(self.params)
        network = tf_utils.TFNet(self.name,
            logDir= tf_data_dir + 'tf_training/openai/',
            modelDir= project_dir + 'tf_models/',
            outputDir= project_dir + 'tf_outputs/',)
        log_name = network.get_log_name()
        print log_name
        return tf_utils.TFSummary(log_name[0])

    def evaluate(self, i, batches, dataset="val", sess=None):
        """Returns validation accuracy on #batches of saved model i"""
        out = self.run(dataset, batches, i, self.sess)
        if not out:
            return None

        accs = 0
        for inputs, results in out:
            accs += results["acc"]
        return accs/batches
