#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

"""

from __future__ import print_function
import numpy
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import time
import random
import argparse
import logging
import sys
import json
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import tensorflow.contrib.slim as slim
from sklearn.model_selection import KFold, cross_val_score

__author__ = 'hhemken'
log = logging.getLogger(__file__)


class MlpException(Exception):
    """
    Generic exception to be raised within the Mlp class.
    """

    def __init__(self, msg):
        log.exception(msg)
        super(MlpException, self).__init__(msg)


class Mlp(object):
    """

    """

    def __init__(self,
                 output_dir='.',
                 session_name='mlp_run',
                 n_inputs=125,
                 n_outputs=2,
                 checkpoint_path=None,
                 hidden_layers=[150, 75],
                 random_seed=7,
                 data_file_path="../data/fann/mushroom_train-output_last.csv",
                 learning_rate=0.01,
                 n_epochs=400,
                 batch_size=50):
        """

        """
        self.session_name = session_name
        self.output_dir = output_dir
        self.output_layer = None
        self.Xdata = None
        self.ydata = None
        self.accuracy = None
        self.training_op = None
        self.n_samples = 0
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.checkpoint_path = checkpoint_path
        self.hidden_layers = hidden_layers
        self.random_seed = random_seed
        self.data_file_path = data_file_path
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        self.y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")

    def local_leaky_relu(self, z, name='leaky_relu'):
        return tf.nn.leaky_relu(z, name=name, alpha=0.01)

    def construction_phase(self):
        """

        :return:
        """
        with tf.name_scope("dnn"):
            he_init = tf.contrib.layers.variance_scaling_initializer()
            working_layer = self.X
            hidden_layer_num = 1
            for layer_size in self.hidden_layers:
                working_layer = fully_connected(working_layer,
                                                layer_size,
                                                scope="hidden-%d" % hidden_layer_num,
                                                weights_initializer=he_init,
                                                activation_fn=self.local_leaky_relu)
                hidden_layer_num += 1
            self.output_layer = fully_connected(working_layer,
                                                self.n_outputs,
                                                scope="outputs",
                                                weights_initializer=he_init,
                                                activation_fn=self.local_leaky_relu)
        with tf.name_scope("loss"):
            cost = tf.reduce_mean(tf.square(self.output_layer - self.y))

        with tf.name_scope("train"):
            self.training_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        with tf.name_scope("eval"):
            # https://stackoverflow.com/questions/34097457/tensorflow-training
            mse = tf.reduce_mean(tf.square(self.output_layer - self.y))
            self.accuracy = tf.reduce_mean(tf.cast(mse, tf.float32))

    def execution_phase(self):
        """

        :return:
        """
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        random_state = rand.randint(0, 2 ** 32 - 1)
        num_examples, _ = self.Xdata.shape
        num_batches = num_examples / self.batch_size
        log.info("execution phase seed %s", str(random_state))
        log.info('num_batches          %d', num_batches)
        log.info('Xdata: %s %s', type(self.Xdata), str(self.Xdata.shape))
        log.info('ydata: %s %s', type(self.ydata), str(self.ydata.shape))
        with tf.Session() as sess:
            log.info('start session')
            if self.checkpoint_path is not None:
                try:
                    # We can restore the parameters of the network by calling restore on this saver which
                    # is an instance of tf.train.Saver() class.
                    log.info('loading meta graph from %s.meta', self.checkpoint_path)
                    saver = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
                    log.info('restoring checkpoint %s')
                    saver.restore(sess, self.checkpoint_path)
                except Exception as ex_obj:
                    raise MlpException('error loading checkpoint file "%s": %s' % (self.checkpoint_path, str(ex_obj)))
            log.info("Model restored from %s", self.checkpoint_path)
            init.run()
            try:
                for epoch in range(self.n_epochs):
                    log.info('epoch  %6d --------------------------------------------', epoch)
                    # for iteration in range(mnist.train.num_examples // batch_size):
                    for iteration in range(num_batches):
                        # numpy ndarray objects:
                        X_batch, X_test, y_batch, y_test = train_test_split(self.Xdata,
                                                                            self.ydata,
                                                                            test_size=self.batch_size / 10,
                                                                            train_size=self.batch_size,
                                                                            random_state=random_state)
                        sess.run(self.training_op, feed_dict={self.X: X_batch, self.y: y_batch})
                        if iteration == 0:
                            log.info('X_batch: %s %s', type(X_batch), str(X_batch.shape))
                            log.info('y_batch: %s %s', type(y_batch), str(y_batch.shape))
                        if iteration % 10 == 0:
                            acc_train = self.accuracy.eval(feed_dict={self.X: X_batch, self.y: y_batch})
                            acc_test = self.accuracy.eval(feed_dict={self.X: X_test, self.y: y_test})
                            log.info("epoch %6d    batch iteration %6d/%d    error: train %10.6f    test %10.6f",
                                     epoch, iteration, num_batches, acc_train, acc_test)
                    save_path = saver.save(sess, "%s/%s_epoch-%03d.ckpt" % (self.output_dir, self.session_name, epoch))
                    log.info('save_path: %s', str(save_path))
                save_path = saver.save(sess, "%s/%s_final.ckpt" % (self.output_dir, self.session_name))
                log.info('save_path: %s', str(save_path))
            except Exception as ex_obj:
                raise MlpException("error in execution loop: %s" % str(ex_obj))

    def read_data(self):
        """

        :return:
        """
        try:
            log.info('create dataframe')
            dataframe = pandas.read_csv(self.data_file_path, delim_whitespace=True, header=None)
            log.info('get dataframe values')
            dataset = dataframe.values
        except Exception as ex_obj:
            raise MlpException("error creating dataframe: %s" % str(ex_obj))
        try:
            log.info('split into input (X) and output (Y) variables')
            self.Xdata = dataset[:, 0:self.n_inputs]
            self.n_samples = self.Xdata.shape[0]
            log.info('get Y data')
            # Y data are at end of each row
            # notation: http://structure.usc.edu/numarray/node26.html
            if self.n_outputs > 1:
                # get columns from column index inputs to column index inputs + outputs fom matrix of data
                log.info('> 1 outputs')
                self.ydata = dataset[:, self.n_inputs:self.n_inputs + self.n_outputs]
            else:
                # get column at column index inputs from matrix of data
                log.info('1 output')
                self.ydata = dataset[:, self.n_inputs]
        except Exception as ex_obj:
            raise MlpException("error splitting x and y values: %s" % str(ex_obj))
        log.info('Xdata: %s %s', type(self.Xdata), str(self.Xdata.shape))
        log.info('ydata: %s %s', type(self.ydata), str(self.ydata.shape))

    def load_checkpoint(self, checkpoint_path):
        """

        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                # Restore variables from disk.
                saver.restore(sess, checkpoint_path)
            except Exception as ex_obj:
                raise MlpException('error loading checkpoint file "%s": %s' % (checkpoint_path, str(ex_obj)))
        log.info("Model restored from %s", checkpoint_path)


def test_housing():
    """

    :return:
    """
    # load housing dataset
    inputs = 13
    outputs = 1
    hidden_layers = [6]
    random_seed = 7
    data_file_path = "../data/fann/housing.csv"
    learning_rate = 0.01
    n_epochs = 400
    batch_size = 50
    log.info('create mlp object')
    mlp = Mlp(n_inputs=inputs,
              n_outputs=outputs,
              hidden_layers=hidden_layers,
              random_seed=random_seed,
              data_file_path=data_file_path,
              learning_rate=learning_rate,
              n_epochs=n_epochs,
              batch_size=batch_size)
    log.info('read data')
    mlp.read_data()
    log.info('construction phase')
    mlp.construction_phase()
    log.info('execution phase')
    mlp.execution_phase()


def test_mushroom():
    """

    :return:
    """
    # # load mushroom dataset
    inputs = 125
    outputs = 2
    hidden_layers = [150, 75]
    random_seed = 7
    data_file_path = "../data/fann/mushroom_train-output_last.csv"
    learning_rate = 0.01
    n_epochs = 400
    batch_size = 50
    log.info('create mlp object')
    mlp = Mlp(n_inputs=inputs,
              n_outputs=outputs,
              hidden_layers=hidden_layers,
              random_seed=random_seed,
              data_file_path=data_file_path,
              learning_rate=learning_rate,
              n_epochs=n_epochs,
              batch_size=batch_size)
    log.info('read data')
    mlp.read_data()
    log.info('construction phase')
    mlp.construction_phase()
    log.info('execution phase')
    mlp.execution_phase()


if __name__ == '__main__':
    log_format = ("%(asctime)s.%(msecs)03d [%(process)d] %(threadName)s: %(levelname)-06s: " +
                  "%(module)s::%(funcName)s:%(lineno)s: %(message)s")
    log_datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        datefmt=log_datefmt)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_file_path', default=False,
                               help="Path of JSON data file.")
    parent_parser.add_argument('--output_dir', default=False,
                               help="Directory in which output files will be written.")
    parent_parser.add_argument('--num_ann_inputs', default=False,
                               help="Have another option on the command line.")
    parent_parser.add_argument('--num_ann_outputs', default=False,
                               help="Have another option on the command line.")
    parent_parser.add_argument('--num_input_files', default=False,
                               help="Have another option on the command line.")
    parent_parser.add_argument('--hidden_layers', default=False,
                               help="JSON representation of a list of hidden layer sizes.")
    parent_parser.add_argument('--session_name', default=False,
                               help="Name of the session, used to name saved models and outputs.")
    parent_parser.add_argument('--learning_rate', default=False,
                               help="Fixed learning rate to use.")
    parent_parser.add_argument('--n_epochs', default=False,
                               help="Number of epochs to run.")
    parent_parser.add_argument('--batch_size', default=False,
                               help="Batch size.")
    parent_parser.add_argument('--checkpoint_path', default=None,
                               help="File system path to checkpoint file.")
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='command')
    # test_mushroom command line command
    test_mushroom_parser = subparsers.add_parser('test_mushroom', parents=[parent_parser],
                                                 help="Will run test_mushroom.")
    # test_housing command line command
    test_housing_parser = subparsers.add_parser('test_housing', parents=[parent_parser], help="Will run test_housing.")
    # run_mlp command line command
    run_mlp_parser = subparsers.add_parser('run_mlp', parents=[parent_parser],
                                           help="Will run mlp using input parameters.")
    # run_inference command line command
    run_inference_parser = subparsers.add_parser('run_inference', parents=[parent_parser],
                                                 help="Will run mlp using input parameters.")

    rand = random.SystemRandom()
    args = parser.parse_args()
    log.info('args: %s', str(args))

    exit_code = 1
    try:
        if args.command == 'test_mushroom':
            log.info("running test_mushroom")
            test_mushroom()
        elif args.command == 'test_housing':
            log.info("running test_housing")
            test_housing()
        elif args.command == 'run_mlp':
            command_line_random_seed = rand.randint(0, 2 ** 32 - 1)
            log.info("random seed          %s", str(command_line_random_seed))
            log.info("session name         %s", str(args.session_name))
            log.info("data file path       %s", args.data_file_path)
            log.info("output directory     %s", args.output_dir)
            log.info("num_ann_inputs       %s", args.num_ann_inputs)
            log.info("num_ann_outputs      %s", args.num_ann_outputs)
            log.info("session_name         %s", args.session_name)
            log.info("learning rate        %s", args.learning_rate)
            log.info("checkpoint path      %s", args.checkpoint_path)
            log.info("hidden_layers string %s", args.hidden_layers)
            actual_hidden_layers = json.loads(args.hidden_layers)
            log.info("hidden_layers        %s", str(actual_hidden_layers))
            log.info('create mlp object')
            command_line_mlp = Mlp(session_name=args.session_name,
                                   output_dir=args.output_dir,
                                   n_inputs=int(args.num_ann_inputs),
                                   n_outputs=int(args.num_ann_outputs),
                                   checkpoint_path=args.checkpoint_path,
                                   hidden_layers=actual_hidden_layers,
                                   random_seed=int(command_line_random_seed),
                                   data_file_path=args.data_file_path,
                                   learning_rate=float(args.learning_rate),
                                   n_epochs=int(args.n_epochs),
                                   batch_size=int(args.batch_size))
            log.info('read data')
            command_line_mlp.read_data()
            log.info('construction phase')
            command_line_mlp.construction_phase()
            log.info('execution phase')
            command_line_mlp.execution_phase()
        elif args.command == 'run_inference':
            log.info("running run_inference")
            command_line_mlp = Mlp()
            command_line_mlp.load_checkpoint(args.checkpoint_path)

        exit_code = 0
    except MlpException as keras_regression_exception:
        log.exception(keras_regression_exception)
    except Exception as generic_exception:
        logging.exception(generic_exception)
    finally:
        logging.shutdown()
        sys.exit(exit_code)
