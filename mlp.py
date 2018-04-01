#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Example Keras MLP regression script with inference, reloading, re-fitting, and re-inferencing
    Based on https://www.learnopencv.com/deep-learning-using-keras-the-basics/
"""

# ToDo:
# reporting, tensorboard
# runs with existing self.model
# vary number of epochs
# make into parameterized class & helper functions
# run several rounds with different training data - done
# systematically vary learning rate, optimizer, etc - done


import sys
import argparse
import datetime
import logging
import random
import datetime
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing
from keras.models import load_model
from keras.callbacks import CSVLogger, TensorBoard
from keras import optimizers

__author__ = 'hhemken'

log = logging.getLogger(__file__)
rand = random.SystemRandom()
# ======================================================================================================================
# https://keras.io/optimizers/
# Parameters common to all Keras optimizers
# The parameters clipnorm and clipvalue can be used with all optimizers to control gradient clipping:
# All parameter gradients will be clipped to
# a maximum norm of 1.
#    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
#    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

# SGD
# Stochastic gradient descent optimizer.
# Includes support for momentum, learning rate decay, and Nesterov momentum.
# Arguments
# 	lr: float >= 0. Learning rate.
# 	momentum: float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
# 	decay: float >= 0. Learning rate decay over each update.
# 	nesterov: boolean. Whether to apply Nesterov momentum.
sgd_opt = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# ----------------------------------------------------------------------------------------------------------------------
# RMSprop
# RMSProp optimizer.
# It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which
#   can be freely tuned).
# This optimizer is usually a good choice for recurrent neural networks.
# Arguments
# 	lr: float >= 0. Learning rate.
# 	rho: float >= 0.
# 	epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
# 	decay: float >= 0. Learning rate decay over each update.
# References
# rmsprop: Divide the gradient by a running average of its recent magnitude
# http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
rmsprop_opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# ----------------------------------------------------------------------------------------------------------------------
# Adagrad
# Adagrad optimizer.
# It is recommended to leave the parameters of this optimizer at their default values.
# Arguments
# 	lr: float >= 0. Learning rate.
# 	epsilon: float >= 0. If None, defaults to K.epsilon().
# 	decay: float >= 0. Learning rate decay over each update.
# References
# Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
# http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
adagrad_opt = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
# ----------------------------------------------------------------------------------------------------------------------
# Adadelta
# Adadelta optimizer.
# It is recommended to leave the parameters of this optimizer at their default values.
# Arguments
# 	lr: float >= 0. Learning rate. It is recommended to leave it at the default value.
# 	rho: float >= 0.
# 	epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
# 	decay: float >= 0. Learning rate decay over each update.
# References
# Adadelta - an adaptive learning rate method
# http://arxiv.org/abs/1212.5701
adadelta_opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# ----------------------------------------------------------------------------------------------------------------------
# Adam
# Adam optimizer.
# Default parameters follow those provided in the original paper.
# Arguments
# 	lr: float >= 0. Learning rate.
# 	beta_1: float, 0 < beta < 1. Generally close to 1.
# 	beta_2: float, 0 < beta < 1. Generally close to 1.
# 	epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
# 	decay: float >= 0. Learning rate decay over each update.
# 	amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence
#       of Adam and Beyond".
# References
# Adam - A Method for Stochastic Optimization
#    http://arxiv.org/abs/1412.6980v8
# On the Convergence of Adam and Beyond
#    https://openreview.net/forum?id=ryQu7f-RZ
adam_opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# ----------------------------------------------------------------------------------------------------------------------
# Adamax
# Adamax optimizer from Adam paper's Section 7.
# It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the paper.
# Arguments
# 	lr: float >= 0. Learning rate.
# 	beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
# 	epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
# 	decay: float >= 0. Learning rate decay over each update.
# References
# Adam - A Method for Stochastic Optimization
# http://arxiv.org/abs/1412.6980v8
adamax_opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# ----------------------------------------------------------------------------------------------------------------------
# Nadam
# Nesterov Adam optimizer.
# Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum.
# Default parameters follow those provided in the paper. It is recommended to leave the parameters of this
#   optimizer at their default values.
# Arguments
# 	lr: float >= 0. Learning rate.
# 	beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
# 	epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
# References
# Nadam report
# 	http://cs229.stanford.edu/proj2015/054_report.pdf
# On the importance of initialization and momentum in deep learning
# 	http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
nadam_opt = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# ----------------------------------------------------------------------------------------------------------------------
# TFOptimizer
# Wrapper class for native TensorFlow optimizers.
# optimizers.TFOptimizer(optimizer)
# ----------------------------------------------------------------------------------------------------------------------
AVAILABLE_OPTIMIZERS = {'adam': adam_opt,
                        'rmsprop': rmsprop_opt,
                        'adamax': adamax_opt,
                        'nadam': nadam_opt,
                        'adagrad': adagrad_opt,
                        'adadelta': adadelta_opt,
                        'sgd': sgd_opt}

# https://keras.io/activations/
AVAILABLE_ACTIVATIONS = ['elu', 'relu', 'selu']

# https://keras.io/initializers/
AVAILABLE_INITIALIZERS = ['lecun_uniform',
                          'lecun_normal',
                          'glorot_uniform',
                          'glorot_normal',
                          'he_uniform',
                          'he_normal']


# ======================================================================================================================


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

    BATCH_SIZE_DIVISOR = 50
    DEFAULT_REPLICATES = 3
    EPOCHS_TO_RUN = 1000

    def __init__(self):
        """

       """
        self.x_train = None
        self.y_train = None
        self.x_validate = None
        self.y_validate = None
        self.x_test = None
        self.y_test = None
        self.inputs = 32 * 32
        self.outputs = 64 * 64
        self.model = None
        self.num_examples = None
        self.num_features = None
        self.train_data_file_path = "ann_test_data/ann_data-32x64-d.dat"
        self.test_data_file_path = "ann_test_data/ann_data-32x64-a.dat"

    def create_model(self,
                     activation=AVAILABLE_ACTIVATIONS,
                     initializer=AVAILABLE_INITIALIZERS,
                     num_hidden_layers=1,
                     optimizer=AVAILABLE_OPTIMIZERS):
        """
        
        :return: 
        """
        log.info('%s', '=' * 80)
        log.info('create model')
        self.model = Sequential()
        # self.model.add(Dense(1, input_shape=(num_features,), activation='linear'))
        log.info('create input layer')
        self.model.add(
            Dense(self.inputs, input_shape=(self.inputs,), kernel_initializer=initializer, activation=activation))
        for i in range(num_hidden_layers):
            log.info('create hidden layer %d', i)
            self.model.add(Dense(self.outputs,
                                 input_shape=(self.outputs,),
                                 kernel_initializer=initializer,
                                 activation=activation))
        log.info('create output layer')
        self.model.add(
            Dense(self.outputs, input_shape=(self.outputs,), kernel_initializer=initializer, activation=activation))
        log.info('compile network')
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae', 'acc'])

    def fit_model(self, epochs_to_run=EPOCHS_TO_RUN, output_file_root=None):
        """

        :return:
        """
        callbacks = list()
        tb_callback = TensorBoard(log_dir='./tensorboard',
                                  histogram_freq=250,
                                  batch_size=32,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=True,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)
        # callbacks.append(tb_callback)
        if output_file_root:
            csv_logger = CSVLogger('%s-epochs.csv' % output_file_root)
            callbacks.append(csv_logger)
        log.info('%s', '=' * 80)
        log.info('fit')
        batch_size = (self.num_examples / self.BATCH_SIZE_DIVISOR if self.num_examples > self.BATCH_SIZE_DIVISOR
                      else self.BATCH_SIZE_DIVISOR)
        batch_size = max(batch_size, 10)
        log.info('batch size        %d', batch_size)
        log.info('epochs to run     %d', epochs_to_run)
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs_to_run,
                                 callbacks=callbacks)
        log.info('%s', '=' * 80)
        log.info('history: %s', str(history))
        log.info('%s', '=' * 80)
        log.info('summary')
        self.model.summary()
        log.info('%s', '=' * 80)
        log.info('evaluate the model')
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        log.info('scores')
        log.info("%s: %.2f%%", self.model.metrics_names[1], scores[1] * 100)
        log.info('current cv score: %.3f', scores[1] * 100.0)
        for i in range(len(scores)):
            log.info("metric %12s: %10.6f%%", self.model.metrics_names[i], scores[i])

    def load_csv(self, data_file_path):
        """

        :return:
        """
        # Arguments:
        #     path: path where to cache the dataset locally (relative to ~/.keras/datasets).
        #     seed: Random seed for shuffling the data before computing the test split.
        #     test_split: fraction of the data to reserve as test set.
        # Returns: Tuple of Numpy arrays: (self.x_train, self.y_train), (self.x_test, self.y_test).
        # Data file in numpy.savez; savez_compressed option saves several arrays into a compressed .npz archive
        #     hhemken@esx11:~/workspace/deep_learning$ ls -l ~/.keras/datasets
        #     total 56
        #     -rw-rw-r-- 1 hhemken hhemken 57026 Mar 16 13:41 boston_housing.npz
        # (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
        log.info('load dataframe from csv file %s', data_file_path)
        dataframe = pandas.read_csv(data_file_path, delim_whitespace=False, header=None)
        log.info('get dataframe values')
        dataset = dataframe.values
        log.info('split into input (X) and output (Y) variables')
        x_data = dataset[:, 0:self.inputs]
        log.info('get Y data')
        # y data are at end of each row
        # notation: http://structure.usc.edu/numarray/node26.html
        if self.outputs > 1:
            # get columns from column index self.inputs to column index self.inputs + self.outputs fom matrix of data
            log.info('> 1 outputs')
            y_data = dataset[:, self.inputs:self.inputs + self.outputs]
        else:
            # get column at column index self.inputs from matrix of data
            log.info('1 output')
            y_data = dataset[:, self.inputs]
        return x_data, y_data

    def load_model(self, model_name):
        """

        :return:
        """
        log.info('%s', '=' * 80)
        log.info('load model        %s.h5', model_name)
        self.model = load_model('%s.h5' % model_name)

    def continued_training(self,
                           training_data_files,
                           test_data_file,
                           initial_model_file,
                           output_file_root,
                           epochs_to_run=EPOCHS_TO_RUN):
        """

        :return:
        """
        # ---------------------------------------------------------------------------------------------------------------
        model_file = initial_model_file
        for data_file in training_data_files:
            log.info('%s', '=' * 80)
            log.info('training data     %s', data_file)
            self.x_train, self.y_train = self.load_csv(data_file)
            log.info('test data         %s', test_data_file)
            self.x_test, self.y_test = self.load_csv(test_data_file)
            self.num_examples = self.x_train.shape[0]
            self.num_features = self.x_train.shape[1]

            log.info('x_train shape: %s', self.x_train.shape)
            log.info('y_train shape: %s', self.y_train.shape)
            log.info('x_test shape:  %s', self.x_test.shape)
            log.info('y_test shape:  %s', self.y_test.shape)
            log.info('num examples:  %d', self.num_examples)
            log.info('num features:  %d', self.num_features)

            self.load_model(model_file)
            self.fit_model(epochs_to_run=epochs_to_run)
            self.test_model()
            self.save_model(output_file_root)
            model_file = output_file_root
        # ---------------------------------------------------------------------------------------------------------------

    def initial_training(self,
                         optimizer,
                         activation,
                         initializer,
                         num_hidden_layers,
                         training_data_files,
                         test_data_file,
                         output_file_root,
                         epochs_to_run=EPOCHS_TO_RUN):
        """

        :return:
        """
        # ---------------------------------------------------------------------------------------------------------------
        model_file = None
        for data_file in training_data_files:
            log.info('%s', '=' * 80)
            log.info('training data     %s', data_file)
            self.x_train, self.y_train = self.load_csv(data_file)
            log.info('test data         %s', test_data_file)
            self.x_test, self.y_test = self.load_csv(test_data_file)
            self.num_examples = self.x_train.shape[0]
            self.num_features = self.x_train.shape[1]

            log.info('x_train shape: %s', self.x_train.shape)
            log.info('y_train shape: %s', self.y_train.shape)
            log.info('x_test shape:  %s', self.x_test.shape)
            log.info('y_test shape:  %s', self.y_test.shape)
            log.info('num examples:  %d', self.num_examples)
            log.info('num features:  %d', self.num_features)

            if model_file:
                self.load_model(model_file)
            else:
                self.create_model(optimizer=optimizer,
                                  activation=activation,
                                  initializer=initializer,
                                  num_hidden_layers=num_hidden_layers)
            self.fit_model(epochs_to_run=epochs_to_run, output_file_root=output_file_root)
            self.test_model()
            self.save_model(output_file_root)
            model_file = output_file_root
        # ---------------------------------------------------------------------------------------------------------------

    def orig(self):
        """

        :return:
        """
        # ---------------------------------------------------------------------------------------------------------------
        log.info('%s', '=' * 80)
        log.info('load test data')
        log.info('create training dataframe')
        self.x_train, self.y_train = self.load_csv('ann_test_data/ann_data-32x64-h.dat')
        log.info('create test dataframe')
        self.x_test, self.y_test = self.load_csv(self.test_data_file_path)
        self.num_examples = self.x_train.shape[0]
        self.num_features = self.x_train.shape[1]

        log.info('x_train shape: %s', self.x_train.shape)
        log.info('y_train shape: %s', self.y_train.shape)
        log.info('x_test shape:  %s', self.x_test.shape)
        log.info('y_test shape:  %s', self.y_test.shape)
        log.info('num examples:  %d', self.num_examples)
        log.info('num features:  %d', self.num_features)

        self.load_model('ann_data-32x64-g')
        # self.create_model()
        self.fit_model()
        self.test_model()
        self.save_model('ann_data-32x64-h')
        # ---------------------------------------------------------------------------------------------------------------
        self.load_model('ann_data-32x64-h')
        log.info('%s', '=' * 80)
        log.info('load test data')
        log.info('create training dataframe')
        self.x_train, self.y_train = self.load_csv('ann_test_data/ann_data-32x64-i.dat')
        self.num_examples = self.x_train.shape[0]
        self.num_features = self.x_train.shape[1]

        log.info('x_train shape: %s', self.x_train.shape)
        log.info('y_train shape: %s', self.y_train.shape)
        log.info('num examples:  %d', self.num_examples)
        log.info('num features:  %d', self.num_features)

        self.fit_model()
        log.info('%s', '=' * 80)
        log.info('inference')
        self.test_model()
        self.save_model('ann_data-32x64-i')
        # ---------------------------------------------------------------------------------------------------------------
        self.load_model('ann_data-32x64-i')
        log.info('%s', '=' * 80)
        log.info('load test data')
        log.info('create training dataframe')
        self.x_train, self.y_train = self.load_csv('ann_test_data/ann_data-32x64-j.dat')
        self.num_examples = self.x_train.shape[0]
        self.num_features = self.x_train.shape[1]

        log.info('x_train shape: %s', self.x_train.shape)
        log.info('y_train shape: %s', self.y_train.shape)
        log.info('num examples:  %d', self.num_examples)
        log.info('num features:  %d', self.num_features)

        self.fit_model()

        log.info('%s', '=' * 80)
        log.info('inference')
        self.test_model()
        self.save_model('ann_data-32x64-j')
        # ---------------------------------------------------------------------------------------------------------------
        self.load_model('ann_data-32x64-j')
        log.info('%s', '=' * 80)
        log.info('load test data')
        log.info('create training dataframe')
        self.x_train, self.y_train = self.load_csv('ann_test_data/ann_data-32x64-k.dat')
        self.num_examples = self.x_train.shape[0]
        self.num_features = self.x_train.shape[1]

        log.info('x_train shape: %s', self.x_train.shape)
        log.info('y_train shape: %s', self.y_train.shape)
        log.info('x_test shape:  %s', self.x_test.shape)
        log.info('y_test shape:  %s', self.y_test.shape)
        log.info('num examples:  %d', self.num_examples)
        log.info('num features:  %d', self.num_features)

        self.fit_model()

        log.info('%s', '=' * 80)
        log.info('more inference')
        self.test_model()
        self.save_model('ann_data-32x64-k')

    def search_for_model(self,
                         training_data_files,
                         test_data_file,
                         final_model_file_root,
                         epochs,
                         hidden_layer_numbers,
                         optimizers_to_try=AVAILABLE_OPTIMIZERS,
                         activations_to_try=AVAILABLE_ACTIVATIONS,
                         initializers_to_try=AVAILABLE_INITIALIZERS,
                         num_replicates=DEFAULT_REPLICATES):
        """

        :return:
        """
        replicate_params = dict()
        log.info('%s', ' = =' * 20)
        for optimizer in optimizers_to_try:
            log.info('%s', ' - -' * 20)
            log.info('optimizer         %s', optimizer)
            replicate_params['optimizer'] = optimizer
            for num_hidden_layers in hidden_layer_numbers:
                log.info('num hidden layers %d', num_hidden_layers)
                replicate_params['hidden layers'] = str(num_hidden_layers)
                for activation in activations_to_try:
                    log.info('activation        %s', activation)
                    replicate_params['activation'] = activation
                    for initializer in initializers_to_try:
                        log.info('initializer       %s', initializer)
                        replicate_params['initializer'] = initializer
                        for replicate in range(num_replicates):
                            log.info('%s', ' . .' * 20)
                            log.info('replicate         %d/%d', replicate + 1, num_replicates)
                            log.info('optimizer         %s', replicate_params['optimizer'])
                            log.info('activation        %s', replicate_params['activation'])
                            log.info('initializer       %s', replicate_params['initializer'])
                            log.info('num hidden layers %s', replicate_params['hidden layers'])
                            output_file_root = ('%s-%s-%02d-hl-%02d-repl' %
                                                (final_model_file_root, optimizer, num_hidden_layers, replicate))
                            log.info('output file root  %s', output_file_root)
                            self.initial_training(
                                optimizers_to_try[optimizer],
                                activation,
                                initializer,
                                num_hidden_layers,
                                training_data_files,
                                test_data_file,
                                output_file_root,
                                epochs_to_run=epochs)

    def save_model(self, model_name):
        """

        :return:
        """
        log.info('%s', '=' * 80)
        log.info('save model %s.h5', model_name)
        self.model.save('%s.h5' % model_name)

    def test_model(self):
        """

        :return:
        """
        log.info('%s', '=' * 80)
        log.info('inference')
        self.model.evaluate(self.x_test, self.y_test, verbose=True)
        y_pred = self.model.predict(self.x_test)
        log.info(self.y_test[:5])
        log.info(y_pred[:5, 0])


if __name__ == '__main__':
    start_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # set up logging to file - see previous section for more details
    format_string = '%(asctime)s %(name)-12s %(levelname)-8s %(module)s::%(funcName)s:%(lineno)s %(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=format_string,
                        datefmt='%Y-%m-%dT%H:%M:%S',
                        filename='mlp-%s.log' % start_datetime,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(format_string)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--input_directory',
                               help="Directory with image files to draw upon.")
    parent_parser.add_argument('--output_file_path',
                               help="Path of final data file.")
    parent_parser.add_argument('--num_inputs',
                               help="Dimension of image input to ANN.")
    parent_parser.add_argument('--num_outputs',
                               help="Dimension of image output from ANN.")
    parent_parser.add_argument('--max_input_files',
                               help="Maximum number of input files to sample")
    parent_parser.add_argument('--requested_samples',
                               help="Number of examples desired in output file.")
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='command')
    # test_mushroom command line command
    test_mushroom_parser = subparsers.add_parser('create', parents=[parent_parser],
                                                 help="Will run test_mushroom.")
    # test_housing command line command
    test_housing_parser = subparsers.add_parser('test_housing', parents=[parent_parser], help="Will run test_housing.")
    # write_data_set command line command
    write_data_set_parser = subparsers.add_parser('run_mlp', parents=[parent_parser],
                                                  help="Will run mlp using input parameters.")

    args = parser.parse_args()
    log.info('args: %s', str(args))

    exit_code = 1
    try:
        if args.command == 'run_mlp':
            start_timestamp = datetime.datetime.now()
            mlp = Mlp()
            # mlp.orig()
            # 'adagrad': adagrad_opt, 'adadelta': adadelta_opt
            test_optimizers = {'nadam': nadam_opt,
                               'adam': adam_opt}
            test_initializers = ['glorot_uniform',
                                 'glorot_normal',
                                 'he_uniform',
                                 'he_normal']
            test_activations = ['elu', 'relu', 'selu']
            # session_training_data_files = ['ann_test_data/ann_data-32x64-c.dat']    # 25k
            session_training_data_files = ['ann_test_data/ann_data-32x64-n.dat',
                                           'ann_test_data/ann_data-32x64-o.dat']  # 100k
            session_test_data_file = 'ann_test_data/ann_data-32x64-b.dat'
            replicate_final_model_file_root = 'model_search-%s' % start_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
            replicate_epochs = 150
            mlp.search_for_model(session_training_data_files,
                                 session_test_data_file,
                                 replicate_final_model_file_root,
                                 replicate_epochs,
                                 [3, 4, 5],
                                 optimizers_to_try=test_optimizers,
                                 activations_to_try=test_activations,
                                 initializers_to_try=test_initializers,
                                 num_replicates=5)

        exit_code = 0
    except MlpException as keras_regression_exception:
        log.exception(keras_regression_exception)
    except Exception as generic_exception:
        logging.exception(generic_exception)
    finally:
        logging.shutdown()
        sys.exit(exit_code)
