#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Example Keras MLP regression script with inference, reloading, re-fitting, and re-inferencing
    Based on https://www.learnopencv.com/deep-learning-using-keras-the-basics/
"""

# ToDo: reporting, tensorboard

import sys
import argparse
import logging
import random
import datetime
import pandas
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import CSVLogger, TensorBoard

__author__ = 'hhemken'

log = logging.getLogger(__file__)
rand = random.SystemRandom()
# ======================================================================================================================
# https://keras.io/optimizers/
AVAILABLE_OPTIMIZERS = ['adam',
                        'rmsprop',
                        'adamax',
                        'nadam',
                        'adagrad',
                        'adadelta',
                        'sgd']

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

    def __init__(self, num_inputs, num_outputs):
        """

        """
        self.x_train = None
        self.y_train = None
        self.x_validate = None
        self.y_validate = None
        self.x_test = None
        self.y_test = None
        self.inputs = num_inputs
        self.outputs = num_outputs
        self.hidden_layers = None
        self.model = None
        self.num_examples = None
        self.num_features = None
        self.train_data_file_path = None
        self.validation_data_file_path = None
        self.test_data_file_path = None

    def create_model(self,
                     activation='relu',
                     initializer='glorot_normal',
                     hidden_layer_sizes=None,
                     optimizer='adam'):
        """
        
        :return: 
        """
        self.hidden_layers = hidden_layer_sizes
        log.info('%s', '=' * 80)
        log.info('create model')
        self.model = Sequential()
        log.info('create input layer')
        self.model.add(
            Dense(self.inputs, input_shape=(self.inputs,), kernel_initializer=initializer, activation=activation))
        layer_cnt = 1
        log.info('add hidden layers     %s', str(self.hidden_layers))
        for hidden_layer_size in self.hidden_layers:
            log.info('create hidden layer %d', layer_cnt)
            self.model.add(Dense(hidden_layer_size,
                                 kernel_initializer=initializer,
                                 activation=activation))
            layer_cnt += 1

        log.info('create output layer')
        self.model.add(
            Dense(self.outputs, input_shape=(self.outputs,), kernel_initializer=initializer, activation=activation))
        log.info('compile network')
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae', 'acc'])

    def fit_model(self, epochs_to_run=EPOCHS_TO_RUN, final_model_file_root=None, session_name=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")):
        """

        :return:
        """
        callbacks = list()
        # tensorboard --logdir=./tensorboard/logs/
        tb_callback = TensorBoard(log_dir='./tensorboard/logs/{}'.format(session_name),
                                  # histogram_freq=250,  # https://github.com/aurora95/Keras-FCN/issues/50
                                  histogram_freq=0,
                                  batch_size=32,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=True,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)
        callbacks.append(tb_callback)
        if final_model_file_root:
            csv_logger = CSVLogger('%s-epochs.csv' % final_model_file_root)
            callbacks.append(csv_logger)
        log.info('%s', '=' * 80)
        log.info('fit')
        log.info('session_name:     %s', session_name)
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
        log.info('history:          %s', dir(history))
        # log.info('history.epoch:   %s', str(history.epoch))
        # log.info('history.history: %s', str(history.history))
        log.info('history.on_train_begin: %s', dir(history.on_train_begin))
        log.info('history.on_train_end: %s', dir(history.on_train_end))
        log.info('history.on_batch_begin: %s', dir(history.on_batch_begin))
        log.info('history.on_batch_end: %s', dir(history.on_batch_end))
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

    def load_csv(self, data_file_path, comma_separated):
        """

        :return:
        """
        whitespace_separated = False if comma_separated else True
        log.info('load dataframe from csv file %s', data_file_path)
        dataframe = pandas.read_csv(data_file_path, delim_whitespace=whitespace_separated, header=None)
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

    def continue_training(self,
                          training_data_files,
                          test_data_file,
                          comma_separated,
                          initial_model_file,
                          session_name,
                          epochs_to_run=EPOCHS_TO_RUN):
        """

        :return:
        """
        # ---------------------------------------------------------------------------------------------------------------
        rand.shuffle(training_data_files)
        log.info('load model %s', initial_model_file)
        self.load_model(initial_model_file)
        cnt = 1
        for data_file in training_data_files:
            log.info('%s', '=' * 80)
            log.info('training data     %s', data_file)
            log.info('training round    %d/%d', cnt, len(training_data_files))
            self.x_train, self.y_train = self.load_csv(data_file, comma_separated)
            log.info('test data         %s', test_data_file)
            self.x_test, self.y_test = self.load_csv(test_data_file, comma_separated)
            self.num_examples = self.x_train.shape[0]
            self.num_features = self.x_train.shape[1]

            log.info('x_train shape: %s', self.x_train.shape)
            log.info('y_train shape: %s', self.y_train.shape)
            log.info('x_test shape:  %s', self.x_test.shape)
            log.info('y_test shape:  %s', self.y_test.shape)
            log.info('num examples:  %d', self.num_examples)
            log.info('num features:  %d', self.num_features)

            self.fit_model(epochs_to_run=epochs_to_run, session_name=session_name)
            self.test_model()
            self.save_model(session_name + '-%02d' % cnt)
            cnt += 1
        # ---------------------------------------------------------------------------------------------------------------

    def initial_training(self,
                         optimizer,
                         activation,
                         initializer,
                         hidden_layer_sizes,
                         training_data_files,
                         test_data_file,
                         comma_separated,
                         session_name,
                         epochs_to_run=EPOCHS_TO_RUN):
        """

        :return:
        """
        # ---------------------------------------------------------------------------------------------------------------
        rand.shuffle(training_data_files)
        log.info('%s', '=' * 80)
        self.create_model(optimizer=optimizer,
                          activation=activation,
                          initializer=initializer,
                          hidden_layer_sizes=hidden_layer_sizes)
        cnt = 1
        for data_file in training_data_files:
            log.info('%s', '-' * 80)
            log.info('training data     %s', data_file)
            log.info('training round    %d/%d', cnt, len(training_data_files))
            self.x_train, self.y_train = self.load_csv(data_file, comma_separated)
            log.info('test data         %s', test_data_file)
            self.x_test, self.y_test = self.load_csv(test_data_file, comma_separated)
            self.num_examples = self.x_train.shape[0]
            self.num_features = self.x_train.shape[1]

            log.info('x_train shape: %s', self.x_train.shape)
            log.info('y_train shape: %s', self.y_train.shape)
            log.info('x_test shape:  %s', self.x_test.shape)
            log.info('y_test shape:  %s', self.y_test.shape)
            log.info('num examples:  %d', self.num_examples)
            log.info('num features:  %d', self.num_features)

            self.fit_model(epochs_to_run=epochs_to_run, session_name=session_name)
            self.test_model()
            self.save_model(session_name)
            cnt += 1
        # ---------------------------------------------------------------------------------------------------------------

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


def test_housing():
    """

    :return:
    """
    num_inputs = 13
    num_outputs = 1
    train_data_file_path = 'test_datasets/housing-train.csv'
    validation_data_file_path = ''
    test_data_file_path = 'test_datasets/housing-test.csv'
    mushroom_mlp = Mlp(num_inputs, num_outputs)
    mushroom_mlp.initial_training('nadam',
                                  'relu',
                                  'he_uniform',
                                  [6],
                                  [train_data_file_path],
                                  test_data_file_path,
                                  'housing-nadam-relu-he_uniform-150-75',
                                  epochs_to_run=100)


def test_mushroom():
    """

    :return:
    """
    num_inputs = 125
    num_outputs = 2
    train_data_file_path = 'test_datasets/mushroom_train-output_last.csv'
    validation_data_file_path = ''
    test_data_file_path = 'test_datasets/mushroom_test-output_last.csv'
    mushroom_mlp = Mlp(num_inputs, num_outputs)
    mushroom_mlp.initial_training('nadam',
                                  'relu',
                                  'he_uniform',
                                  [150, 75],
                                  [train_data_file_path],
                                  test_data_file_path,
                                  False,
                                  'mushroom-nadam-relu-he_uniform-150-75',
                                  epochs_to_run=50)


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
    parent_parser.add_argument('--num_inputs',
                               help="Dimension of input to ANN.")
    parent_parser.add_argument('--num_outputs',
                               help="Dimension of output from ANN.")
    parent_parser.add_argument('--optimizer', default='adam',
                               help="Optimizer to use (default adam).")
    parent_parser.add_argument('--activation', default='relu',
                               help="Activation to use (default relu).")
    parent_parser.add_argument('--initializer', default='glorot_normal',
                               help="Initializer to use (default glorot_normal).")
    parent_parser.add_argument('--hidden_layer_sizes',
                               help="List of hidden layer sizes in JSON format.")
    parent_parser.add_argument('--training_data_files',
                               help="List of training files to use in JSON format.")
    parent_parser.add_argument('--test_data_file',
                               help="Path to data file to be used for testing.")
    parent_parser.add_argument('--model_file_root',
                               help="File name root of saved model to load for continued training, "
                                    "excluding '.h5' extension.")
    parent_parser.add_argument('--epochs_to_run',
                               help="Number of epochas to run per data file.")
    parent_parser.add_argument('--comma_separated', default=False,
                               help="Whether input data are comma-separated as opposed to space separated.")
    parent_parser.add_argument('--session_name', default=False,
                               help="Unique name to track session results, e.g. with Tensorboard.")
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='command')
    # test_mushroom command line command
    test_mushroom_parser = subparsers.add_parser('create', parents=[parent_parser],
                                                 help="Will run test_mushroom.")
    # test_housing command line command
    test_housing_parser = subparsers.add_parser('test_housing', parents=[parent_parser], help="Will run test_housing.")
    # test_housing command line command
    test_mushroom_parser = subparsers.add_parser('test_mushroom', parents=[parent_parser],
                                                 help="Will run test_mushroom.")
    # continue_training command line command
    write_data_set_parser = subparsers.add_parser('initial_training', parents=[parent_parser],
                                                  help="Create new model and start training.")
    # continue_training command line command
    write_data_set_parser = subparsers.add_parser('continue_training', parents=[parent_parser],
                                                  help="Load existing model and continue training.")

    args = parser.parse_args()
    log.info('%s', str(args))

    exit_code = 1
    try:
        log.info('commansd is %s', args.command)
        if args.command == 'continue_training':
            mlp = Mlp(int(args.num_inputs), int(args.num_outputs))
            mlp.continue_training(json.loads(args.training_data_files),
                                  args.test_data_file,
                                  args.comma_separated,
                                  args.model_file_root,
                                  args.session_name,
                                  int(args.epochs_to_run))

        elif args.command == 'initial_training':
            mlp = Mlp(int(args.num_inputs), int(args.num_outputs))
            mlp.initial_training(args.optimizer,
                                 args.activation,
                                 args.initializer,
                                 json.loads(args.hidden_layer_sizes),
                                 json.loads(args.training_data_files),
                                 args.test_data_file,
                                 args.comma_separated,
                                 args.session_name,
                                 int(args.epochs_to_run))

        elif args.command == 'test_mushroom':
            test_mushroom()

        if args.command == 'test_housing':
            log.info("running test_housing")
            test_housing()

        exit_code = 0
    except MlpException as keras_regression_exception:
        log.exception(keras_regression_exception)
    except Exception as generic_exception:
        logging.exception(generic_exception)
    finally:
        logging.shutdown()
        sys.exit(exit_code)
