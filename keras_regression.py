"""
    from http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
"""
import numpy
import pandas
import time
import argparse
import logging
import sys
import json
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

__author__ = 'hhemken'
log = logging.getLogger(__file__)


class KerasRegressionException(Exception):
    """
    Generic exception to be raikeras_regression within the KerasRegression class.
    """

    def __init__(self, msg):
        log.exception(msg)
        super(KerasRegressionException, self).__init__(msg)


class KerasRegression(object):
    """
    """

    KFOLD_BATCH_SIZE = 10
    KFOLD_NUM_EPOCHS = 10
    KFOLD_NUM_SPLITS = 5

    def __init__(self, seed=42,
                 data_file_path=None,
                 output_dir='.',
                 num_ann_inputs=0,
                 num_ann_outputs=0,
                 hidden_layers=None,
                 session_name='anonymous_keras_run'):
        """

       :param seed:
       :param data_file_path:
       """
        self.inputs = num_ann_inputs
        self.outputs = num_ann_outputs
        self.data_file_path = data_file_path
        self.output_dir = output_dir
        self.hidden_layers = hidden_layers[:]
        self.session_name = session_name
        self.model = None
        self.seed = seed
        numpy.random.seed(self.seed)
        self.x = None
        self.y = None

    def model_creator(self):
        # create model
        self.model = Sequential()
        self.model.add(Dense(self.inputs, input_dim=self.inputs, kernel_initializer="normal", activation="relu"))
        for hidden_layer_size in self.hidden_layers:
            self.model.add(
                Dense(hidden_layer_size, input_dim=hidden_layer_size, kernel_initializer='normal', activation='relu'))
        # output layer
        self.model.add(Dense(self.outputs, input_dim=self.outputs, kernel_initializer="normal"))
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model

    def basic_cross_validation(self):
        """

        :return:
        """
        kfold = KFold(n_splits=self.KFOLD_NUM_SPLITS, random_state=self.seed)
        cvscores = list()
        model_cv_scores = list()
        split_num = 1
        for train, test in kfold.split(self.x, self.y):
            # create model
            self.model = Sequential()
            self.model.add(Dense(self.inputs, input_dim=self.inputs, kernel_initializer="normal", activation="relu"))
            for hidden_layer_size in self.hidden_layers:
                self.model.add(Dense(hidden_layer_size, input_dim=hidden_layer_size, kernel_initializer='normal',
                                     activation='relu'))
            # output layer
            self.model.add(Dense(self.outputs, input_dim=self.outputs, kernel_initializer="normal"))
            # Compile model
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # Fit the model
            self.model.fit(self.x[train], self.y[train], epochs=self.KFOLD_NUM_EPOCHS, batch_size=self.KFOLD_BATCH_SIZE,
                           verbose=1)
            # evaluate the model
            scores = self.model.evaluate(self.x[test], self.y[test], verbose=1)
            log.info("%s: %.2f%%", self.model.metrics_names[1], scores[1] * 100)
            current_cv_score = scores[1] * 100
            cvscores.append(current_cv_score)
            # save model to HDF5
            model_file_path = "{}/{}-split-{}-model.hdf5".format(self.output_dir, self.session_name, str(split_num))
            save_model(self.model, model_file_path)
            log.info("Saved model to %s", model_file_path)
            model_cv_scores.append((model_file_path, str(current_cv_score)))
            split_num += 1
        log.info("avg cross validation score %.2f%% (+/- %.2f%%)", numpy.mean(cvscores), numpy.std(cvscores))
        log.info("model     cv score")
        for model_file, cv_score in model_cv_scores:
            log.info("{} {}".format(model_file, cv_score))

    def run(self):
        """

        :return:
        """
        dataframe = pandas.read_csv(self.data_file_path, delim_whitespace=True, header=None)
        dataset = dataframe.values
        # split into input (X) and output (Y) variables
        self.x = dataset[:, 0:self.inputs]
        # Y data are at end of each row
        # notation: http://structure.usc.edu/numarray/node26.html
        if self.outputs > 1:
            # get columns self.inputs to self.inputs + self.outputs fom matrix of data
            self.y = dataset[:, self.inputs:self.inputs + self.outputs]
        else:
            # get column self.inputs from matrix of data
            self.y = dataset[:, self.inputs]
        self.basic_cross_validation()


def test_housing():
    """

    :return:
    """
    # load housing dataset
    inputs = 13
    outputs = 1
    hidden_layers = [6]
    random_seed = 7
    keras_reg = KerasRegression(seed=random_seed,
                                num_ann_inputs=inputs,
                                num_ann_outputs=outputs,
                                data_file_path="test_datasets/housing.csv",
                                hidden_layers=hidden_layers,
                                session_name='housing')
    keras_reg.run()


def test_mushroom():
    """

    :return:
    """
    # # load mushroom dataset
    inputs = 125
    outputs = 2
    hidden_layers = [150, 75]
    random_seed = 7
    keras_reg = KerasRegression(seed=random_seed,
                                num_ann_inputs=inputs,
                                num_ann_outputs=outputs,
                                data_file_path="test_datasets/mushroom_train-output_last.csv",
                                hidden_layers=hidden_layers,
                                session_name='mushroom')
    keras_reg.run()


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
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='command')
    # test_mushroom command line command
    test_mushroom_parser = subparsers.add_parser('test_mushroom', parents=[parent_parser],
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
        if args.command == 'test_mushroom':
            log.info("running test_mushroom")
            test_mushroom()
        if args.command == 'test_housing':
            log.info("running test_housing")
            test_housing()
        elif args.command == 'run_mlp':
            log.info("data file path       %s", args.data_file_path)
            log.info("output directory     %s", args.output_dir)
            log.info("num_ann_inputs       %s", args.num_ann_inputs)
            log.info("num_ann_outputs      %s", args.num_ann_outputs)
            log.info("session_name         %s", args.session_name)
            log.info("hidden_layers string %s", args.hidden_layers)
            actual_hidden_layers = json.loads(args.hidden_layers)
            log.info("hidden_layers        %s", str(actual_hidden_layers))
            keras_regression = KerasRegression(data_file_path=args.data_file_path,
                                               output_dir=args.output_dir,
                                               num_ann_inputs=int(args.num_ann_inputs),
                                               num_ann_outputs=int(args.num_ann_outputs),
                                               hidden_layers=actual_hidden_layers,
                                               session_name=args.session_name)
            keras_regression.run()

        exit_code = 0
    except KerasRegressionException as keras_regression_exception:
        log.exception(keras_regression_exception)
    except Exception as generic_exception:
        logging.exception(generic_exception)
    finally:
        logging.shutdown()
        sys.exit(exit_code)
