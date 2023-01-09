#! /usr/bin/python3
import click
import logging.handlers
import matplotlib.pyplot as plt
import tensorflow.keras
# import pygad.kerasga
# import numpy
# import pygad
import pandas as pd
import sys
# import time

from tensorflow.keras.layers import Input, Dense  # , Activation, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# https://www.tensorflow.org/api_docs/python/tf/keras/losses
# from tensorflow.keras.losses import mean_squared_error
# from math import sqrt

import utils

log = logging.getLogger(__file__)

BATCH_SIZE_DIVISOR = 50
LOG_FILE_NAME = "mlp.log"


class Mlp(object):

    def __init__(self, data_file_path, num_inputs, num_outputs):
        self.data_file_path = data_file_path
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def scale(self, X, axis, x_min, x_max):
        """
        https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
        """
        nom = (X - X.min(axis=axis)) * (x_max - x_min)
        denom = X.max(axis=axis) - X.min(axis=0)
        denom[denom == axis] = 1
        return x_min + nom / denom

    def extract_data_sets(self):
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        input_data = pd.read_csv(self.data_file_path)
        log.info('split into input (X) and output (Y) variables')
        self.X = pd.DataFrame(input_data.iloc[:, 0:self.num_inputs].values)
        self.y = pd.DataFrame(input_data.iloc[:, self.num_inputs:self.num_inputs + self.num_outputs].values)
        log.info('X: {} rows, {} cols'.format(self.X.shape[0], self.X.shape[1]))
        log.info('y: {} rows, {} cols'.format(self.y.shape[0], self.y.shape[1]))

        data_inputs = self.X.sample(frac=0.95)
        data_outputs = self.y.sample(frac=0.95)
        self.X_test = self.X.drop(data_inputs.index)
        self.y_test = self.y.drop(data_outputs.index)
        log.info('data_inputs: {} rows, {} cols'.format(data_inputs.shape[0], data_inputs.shape[1]))
        log.info('data_outputs: {} rows, {} cols'.format(data_outputs.shape[0], data_outputs.shape[1]))
        log.info('self.X_test: {} rows, {} cols'.format(self.X_test.shape[0], self.X_test.shape[1]))
        log.info('self.y_test: {} rows, {} cols'.format(self.y_test.shape[0], self.y_test.shape[1]))
        return data_inputs, data_outputs, self.X_test, self.y_test

    def create_model(
            self,
            data_inputs,
            data_outputs,
            hidden_layer_sizes=None,
            loss="mse",
            metrics=None,
            optimizer='rmsprop',
    ):
        """

        :return:
        """
        hidden_layers = hidden_layer_sizes
        log.info('%s', '=' * 80)
        log.info('create model')
        # self.model = Sequential()
        previous_layer = Input(shape=(data_inputs.shape[1],))
        log.info('create input layer')
        layer_cnt = 1
        log.info('add hidden layers     %s', str(hidden_layers))
        for hidden_layer_size in hidden_layers:
            log.info('create hidden layer %d', layer_cnt)
            dense_layer = Dense(hidden_layer_size, activation='relu')(previous_layer)
            previous_layer = dense_layer
            layer_cnt += 1

        log.info('create output layer')
        output = Dense(data_outputs.shape[1], )(previous_layer)
        log.info('compile network')
        # self.model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
        self.model = Model(inputs=previous_layer, outputs=output)
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)
        self.model.summary()


    def load_saved_model(self, model_name):
        """

        :return:
        """
        log.info('%s', '=' * 80)
        log.info('loading model     %s.h5', model_name)
        model = load_model('%s.h5' % model_name)
        log.info('loaded model summary:')
        model.summary()
        log.info('loaded model      %s.h5', model_name)
        return model

    def save_model(self, model, model_name):
        """

        :return:
        """
        log.info('%s', '=' * 80)
        log.info('save model %s.h5', model_name)
        model.save('%s.h5' % model_name)
        log.info(model.summary())

    def run(self, run_data_file_path, run_data_inputs, run_data_outputs, hidden_layer_sizes=None,
            create_new_model=False, model_name='trained_weights', epochs_to_run=125, batch_size=128):

        if create_new_model:
            self.create_model(run_data_inputs, run_data_outputs, hidden_layer_sizes=hidden_layer_sizes)
        else:
            self.model = self.load_saved_model(model_name)

        log.info(f'training data file: {run_data_file_path}')
        run_history = self.model.fit(
            run_data_inputs,
            run_data_outputs,
            batch_size=batch_size,
            epochs=epochs_to_run,
            verbose=2,
            validation_split=0.1)
        log.info('save model %s.h5', model_name)
        self.model.save('%s.h5' % model_name)
        # log.info('history epoch {}'.format(run_history.epoch))
        # log.info('history       {}'.format(run_history.history))
        log.info('evaluate the model')
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        # log.info('scores')
        # log.info("%s: %.2f%%", self.model.metrics_names[0], scores[0] * 100)          <--

        predictions = self.model.predict(self.X_test)
        log.info('X_test:')
        log.info(self.X_test)
        log.info('y_test:')
        log.info(self.y_test)
        log.info("Predictions:\n{}".format(predictions))
        mae = tensorflow.keras.losses.MeanAbsoluteError()
        abs_error = mae(self.y_test, predictions).numpy()
        log.info("Absolute Error : {}".format(abs_error))
        log.info('data_file_path:  {}'.format(run_data_file_path))

        return run_history


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# click

click_options = [
    click.option(
        "--debug",
        default=False,
        is_flag=True,
        help="Set logging level to DEBUG.",
    ),
    click.option(
        '--data_file_path',
        required=True,
        help="Path of file to be processed.",
    ),
    click.option(
        '--output_path',
        help="Path of file with all literal dictionary keys converted to string constants.",
    ),
]


def add_options(options):
    """
    create a decorator containing a specific list of click options
    https://stackoverflow.com/questions/40182157/shared-options-and-flags-between-commands
    :param options: a list of defined click options
    :return: the decorator function
    """

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@click.group()
def cli():
    pass


@cli.command('test_petrol_consumption')
@add_options(click_options)
def test_petrol_consumption(**kwargs):
    LOG_FILE_NAME = "mlp-test_petrol_consumption.log"
    utils.setup_logging(log_file_path=LOG_FILE_NAME)
    mlp = Mlp(kwargs['data_file_path'], 4, 1)
    data_inputs, data_outputs, X_test, y_test = mlp.extract_data_sets()
    hidden_layer_sizes = [4, 4, 4]
    epochs_to_run = 50000
    history = mlp.run(kwargs['data_file_path'],
                  data_inputs,
                  data_outputs,
                  hidden_layer_sizes=hidden_layer_sizes,
                  create_new_model=True,
                  epochs_to_run=epochs_to_run
                  )


if __name__ == "__main__":
    exit_code = 1
    try:
        cli()
        exit_code = 0
    except Exception as generic_exception:
        log.exception("{}".format(generic_exception))
    finally:
        sys.exit(exit_code)
