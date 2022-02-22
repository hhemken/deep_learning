from pathlib import Path
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import click
import datetime
import glob
import json
import logging.handlers
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import pandas as pd
import re
import secrets
import sys
import tensorflow.keras
import time

from utils import setup_logging, add_click_options, load_csv

log = logging.getLogger(__file__)


class Ann(object):

    def __init__(self,
                 # https://keras.io/api/layers/activations/
                 # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-elu-with-keras.md
                 # https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6
                 # elu allows for negative inputs
                 activation='elu',
                 batch_size=128,
                 epochs_to_run=100,
                 hidden_layer_sizes=None,
                 loss="mse",
                 metrics=None,
                 number_of_inputs=None,
                 number_of_outputs=None,
                 optimizer='rmsprop',
                 training_fraction=0.95,
                 validation_split=0.1,
                 verbose=2,
                 ):
        # incoming parameters
        self.activation = activation
        self.batch_size = batch_size
        self.epochs_to_run = epochs_to_run
        self.hidden_layer_sizes = hidden_layer_sizes
        self.loss = loss
        self.metrics = metrics
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.optimizer = optimizer
        self.training_fraction = training_fraction
        self.validation_split = validation_split
        self.verbose = verbose
        # internal attributes
        self.data_inputs = None
        self.data_outputs = None
        self.model = None
        self.run_data_file_path = None
        self.x_test = None
        self.y_test = None

    def continue_training(self, data_file_path, model_name):
        self.run_data_file_path = data_file_path
        self.extract_data_sets(self.run_data_file_path)
        self.load_saved_model(model_name)
        return self.fit_model(model_name)

    def create_model(self):
        input_layer = Input(shape=(self.data_inputs.shape[1],))
        layer_cnt = 1
        previous_layer = input_layer
        for hidden_layer_size in self.hidden_layer_sizes:
            log.info(f'create hidden layer {layer_cnt}')
            # ToDo per layer activation functions?
            new_layer = Dense(
                hidden_layer_size,
                activation=self.activation
            )(previous_layer)
            previous_layer = new_layer
            layer_cnt += 1
        output = Dense(self.data_outputs.shape[1], )(previous_layer)
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics)
        self.model.summary()

    def extract_data_sets(self, data_file_path):
        input_data = load_csv(data_file_path)
        log.info('split into input (X) and output (Y) variables')
        log.info(f'{self.number_of_inputs} inputs and {self.number_of_outputs} outputs from {data_file_path}')
        x = pd.DataFrame(input_data.iloc[:, 0:self.number_of_inputs].values)
        y = pd.DataFrame(
            input_data.iloc[:, self.number_of_inputs:self.number_of_inputs + self.number_of_outputs].values)
        self.data_inputs = x.sample(frac=self.training_fraction)
        self.data_outputs = y.sample(frac=self.training_fraction)
        self.x_test = x.drop(self.data_inputs.index)
        self.y_test = y.drop(self.data_outputs.index)

    def fit_model(self, model_name):
        log.info(f'training data file: {self.run_data_file_path}')
        history = self.model.fit(
            self.data_inputs,
            self.data_outputs,
            batch_size=self.batch_size,
            epochs=self.epochs_to_run,
            verbose=self.verbose,
            validation_split=self.validation_split)
        log.info(f'save model {model_name}.h5')
        self.model.save(f'{model_name}.h5')
        # log.info('history epoch {}'.format(history.epoch))
        # log.info('history       {}'.format(history.history))
        log.info('evaluate the model')
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        log.info('scores')
        log.info(f'{self.model.metrics_names[1]}: {scores[1] * 100:.2f}%%')
        return history

    def inference(self, x_data):
        calculated_y = self.model.predict(x_data)
        return calculated_y

    def initial_training(self, data_file_path, model_name):
        self.run_data_file_path = data_file_path
        self.extract_data_sets(self.run_data_file_path)
        self.create_model()
        return self.fit_model(model_name)

    def load_saved_model(self, model_name):
        log.info('=' * 80)
        log.info(f'loading model    {model_name}.h5')
        self.model = load_model(f'{model_name}.h5')
        log.info('loaded model summary:')
        self.model.summary()
        log.info(f'loaded model     {model_name}.h5')

    def save_model(self, model_name):
        log.info('=' * 80)
        log.info(f'save model {model_name}.h5')
        self.model.save(f'{model_name}.h5')
        log.info(self.model.summary())

    def test_model(self):
        calculated_y = self.inference(self.x_test)
        log.info('y_test:')
        # log.info(self.y_test - 0.5)
        log.info("Predictions:")
        log.info(f'{calculated_y}')
        mae = tensorflow.keras.losses.MeanAbsoluteError()
        abs_error = mae(self.y_test, calculated_y).numpy()
        log.info(f'Absolute Error:  {abs_error}')
        log.info(f'data_file_path:  {self.run_data_file_path}')


common_click_options = [
    click.option(
        f"--debug",
        default=False,
        is_flag=True,
        help="Set logging level to DEBUG.",
    ),
    click.option(
        f"--epochs",
        required=True,
        type=int,
        help="Path to file with input data.",
    ),
    click.option(
        f"--input_file",
        required=True,
        help="Path to file with input data.",
    ),
    click.option(
        f"--model_name",
        required=True,
        help="Root name of the model file to be saved or loaded.",
    ),
]

create_model_click_options = [
    click.option(
        f"--hidden_layer_sizes",
        required=True,
        help="A JSON string without embedded spaces representing the sizes of the consecutive hidden layers, e.g.: "
        "[64,64,64,64,64]'",
    ),
]

@click.group()
def cli():
    pass


@add_click_options(common_click_options)
@add_click_options(create_model_click_options)
@cli.command('create_model')
def create_model(**kwargs):
    setup_logging(
        log_file_path='ann-create_model.log',
        log_level=logging.INFO if not kwargs['debug'] else logging.DEBUG,
    )
    data_file_path = kwargs['input_file']
    hidden_layer_sizes = json.loads(kwargs['hidden_layer_sizes'])
    model_name = kwargs['model_name']
    number_of_inputs = 25
    number_of_outputs = 5
    ann = Ann(
        epochs_to_run=kwargs['epochs'],
        hidden_layer_sizes=hidden_layer_sizes,
        metrics=["mae"],
        number_of_inputs=number_of_inputs,
        number_of_outputs=number_of_outputs,
    )
    run_history = ann.initial_training(data_file_path, model_name)

    history_dict = run_history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "b", label="Training Loss")
    plt.plot(epochs, val_loss_values, "r", label="Validation Loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0.0, 0.01)
    plt.legend()
    plt.show()

    plt.clf()
    acc = history_dict["mae"]
    val_acc = history_dict["val_mae"]
    plt.plot(epochs, acc, "b", label="Training MAE")
    plt.plot(epochs, val_acc, "r", label="Validation MAE")
    plt.title("Training and validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.ylim(0.0, 0.05)
    plt.legend()
    plt.show()


@add_click_options(common_click_options)
@cli.command('continue_training')
def continue_training(**kwargs):
    setup_logging(
        log_file_path='ann-continue_training.log',
        log_level=logging.INFO if not kwargs['debug'] else logging.DEBUG,
    )
    data_file_path = kwargs['input_file']
    model_name = kwargs['model_name']
    number_of_inputs = 25
    number_of_outputs = 5
    ann = Ann(
        epochs_to_run=kwargs['epochs'],
        metrics=["mae"],
        number_of_inputs=number_of_inputs,
        number_of_outputs=number_of_outputs,
    )
    run_history = ann.continue_training(data_file_path, model_name)


if __name__ == "__main__":
    exit_code = 1
    try:
        cli()
        exit_code = 0
    except Exception as generic_exception:
        log.info("{}".format(generic_exception))
    finally:
        sys.exit(exit_code)
