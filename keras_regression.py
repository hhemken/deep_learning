"""
    from http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
"""
import numpy
import pandas
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

__author__ = 'hhemken'
log = logging.getLogger(__file__)


class KerasRegression(object):
    """
    """

    def __init__(self, inputs, outputs, hidden_layers):
        """

        """
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers[:]
        self.model = None

    def model_creator(self):
        # create model
        self.model = Sequential()
        # self.model.add(Dense(self.inputs, input_dim=self.inputs, init='normal', activation='relu'))
        self.model.add(Dense(self.inputs, kernel_initializer="normal", activation="relu"))
        for hidden_layer_size in self.hidden_layers:
            self.model.add(Dense(hidden_layer_size, init='normal', activation='relu'))
        # self.model.add(Dense(self.outputs, init='normal'))
        self.model.add(Dense(self.outputs, kernel_initializer="normal"))
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model

    def run(self, seed, data_file_path):
        """

        :param seed:
        :param data_file_path:
        :return:
        """
        numpy.random.seed(seed)
        estimators = []
        dataframe = pandas.read_csv(data_file_path, delim_whitespace=True, header=None)
        dataset = dataframe.values
        # split into input (X) and output (Y) variables
        X = dataset[:, 0:self.inputs]
        if self.outputs > 1:
            Y = dataset[:, self.inputs:self.inputs + self.outputs]
        else:
            Y = dataset[:, self.inputs]
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(build_fn=self.model_creator, nb_epoch=50, batch_size=5, verbose=1)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=10, random_state=seed)
        results = cross_val_score(pipeline, X, Y, cv=kfold)
        log.info("\nResult: %.2f (%.2f) MSE" % (results.mean(), results.std()))
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")


def test_housing():
    """

    :return:
    """
    # load housing dataset
    inputs = 13
    outputs = 1
    hidden_layers = [6]
    keras_reg = KerasRegression(inputs, outputs, hidden_layers)
    random_seed = 7
    keras_reg.run(random_seed, "housing.csv")


def test_trade_forecast():
    """

    :return:
    """
    inputs = 10
    outputs = 2
    hidden_layers = [10, 10, 10, 10, 10, 10]
    keras_reg = KerasRegression(inputs, outputs, hidden_layers)
    random_seed = time.time()
    keras_reg.run(random_seed, "datasets/trade_forecast/tickers-12.dat")


def test_mushroom():
    """

    :return:
    """
    # # load mushroom dataset
    inputs = 125
    outputs = 2
    hidden_layers = [150, 75]
    keras_reg = KerasRegression(inputs, outputs, hidden_layers)
    random_seed = 7
    keras_reg.run(random_seed, "mushroom_train-output_last.csv")


if __name__ == '__main__':
    quote_data_log_format = ("%(asctime)s.%(msecs)03d %(levelname)-06s: " +
                             "%(module)s::%(funcName)s:%(lineno)s: %(message)s")
    quote_data_log_datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(level=logging.DEBUG,
                        format=quote_data_log_format,
                        datefmt=quote_data_log_datefmt)

    test_trade_forecast()
