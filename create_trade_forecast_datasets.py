#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Describe what this module does

Unit tests and useful commands:

./module_template.py my_function arf
./module_template.py my_function --double arf
./module_template.py load_data arf
./module_template.py write_data_set arf
./module_template.py load_data --data_file_path xxx arf
./module_template.py write_data_set --write_ticker_files  arf

"""
import json
import codecs
import time
import re
import datetime
import random
import logging
import sys
import os
import glob
import argparse
import numpy as np

# from Tintest.System.Tintri.Realstore import Realstore

__author__ = 'hhemken'
log = logging.getLogger(__file__)


class TradeForecastDataException(Exception):
    """
    Generic exception to be raitrade_forecast_data within the TradeForecastData class.
    """

    def __init__(self, msg):
        log.exception(msg)
        super(TradeForecastDataException, self).__init__(msg)


class TradeForecastData(object):
    """
    Describe what TradeForecastData is for
    """

    A_CLASS_CONSTANT = 'Activation in progress'
    ANOTHER_CLASS_CONSTANT = 30

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.ticker_data = None
        self.output_dir = None

    def load_data(self, data_file_path):
        """
        load the data file into an instance variable
        :param data_file_path Describe my_arg
        :return None
        """
        log.info('loading file "%s"', data_file_path)
        with open(data_file_path, 'r') as f:
            self.ticker_data = json.load(f)
            # log.info('%s', str(self.ticker_data))

    def load_data_from_dir(self, data_dir):
        """
        Loads daily percent change, as written by write_ticker_files()

            JSON with each day's percent change:
            {
                "2000-08-21": 0.49591842420527166,
                "2000-08-22": 2.0989463786952878,
                "2000-08-23": 5.527296466556056,
            ...
                "2017-08-18": -0.22805017103763692
            }

        :return:
        """
        self.ticker_data = dict()
        # for data_file in os.scandir(path=data_dir):
        for data_file in os.listdir(data_dir):
            log.info('data file %s', data_file)
            # log.info('file name %s', os.path.basename(data_file.path))
            ticker, file_extension = os.path.splitext(os.path.basename(data_file))
            log.info('ticker         %s, file_extension %s', ticker, file_extension)
            self.ticker_data[ticker] = dict()
            data_file_path = data_dir + '/' + data_file
            log.info('data_file_path %s', data_file_path)
            with open(data_file_path, 'r') as f:
                pct_change_data = json.load(f)
                log.info('%s has %d entries', data_file, len(pct_change_data))
                self.ticker_data[ticker]['data'] = dict()
                for date_str in pct_change_data:
                    self.ticker_data[ticker]['data'][date_str] = dict()
                    self.ticker_data[ticker]['data'][date_str]['change'] = pct_change_data[date_str]

    def window(self, iterable, size=8):
        i = iter(iterable)
        win = list()
        for e in range(0, size):
            win.append(next(i))
        if len(win) == size:
            yield win
        for e in i:
            win = win[1:] + [e]
            if len(win) == size:
                yield win

    def write_ticker_files(self, output_dir):
        """
        write out the percent change value keyed by date to output files by ticker name
        :param output_dir Describe my_arg
        :return None
        """
        self.output_dir = output_dir
        log.info('output_dir %s', self.output_dir)
        counts = list()
        for ticker in self.ticker_data:
            log.info('----------' * 8)
            output_file_path = "{}/{}.json".format(self.output_dir, ticker)
            log.info('    output file %s', output_file_path)
            output_data = dict()
            cnt = 0
            for date_str in sorted(self.ticker_data[ticker]['data'].keys()):
                value = self.ticker_data[ticker]['data'][date_str]['change']
                if value == float('Inf'):
                    value = 0.0
                output_data[date_str] = value
                cnt += 1
            log.info('    rows %d', cnt)
            counts.append(cnt)
            with open(output_file_path, 'wb') as f:
                json.dump(output_data, codecs.getwriter('utf-8')(f), ensure_ascii=False, indent=4, sort_keys=True)
        np_counts = np.array(counts)
        log.info('----------' * 8)
        log.info('%d tickers', len(counts))
        log.info('avg counts: %f +/- %f', np.mean(np_counts), np.std(np_counts))
        log.info('    sum:      %d', np.sum(np_counts))
        log.info('    max: %d', np.max(np_counts))
        log.info('    min: %d', np.min(np_counts))
        log.info('    10:  %d', np.percentile(np_counts, 10))
        log.info('    20:  %d', np.percentile(np_counts, 20))
        log.info('    30:  %d', np.percentile(np_counts, 30))
        log.info('    40:  %d', np.percentile(np_counts, 40))
        log.info('    50:  %d', np.percentile(np_counts, 50))
        log.info('    60:  %d', np.percentile(np_counts, 60))
        log.info('    70:  %d', np.percentile(np_counts, 70))
        log.info('    80:  %d', np.percentile(np_counts, 80))
        log.info('    90:  %d', np.percentile(np_counts, 90))
        log.info('    95:  %d', np.percentile(np_counts, 95))

    def write_data_set(self, output_dir, num_ann_inputs, num_ann_outputs):
        """
        write out the percent change value keyed by date to output files by ticker name
        :param output_dir Describe my_arg
        :return None
        """
        self.output_dir = output_dir
        log.info('output_dir %s', self.output_dir)
        row_counts = list()
        window_counts = list()
        window_size = int(num_ann_inputs) + int(num_ann_outputs)
        log.info('    window size %d', window_size)
        output_file_path = "{}/tickers-{}.dat".format(self.output_dir, window_size)
        log.info('    output file %s', output_file_path)
        with open(output_file_path, 'w') as f:
            for ticker in self.ticker_data:
                log.info('----------' * 8)
                log.info('%s', ticker)
                output_data = list()
                cnt = 0
                for date_str in sorted(self.ticker_data[ticker]['data'].keys()):
                    value = self.ticker_data[ticker]['data'][date_str]['change']
                    if value == float('Inf'):
                        value = 0.0
                    output_data.append(value)
                    cnt += 1
                log.info('    rows        %d', cnt)
                row_counts.append(cnt)
                win_cnt = 0
                for each in self.window(output_data, size=16):
                    f.write(" ".join([str(number) for number in each]) + '\n')
                    win_cnt += 1
                log.info('    windows     %d', win_cnt)
                window_counts.append(win_cnt)


def my_function(my_arg):
    """
    Describe what my_function does
    :param my_arg Describe my_arg
    :return None
    """
    # do something outside of the class
    log.info('received "%s"', my_arg)


if __name__ == '__main__':
    tintri_log_format = ("%(asctime)s.%(msecs)03d [%(process)d] %(threadName)s: %(levelname)-06s: " +
                         "%(module)s::%(funcName)s:%(lineno)s: %(message)s")
    tintri_log_datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(level=logging.DEBUG,
                        format=tintri_log_format,
                        datefmt=tintri_log_datefmt)

    parent_parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('required_option', help='An option that always needs to be specified on the command line')
    parent_parser.add_argument('--data_file_path', default=False,
                               help="Path of JSON data file.")
    parent_parser.add_argument('--data_dir', default=False,
                               help="Directory with JSON data files.")
    parent_parser.add_argument('--output_dir', default=False,
                               help="Directory in which output files will be written.")
    parent_parser.add_argument('--num_ann_inputs', default=False,
                               help="Have another option on the command line.")
    parent_parser.add_argument('--num_ann_outputs', default=False,
                               help="Have another option on the command line.")
    parent_parser.add_argument('--num_input_files', default=False,
                               help="Have another option on the command line.")
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='command')
    # load_data command line command
    load_data_parser = subparsers.add_parser('load_data', parents=[parent_parser], help="Will run load_data.")
    # load_data command line command
    load_data_from_dir_parser = subparsers.add_parser('load_data_from_dir', parents=[parent_parser],
                                                      help="Will run load_data from a directory.")
    # write_data_set command line command
    write_data_set_parser = subparsers.add_parser('write_data_set', parents=[parent_parser],
                                                  help="Will run write_data_set.")

    # # my_function command line command
    # my_function_parser = subparsers.add_parser('my_function', help="Add, repurpose, or unlock a drive.")
    # my_function_parser.add_argument('--double', default=False, action='store_true', help="???")

    args = parser.parse_args()
    log.info('args: %s', str(args))
    test_trade_forecast_data_util = TradeForecastData()

    exit_code = 1
    try:
        if args.command == 'load_data':
            if args.data_file_path:
                log.info("data file path   %s", args.data_file_path)
                log.info("output directory %s", args.output_dir)
                test_trade_forecast_data_util.load_data(args.data_file_path)
                test_trade_forecast_data_util.write_ticker_files(args.output_dir)
            else:
                raise TradeForecastDataException("missing data file path")
        elif args.command == 'load_data_from_dir':
            # JSON with each day's percent change
            # {
            #     "2000-08-21": 0.49591842420527166,
            #     "2000-08-22": 2.0989463786952878,
            #     "2000-08-23": 5.527296466556056,
            # ...
            #     "2017-08-18": -0.22805017103763692
            # }
            if args.data_dir:
                log.info("data directory   %s", args.data_dir)
                test_trade_forecast_data_util.load_data_from_dir(args.data_dir)
            else:
                raise TradeForecastDataException("missing data directory")
            if args.output_dir:
                log.info("output directory %s", args.output_dir)
                test_trade_forecast_data_util.write_data_set(args.output_dir, args.num_ann_inputs, args.num_ann_outputs)
            else:
                raise TradeForecastDataException("missing output directory")
        elif args.command == 'write_data_set':
            log.info("data file path   %s", args.data_file_path)
            log.info("output directory %s", args.output_dir)
            log.info("num_ann_inputs   %s", args.num_ann_inputs)
            log.info("num_ann_outputs  %s", args.num_ann_outputs)
            test_trade_forecast_data_util.load_data(args.data_file_path)
            test_trade_forecast_data_util.write_data_set(args.output_dir, args.num_ann_inputs, args.num_ann_outputs)
        elif args.command == 'my_function':  # call the my_function module function
            if args.double:
                my_function(args.required_option + " " + args.required_option)
                log.info("invoked my_function() with 2x %s", args.required_option)
            else:
                my_function(args.required_option)
                log.info("invoked my_function() with %s", args.required_option)

        exit_code = 0
    except TradeForecastDataException as trade_forecast_data_exception:
        log.exception(trade_forecast_data_exception)
    except Exception as generic_exception:
        logging.exception(generic_exception)
    finally:
        logging.shutdown()
    sys.exit(exit_code)
