import logging.handlers
import pandas as pd


def add_click_options(options):
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


def load_csv(data_file_path, cutoff_between_0_1=False):
    data = list()
    with open(data_file_path, "r") as f:
        for line_index, line in enumerate(f):
            fields = line.strip().split(',')
            for field_index, field in enumerate(fields):
                field = float(field)
                if cutoff_between_0_1 and field < 0.0:
                    fields[field_index] = 0.0
                elif cutoff_between_0_1 and field > 1.0:
                    fields[field_index] = 1.0
                else:
                    fields[field_index] = field
            data.append(fields)
    return pd.DataFrame(data)


def setup_logging(
        log_level=logging.INFO,
        backup_count=25,
        max_bytes_per_file=20 * 1024 * 1024,
        log_file_path="mlp.log",
):
    """
    Set up a cumulative rotating log file for all of the logger calls.

    :param log_level: python logging level, e.g. logging.INFO or
        logging.DEBUG
    :type log_level: ``int``
    :param backup_count: number of rotating log files to preserve
    :type backup_count: ``int``
    :param max_bytes_per_file: maximum log file size in bytes
    :type max_bytes_per_file: ``int``
    :param log_file_path: full path to log file
    :type log_file_path: ``str``
    """
    local_log_format = (
            "%(asctime)s.%(msecs)03d [%(process)d] %(threadName)s: "
            + "%(levelname)-06s: %(module)s::%(funcName)s:%(lineno)s | "
            + "%(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    local_log_datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(
        level=log_level, format=local_log_format, datefmt=local_log_datefmt
    )
    # cumulative rotating log file
    formatter = logging.Formatter(local_log_format, datefmt=date_format)
    log_level = log_level
    handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=max_bytes_per_file, backupCount=backup_count
    )
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logging.getLogger("").addHandler(handler)
