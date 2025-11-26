import logging
import colorlog


def reset_console_logger(logger):

  # logger.handlers.clear()
  for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
      logger.removeHandler(handler)
  # logger.setLevel(logging.DEBUG)

  log_colors_config = {
      'DEBUG': 'cyan',  # cyan white
      'INFO': 'green',
      'WARNING': 'yellow',
      'ERROR': 'red',
      'CRITICAL': 'bold_red,bg_white',
  }
  file_formatter = logging.Formatter(
      fmt=
      '[%(asctime)s.%(msecs)03d] %(filename)s f:%(funcName)s l:%(lineno)d [%(levelname)s] : %(message)s',
      datefmt='%Y-%m-%d  %H:%M:%S')

  console_formatter = colorlog.ColoredFormatter(
      fmt=
      '%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s f:%(funcName)s l:%(lineno)d [%(levelname)s] : %(message)s',
      datefmt='%H:%M:%S',
      log_colors=log_colors_config)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)
  console_handler.setFormatter(console_formatter)

  logger.addHandler(console_handler)


def reset_file_logger(logger, log_path):
  for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
      logger.removeHandler(handler)

  file_formatter = logging.Formatter(
      fmt=
      '[%(asctime)s.%(msecs)03d] %(filename)s f:%(funcName)s l:%(lineno)d [%(levelname)s] : %(message)s',
      datefmt='%Y-%m-%d  %H:%M:%S')

  file_handler = logging.FileHandler(log_path)
  file_handler.setFormatter(file_formatter)
  file_handler.setLevel(logging.DEBUG)
  logger.addHandler(file_handler)


def get_logger(loggername=__name__, log_path=None):
  logger = logging.getLogger(loggername)
  logger.handlers.clear()
  logger.setLevel(logging.DEBUG)

  reset_console_logger(logger)

  if log_path:
    reset_file_logger(logger, log_path)

  return logger


def get_loguru_logger(loggername=__name__, log_path=None):
  from loguru import logger

  if log_path:
    logger.add(
        log_path,
        format=
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}"
    )
  return logger
