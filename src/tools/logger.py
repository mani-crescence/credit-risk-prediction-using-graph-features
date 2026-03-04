import logging, sys

logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                       format='%(asctime)s - %(levelname)s - %(message)s')
def my_exception_hook(type, value, tb):
    message = f'{type}\n{value}\n{tb}'
    logging.error(f'an unhandled error raised {message}. \n This error occurs inside cleaning.py file.')

sys.excepthook = my_exception_hook