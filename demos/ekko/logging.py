import logging

logger = logging.getLogger('ekko')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(
    logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(lineno)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
)
logger.addHandler(ch)
