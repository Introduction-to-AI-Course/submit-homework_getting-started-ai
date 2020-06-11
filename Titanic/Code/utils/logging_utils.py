#-*- coding utf8 -*-
"""
@author:
@biref:utils for logging
"""

import os
import logging
import logging.handlers

def _get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)

    handlers = logging.handlers.RotatingFileHandler(
        Filename = os.path.join(logdir, logname),
        maxBytes=10*1024*1024,
        backupCount=10
    )

    logger = logging.getLogger("")
    logger.addHandler(handlers)
    logger.setLevel(loglevel)
    return logger