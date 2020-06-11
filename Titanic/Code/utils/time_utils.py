#-*- coding utf8 -*-
"""
@author:
@biref:utils for time
"""

import time
import datetime

def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M")
    return now_str
