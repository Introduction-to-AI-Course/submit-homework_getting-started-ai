#-*- coding utf8 -*-
"""
@author:
@brief:utils for os
"""

import os
import time
import shutil

def _get_signature():
    #to get pid and time
    pid = int(os.getpid())
    now = int(time.time())

    signature = "%d_%d" % (pid, now)
    return signature

def _create_dir(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def _remove_files(files):
    for fir in files:
        os.remove(fir)

def _remove_files(dirs):
    for dir in dirs:
        shutil.rmtree(dir)
