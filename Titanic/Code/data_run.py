#-*- coding utf8 -*-
"""
@author:
@brief:run data and generate training testing data
"""

import os

#-------------------------------------------------
#generator splitter
cmd = "python splitter.py"
os.system(cmd)


#-------------------------------------------------
#generator stats feature
cmd = "python stats_feature.py"
os.system(cmd)

#-------------------------------------------------
#generate interaction feature
cmd = "python feature_interaction"
os.system(cmd)

#-------------------------------------------------
#generate matrice
cmd = "python feature_combination.py"
os.system(cmd)

